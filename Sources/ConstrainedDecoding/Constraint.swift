/// A byte-level language recogniser the decoder can steer a model with.
///
/// The contract is deliberately tiny: an initial state, a pure `advance` that
/// either accepts one byte or rejects it, and `isAccepting`. States are values
/// (`Hashable`), so the decoder can explore token candidates speculatively on
/// copies and memoise masks by state — no undo log, no shared mutable matcher.
///
/// Two properties are the implementer's responsibility, and the test suite
/// checks both for ``JSONSchemaConstraint``:
/// - **Liveness:** every state `advance` returns can still reach an accepting
///   state. Without it a decode can wander into a dead end.
/// - **Bound:** no accepted document is longer than `maxOutputBytes`.
public protocol DecodingConstraint: Sendable {
    associatedtype State: Hashable & Sendable

    var initialState: State { get }
    func advance(_ state: State, byte: UInt8) -> State?
    func isAccepting(_ state: State) -> Bool
    /// Upper bound on accepted output length, saturating at `Int.max`.
    var maxOutputBytes: Int { get }
    /// Bytes the vocabulary must provide as one-byte tokens (see
    /// ``CompiledSchema/uncoveredBytes(singleByteTokens:)``). Empty means "no requirement".
    func uncoveredBytes(singleByteTokens: Set<UInt8>) -> [UInt8]
}

// MARK: - JSON schema constraint

/// One frame of the matcher's pushdown stack. The *whole* stack is the state;
/// see ``MatcherState`` for why that matters to the mask cache.
enum Frame: Hashable, Sendable {
    /// Expecting the first byte of a value for `node`.
    case value(Int)
    /// Emitting literal `part` of an object node at byte `offset`.
    case object(node: Int, part: Int, offset: Int)
    /// Inside a string after its opening quote, with `length` characters so far.
    case string(node: Int, length: Int)
    /// Inside an integer. `digits == 0` means only a minus sign has been seen.
    case integer(node: Int, digits: Int, leadingZero: Bool)
    /// Inside `true` or `false` at byte `offset`.
    case keyword(isTrue: Bool, offset: Int)
    /// Inside an enumeration string after its opening quote.
    case enumeration(node: Int, prefix: [UInt8])
    /// Just after `[`.
    case arrayOpen(node: Int)
    /// After the `count`-th item of an array.
    case arrayNext(node: Int, count: Int)
}

/// The matcher state: the full pushdown stack.
///
/// Masks are memoised by this value, and the key must be the full stack. Two
/// states with the same *top* frame — a string inside an object versus the same
/// string at the root — accept different multi-byte tokens (`",` is legal in the
/// first, not the second). The test suite pins exactly that pair.
public struct MatcherState: Hashable, Sendable, CustomStringConvertible {
    var stack: [Frame]
    public var depth: Int { stack.count }
    public var description: String { "MatcherState(depth: \(stack.count), top: \(stack.last.map { "\($0)" } ?? "done"))" }
}

/// Enforces a ``Schema`` in its canonical compact encoding: no insignificant
/// whitespace, properties in declaration order. One canonical form per value is
/// what keeps the language finite and the state space small; the cost is that
/// the model cannot "think" in whitespace or reorder keys.
public struct JSONSchemaConstraint: DecodingConstraint {
    public let schema: CompiledSchema

    public init(_ schema: Schema) throws {
        self.schema = try CompiledSchema(schema)
    }

    public var initialState: MatcherState { MatcherState(stack: [.value(schema.root)]) }
    public var maxOutputBytes: Int { schema.maxEncodedBytes }

    public func uncoveredBytes(singleByteTokens: Set<UInt8>) -> [UInt8] {
        schema.uncoveredBytes(singleByteTokens: singleByteTokens)
    }

    public func isAccepting(_ state: MatcherState) -> Bool {
        guard let top = state.stack.last else { return true }
        // An integer has no closing delimiter: at the root, a complete integer is
        // a complete document.
        if state.stack.count == 1, case .integer(_, let digits, _) = top { return digits > 0 }
        return false
    }

    public func advance(_ state: MatcherState, byte: UInt8) -> MatcherState? {
        var next = state
        return step(&next.stack, byte) ? next : nil
    }

    /// Feeds one byte. The loop only repeats when a frame finishes *without*
    /// consuming the byte (an integer ends at the first non-digit; an array's
    /// first item starts at the byte after `[`), and each repeat either pops a
    /// frame or hands off to a `.value` frame that always returns — so it terminates.
    private func step(_ stack: inout [Frame], _ byte: UInt8) -> Bool {
        while let top = stack.last {
            let last = stack.count - 1
            switch top {
            case .value(let index):
                guard let node = schema.node(index) else { return false }
                switch node {
                case .object:
                    stack[last] = .object(node: index, part: 0, offset: 0)
                    continue
                case .string:
                    guard byte == JSONBytes.quote else { return false }
                    stack[last] = .string(node: index, length: 0)
                case .integer(_, let allowNegative):
                    if byte == JSONBytes.minus, allowNegative {
                        stack[last] = .integer(node: index, digits: 0, leadingZero: false)
                    } else if JSONBytes.isDigit(byte) {
                        stack[last] = .integer(node: index, digits: 1, leadingZero: byte == JSONBytes.zero)
                    } else {
                        return false
                    }
                case .boolean:
                    if byte == JSONBytes.trueBytes.first {
                        stack[last] = .keyword(isTrue: true, offset: 1)
                    } else if byte == JSONBytes.falseBytes.first {
                        stack[last] = .keyword(isTrue: false, offset: 1)
                    } else {
                        return false
                    }
                case .enumeration:
                    guard byte == JSONBytes.quote else { return false }
                    stack[last] = .enumeration(node: index, prefix: [])
                case .array:
                    guard byte == JSONBytes.openBracket else { return false }
                    stack[last] = .arrayOpen(node: index)
                }
                return true

            case .object(let index, let part, let offset):
                guard case .object(let literals, let children)? = schema.node(index),
                      literals.indices.contains(part) else { return false }
                let literal = literals[part]
                guard literal.indices.contains(offset), literal[offset] == byte else { return false }
                if offset + 1 < literal.count {
                    stack[last] = .object(node: index, part: part, offset: offset + 1)
                } else if children.indices.contains(part) {
                    stack[last] = .object(node: index, part: part + 1, offset: 0)
                    stack.append(.value(children[part]))
                } else {
                    stack.removeLast()
                }
                return true

            case .string(let index, let length):
                guard case .string(let minLength, let maxLength)? = schema.node(index) else { return false }
                if byte == JSONBytes.quote {
                    guard length >= minLength else { return false }
                    stack.removeLast()
                    return true
                }
                guard JSONBytes.isStringByte(byte), length < maxLength else { return false }
                stack[last] = .string(node: index, length: length + 1)
                return true

            case .integer(let index, let digits, let leadingZero):
                guard case .integer(let maxDigits, _)? = schema.node(index) else { return false }
                if JSONBytes.isDigit(byte), !leadingZero, digits < maxDigits {
                    stack[last] = .integer(node: index, digits: digits + 1,
                                           leadingZero: digits == 0 && byte == JSONBytes.zero)
                    return true
                }
                guard digits > 0 else { return false }
                stack.removeLast()
                continue // the integer is complete; the byte belongs to the enclosing frame

            case .keyword(let isTrue, let offset):
                let word = isTrue ? JSONBytes.trueBytes : JSONBytes.falseBytes
                guard word.indices.contains(offset), word[offset] == byte else { return false }
                if offset + 1 == word.count {
                    stack.removeLast()
                } else {
                    stack[last] = .keyword(isTrue: isTrue, offset: offset + 1)
                }
                return true

            case .enumeration(let index, let prefix):
                guard case .enumeration(let values)? = schema.node(index) else { return false }
                if byte == JSONBytes.quote {
                    guard values.contains(prefix) else { return false }
                    stack.removeLast()
                    return true
                }
                let extended = prefix + [byte]
                guard values.contains(where: { $0.starts(with: extended) }) else { return false }
                stack[last] = .enumeration(node: index, prefix: extended)
                return true

            case .arrayOpen(let index):
                guard case .array(let item, let minItems, let maxItems)? = schema.node(index) else { return false }
                if byte == JSONBytes.closeBracket {
                    guard minItems == 0 else { return false }
                    stack.removeLast()
                    return true
                }
                guard maxItems > 0 else { return false }
                stack[last] = .arrayNext(node: index, count: 1)
                stack.append(.value(item))
                continue // this byte starts the first item

            case .arrayNext(let index, let count):
                guard case .array(let item, let minItems, let maxItems)? = schema.node(index) else { return false }
                if byte == JSONBytes.closeBracket {
                    guard count >= minItems else { return false }
                    stack.removeLast()
                    return true
                }
                guard byte == JSONBytes.comma, count < maxItems else { return false }
                stack[last] = .arrayNext(node: index, count: count + 1)
                stack.append(.value(item))
                return true
            }
        }
        return false // the document is complete; nothing more may follow
    }
}

// MARK: - Unconstrained baseline

/// The baseline: every byte allowed, up to a length cap. Running the same
/// decoder with this constraint is what makes the demo's side-by-side honest —
/// the only variable is the mask.
public struct UnconstrainedText: DecodingConstraint {
    public let maxOutputBytes: Int

    /// Clamped to `0...SchemaLimits.maxEncodedBytes`, the same ceiling schemas get.
    public init(maxOutputBytes: Int) {
        self.maxOutputBytes = min(max(0, maxOutputBytes), SchemaLimits.maxEncodedBytes)
    }

    public var initialState: Int { 0 }
    public func advance(_ state: Int, byte: UInt8) -> Int? { state < maxOutputBytes ? state + 1 : nil }
    public func isAccepting(_ state: Int) -> Bool { true }
    public func uncoveredBytes(singleByteTokens: Set<UInt8>) -> [UInt8] { [] }
}
