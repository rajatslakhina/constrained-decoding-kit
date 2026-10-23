/// The subset of JSON Schema this engine can enforce token-by-token.
///
/// The subset is deliberately *bounded*: every string has a `maxLength`, every
/// integer a `maxDigits`, every array a `maxItems`, and every object a fixed set
/// of required properties. A bounded schema describes a finite language, which is
/// what lets ``ConstrainedDecoder`` promise termination regardless of what the
/// model wants to do (see ``CompiledSchema/maxEncodedBytes``).
///
/// Rejected alternative: optional properties, `additionalProperties`, unbounded
/// strings and free whitespace. Each one either makes the language infinite (a
/// model can emit spaces forever — the well-known whitespace runaway) or turns
/// the matcher into a set of parallel states, which multiplies mask cost.
public indirect enum Schema: Sendable, Hashable {
    /// An object whose properties are all required and emitted in declaration order.
    case object([Property])
    /// A string of printable ASCII, without escapes, of `minLength...maxLength` characters.
    case string(minLength: Int = 0, maxLength: Int)
    /// A JSON integer (no fraction, no exponent, no leading zeros) of at most `maxDigits` digits.
    case integer(maxDigits: Int, allowNegative: Bool = false)
    /// `true` or `false`.
    case boolean
    /// A string that must equal one of the listed values.
    case enumeration([String])
    /// An array of `minItems...maxItems` values of one schema.
    case array(of: Schema, minItems: Int = 0, maxItems: Int)

    public struct Property: Sendable, Hashable {
        public let name: String
        public let schema: Schema
        public init(_ name: String, _ schema: Schema) {
            self.name = name
            self.schema = schema
        }
    }
}

/// Hard limits that keep a compiled schema, and therefore every mask computation,
/// small and bounded. Compilation fails fast — before allocating past the
/// limit — rather than discovering the size afterwards.
public enum SchemaLimits {
    public static let maxDepth = 32
    public static let maxDigits = 18
    public static let maxStringLength = 4_096
    public static let maxItems = 1_024
    public static let maxProperties = 256
    public static let maxEnumerationValues = 256
    /// Total compiled nodes. Without it, a tiny `Schema` value with wide objects
    /// nested a few levels deep flattens into millions of nodes (each occurrence
    /// of a shared subtree is compiled separately) and is killed for memory.
    public static let maxNodes = 4_096
    /// Longest document a schema may describe (1 MiB). The termination
    /// argument says a decode ends within `maxEncodedBytes + 1` steps; this cap
    /// is what makes that number small enough to also bound memory. Without it,
    /// three nested nodes (`[[String(4096)] × 1024] × 1024`) describe 4 GB.
    public static let maxEncodedBytes = 1 << 20
}

public enum SchemaError: Error, Equatable, Sendable {
    case invalidBounds(path: String, detail: String)
    case emptyEnumeration(path: String)
    case duplicateEnumerationValue(path: String, value: String)
    case duplicateProperty(path: String, name: String)
    case emptyPropertyName(path: String)
    case unsupportedCharacter(path: String, text: String)
    case tooDeep(limit: Int)
    case tooMany(path: String, detail: String)
}

/// Overflow-free arithmetic for size bounds. Nested arrays multiply quickly
/// (`1024` items of `1024` items of ...), and a size bound that traps is worse
/// than one that saturates at `Int.max` and simply stops being the binding limit.
public enum Saturating {
    public static func add(_ a: Int, _ b: Int) -> Int {
        let (sum, overflow) = a.addingReportingOverflow(b)
        guard overflow else { return sum }
        return b > 0 ? Int.max : Int.min
    }

    public static func multiply(_ a: Int, _ b: Int) -> Int {
        let (product, overflow) = a.multipliedReportingOverflow(by: b)
        guard overflow else { return product }
        return (a > 0) == (b > 0) ? Int.max : Int.min
    }
}

/// Byte classes shared by the compiler and the matcher.
enum JSONBytes {
    static let quote: UInt8 = 0x22          // "
    static let backslash: UInt8 = 0x5C      // \
    static let comma: UInt8 = 0x2C          // ,
    static let colon: UInt8 = 0x3A          // :
    static let openBrace: UInt8 = 0x7B      // {
    static let closeBrace: UInt8 = 0x7D     // }
    static let openBracket: UInt8 = 0x5B    // [
    static let closeBracket: UInt8 = 0x5D   // ]
    static let minus: UInt8 = 0x2D          // -
    static let zero: UInt8 = 0x30
    static let nine: UInt8 = 0x39
    static let trueBytes: [UInt8] = Array("true".utf8)
    static let falseBytes: [UInt8] = Array("false".utf8)

    /// Printable ASCII except `"` and `\`: the only bytes allowed inside a string.
    /// Excluding escapes keeps string states tiny; the cost is that a field can
    /// never contain a quote, a backslash or non-ASCII text (documented limitation).
    static func isStringByte(_ byte: UInt8) -> Bool {
        byte >= 0x20 && byte <= 0x7E && byte != quote && byte != backslash
    }

    static func isDigit(_ byte: UInt8) -> Bool { byte >= zero && byte <= nine }
}

/// A schema lowered into a flat node table the matcher can index cheaply.
public struct CompiledSchema: Sendable, Hashable {
    enum Node: Sendable, Hashable {
        /// `literals.count == children.count + 1`: literal *i* is emitted before
        /// child *i*, and the last literal closes the object.
        case object(literals: [[UInt8]], children: [Int])
        case string(min: Int, max: Int)
        case integer(maxDigits: Int, allowNegative: Bool)
        case boolean
        case enumeration([[UInt8]])
        case array(item: Int, min: Int, max: Int)
    }

    let nodes: [Node]
    let root: Int
    /// Upper bound on the encoded length of any document this schema accepts,
    /// saturating at `Int.max`. Because every token is at least one byte, the
    /// decoder never needs more than `maxEncodedBytes + 1` steps (the `+ 1` is EOS).
    public let maxEncodedBytes: Int
    /// Bytes a vocabulary must offer as single-byte tokens for decoding to be
    /// dead-end free. See ``uncoveredBytes(singleByteTokens:)``.
    let requiredBytes: Set<UInt8>
    /// Whether some string must contain at least one character.
    let needsStringCharacter: Bool

    public init(_ schema: Schema) throws {
        var builder = Builder()
        let (root, size) = try builder.compile(schema, path: "$", depth: 0)
        guard size <= SchemaLimits.maxEncodedBytes else {
            throw SchemaError.tooMany(path: "$", detail: "documents up to \(size == Int.max ? "Int.max (saturated)" : String(size)) bytes; limit \(SchemaLimits.maxEncodedBytes)")
        }
        self.nodes = builder.nodes
        self.root = root
        self.maxEncodedBytes = size
        self.requiredBytes = builder.requiredBytes
        self.needsStringCharacter = builder.needsStringCharacter
    }

    func node(_ index: Int) -> Node? {
        nodes.indices.contains(index) ? nodes[index] : nil
    }

    /// A *sufficient* condition for "no reachable state is a dead end": every
    /// structural byte the grammar can demand is available as a one-byte token,
    /// so whatever a live state needs next, some token supplies it. Returns the
    /// missing bytes, sorted. Not *necessary* — a vocabulary could cover a byte
    /// with only multi-byte tokens in some contexts — so this errs on the side
    /// of refusing a vocabulary rather than discovering the gap mid-decode.
    public func uncoveredBytes(singleByteTokens: Set<UInt8>) -> [UInt8] {
        var missing = requiredBytes.subtracting(singleByteTokens)
        if needsStringCharacter,
           !singleByteTokens.contains(where: JSONBytes.isStringByte) {
            missing.insert(0x61) // representative: "a"
        }
        return missing.sorted()
    }

    private struct Builder {
        var nodes: [Node] = []
        var requiredBytes: Set<UInt8> = []
        var needsStringCharacter = false

        mutating func compile(_ schema: Schema, path: String, depth: Int) throws -> (Int, Int) {
            guard depth < SchemaLimits.maxDepth else { throw SchemaError.tooDeep(limit: SchemaLimits.maxDepth) }
            switch schema {
            case .object(let properties):
                guard properties.count <= SchemaLimits.maxProperties else {
                    throw SchemaError.tooMany(path: path, detail: "\(properties.count) properties")
                }
                var seen = Set<String>()
                var literals: [[UInt8]] = []
                var children: [Int] = []
                var size = 0
                for (index, property) in properties.enumerated() {
                    guard !property.name.isEmpty else { throw SchemaError.emptyPropertyName(path: path) }
                    guard property.name.utf8.allSatisfy(JSONBytes.isStringByte) else {
                        throw SchemaError.unsupportedCharacter(path: path, text: property.name)
                    }
                    guard seen.insert(property.name).inserted else {
                        throw SchemaError.duplicateProperty(path: path, name: property.name)
                    }
                    let opener: UInt8 = index == 0 ? JSONBytes.openBrace : JSONBytes.comma
                    let literal = [opener, JSONBytes.quote] + Array(property.name.utf8) + [JSONBytes.quote, JSONBytes.colon]
                    literals.append(literal)
                    let (child, childSize) = try compile(property.schema, path: "\(path).\(property.name)", depth: depth + 1)
                    children.append(child)
                    size = Saturating.add(size, Saturating.add(literal.count, childSize))
                }
                let closer: [UInt8] = properties.isEmpty ? [JSONBytes.openBrace, JSONBytes.closeBrace] : [JSONBytes.closeBrace]
                literals.append(closer)
                size = Saturating.add(size, closer.count)
                for literal in literals { requiredBytes.formUnion(literal) }
                return (try append(.object(literals: literals, children: children)), size)

            case .string(let minLength, let maxLength):
                guard minLength >= 0, maxLength >= minLength, maxLength <= SchemaLimits.maxStringLength else {
                    throw SchemaError.invalidBounds(path: path, detail: "string length \(minLength)...\(maxLength)")
                }
                requiredBytes.insert(JSONBytes.quote)
                if minLength > 0 { needsStringCharacter = true }
                return (try append(.string(min: minLength, max: maxLength)), Saturating.add(maxLength, 2))

            case .integer(let maxDigits, let allowNegative):
                guard maxDigits >= 1, maxDigits <= SchemaLimits.maxDigits else {
                    throw SchemaError.invalidBounds(path: path, detail: "maxDigits \(maxDigits)")
                }
                requiredBytes.formUnion(JSONBytes.zero...JSONBytes.nine)
                if allowNegative { requiredBytes.insert(JSONBytes.minus) }
                let size = Saturating.add(maxDigits, allowNegative ? 1 : 0)
                return (try append(.integer(maxDigits: maxDigits, allowNegative: allowNegative)), size)

            case .boolean:
                requiredBytes.formUnion(JSONBytes.trueBytes)
                requiredBytes.formUnion(JSONBytes.falseBytes)
                return (try append(.boolean), JSONBytes.falseBytes.count)

            case .enumeration(let values):
                guard !values.isEmpty else { throw SchemaError.emptyEnumeration(path: path) }
                guard values.count <= SchemaLimits.maxEnumerationValues else {
                    throw SchemaError.tooMany(path: path, detail: "\(values.count) enumeration values")
                }
                var seen = Set<String>()
                var encoded: [[UInt8]] = []
                var longest = 0
                for value in values {
                    guard value.utf8.allSatisfy(JSONBytes.isStringByte),
                          value.utf8.count <= SchemaLimits.maxStringLength else {
                        throw SchemaError.unsupportedCharacter(path: path, text: value)
                    }
                    guard seen.insert(value).inserted else {
                        throw SchemaError.duplicateEnumerationValue(path: path, value: value)
                    }
                    let bytes = Array(value.utf8)
                    requiredBytes.formUnion(bytes)
                    encoded.append(bytes)
                    longest = max(longest, bytes.count)
                }
                requiredBytes.insert(JSONBytes.quote)
                return (try append(.enumeration(encoded)), Saturating.add(longest, 2))

            case .array(let item, let minItems, let maxItems):
                guard minItems >= 0, maxItems >= minItems, maxItems <= SchemaLimits.maxItems else {
                    throw SchemaError.invalidBounds(path: path, detail: "items \(minItems)...\(maxItems)")
                }
                let (child, childSize) = try compile(item, path: "\(path)[]", depth: depth + 1)
                requiredBytes.formUnion([JSONBytes.openBracket, JSONBytes.closeBracket])
                if maxItems > 1 { requiredBytes.insert(JSONBytes.comma) }
                // "[" + items + commas + "]"
                let items = Saturating.multiply(maxItems, childSize)
                let commas = max(0, maxItems - 1)
                let size = Saturating.add(Saturating.add(items, commas), 2)
                return (try append(.array(item: child, min: minItems, max: maxItems)), size)
            }
        }

        private mutating func append(_ node: Node) throws -> Int {
            guard nodes.count < SchemaLimits.maxNodes else {
                throw SchemaError.tooMany(path: "$", detail: "more than \(SchemaLimits.maxNodes) schema nodes")
            }
            nodes.append(node)
            return nodes.count - 1
        }
    }
}
