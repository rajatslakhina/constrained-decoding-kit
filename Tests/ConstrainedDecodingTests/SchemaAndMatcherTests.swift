import XCTest
@testable import ConstrainedDecoding

final class SchemaCompilationTests: XCTestCase {
    func testRejectsInvalidSchemas() {
        let cases: [(Schema, SchemaError)] = [
            (.enumeration([]), .emptyEnumeration(path: "$")),
            (.enumeration(["a", "a"]), .duplicateEnumerationValue(path: "$", value: "a")),
            (.enumeration(["say \"hi\""]), .unsupportedCharacter(path: "$", text: "say \"hi\"")),
            (.string(minLength: 3, maxLength: 2), .invalidBounds(path: "$", detail: "string length 3...2")),
            (.string(minLength: -1, maxLength: 2), .invalidBounds(path: "$", detail: "string length -1...2")),
            (.integer(maxDigits: 0), .invalidBounds(path: "$", detail: "maxDigits 0")),
            (.integer(maxDigits: 19), .invalidBounds(path: "$", detail: "maxDigits 19")),
            (.array(of: .boolean, minItems: 2, maxItems: 1), .invalidBounds(path: "$", detail: "items 2...1")),
            (.object([.init("a", .boolean), .init("a", .boolean)]), .duplicateProperty(path: "$", name: "a")),
            (.object([.init("", .boolean)]), .emptyPropertyName(path: "$")),
            (.object([.init("caf\u{E9}", .boolean)]), .unsupportedCharacter(path: "$", text: "caf\u{E9}")),
        ]
        for (schema, expected) in cases {
            XCTAssertThrowsError(try CompiledSchema(schema), "\(schema)") { error in
                XCTAssertEqual(error as? SchemaError, expected)
            }
        }
    }

    func testDepthLimit() {
        var schema = Schema.boolean
        for _ in 0..<SchemaLimits.maxDepth { schema = .array(of: schema, maxItems: 1) }
        XCTAssertThrowsError(try CompiledSchema(schema)) { error in
            XCTAssertEqual(error as? SchemaError, .tooDeep(limit: SchemaLimits.maxDepth))
        }
        var shallow = Schema.boolean
        for _ in 0..<(SchemaLimits.maxDepth - 1) { shallow = .array(of: shallow, maxItems: 1) }
        XCTAssertNoThrow(try CompiledSchema(shallow))
    }

    func testMaxEncodedBytesIsExactForTheWidestDocument() throws {
        // {"a":-999,"b":"xyz","c":[false,false]}
        let schema = Schema.object([
            .init("a", .integer(maxDigits: 3, allowNegative: true)),
            .init("b", .string(maxLength: 3)),
            .init("c", .array(of: .boolean, maxItems: 2)),
        ])
        let widest = #"{"a":-999,"b":"xyz","c":[false,false]}"#
        let compiled = try CompiledSchema(schema)
        XCTAssertEqual(compiled.maxEncodedBytes, widest.utf8.count)
        let constraint = try JSONSchemaConstraint(schema)
        let state = try XCTUnwrap(Fixtures.feed(constraint, widest))
        XCTAssertTrue(constraint.isAccepting(state))
    }

    /// 1024^7 * 5 bytes overflows Int. The size computation must saturate (not
    /// trap), and the result must then be refused by the encoded-size cap.
    func testNestedArraySizeSaturatesThenHitsTheEncodedSizeCap() {
        var schema = Schema.boolean
        for _ in 0..<7 { schema = .array(of: schema, maxItems: SchemaLimits.maxItems) }
        XCTAssertThrowsError(try CompiledSchema(schema)) { error in
            XCTAssertEqual(error as? SchemaError,
                           .tooMany(path: "$", detail: "documents up to Int.max (saturated) bytes; limit \(SchemaLimits.maxEncodedBytes)"))
        }
    }

    /// Three nodes, 4 GB of possible output: refused, so no default decode
    /// budget can be large enough to exhaust memory.
    func testEncodedSizeCapBoundsSmallButHugeSchemas() throws {
        let huge = Schema.array(of: .array(of: .string(maxLength: 4_096), maxItems: 1_024), maxItems: 1_024)
        XCTAssertThrowsError(try CompiledSchema(huge))
        let atLimit = Schema.string(maxLength: SchemaLimits.maxStringLength)
        XCTAssertEqual(try CompiledSchema(atLimit).maxEncodedBytes, SchemaLimits.maxStringLength + 2)
    }

    /// 200 properties of 200 properties of 200 booleans is a few lines of
    /// `Schema` but 8 million compiled nodes. The node limit must stop it early
    /// and cheaply (well under a second), not after allocating them.
    func testWideNestedSchemaIsRejectedBeforeItExplodes() {
        let leaf = Schema.object((0..<200).map { .init("p\($0)", .boolean) })
        let middle = Schema.object((0..<200).map { .init("m\($0)", leaf) })
        let top = Schema.object((0..<200).map { .init("t\($0)", middle) })
        let start = ContinuousClock.now
        XCTAssertThrowsError(try CompiledSchema(top)) { error in
            guard case .tooMany? = error as? SchemaError else { return XCTFail("\(error)") }
        }
        XCTAssertLessThan(ContinuousClock.now - start, .seconds(2))
        XCTAssertNoThrow(try CompiledSchema(leaf))
    }

    func testSaturatingHelpers() {
        XCTAssertEqual(Saturating.add(Int.max, 1), Int.max)
        XCTAssertEqual(Saturating.add(Int.min, -1), Int.min)
        XCTAssertEqual(Saturating.multiply(Int.max / 2 + 1, 2), Int.max)
        XCTAssertEqual(Saturating.multiply(Int.min, -1), Int.max)
        XCTAssertEqual(Saturating.multiply(Int.max, -2), Int.min)
        XCTAssertEqual(Saturating.add(2, 3), 5)
        XCTAssertEqual(Saturating.multiply(-4, 3), -12)
    }

    func testCoverageReportsMissingStructuralBytes() throws {
        let compiled = try CompiledSchema(Fixtures.ticket)
        var bytes = Fixtures.vocabulary.singleByteTokens
        XCTAssertEqual(compiled.uncoveredBytes(singleByteTokens: bytes), [])
        bytes.remove(0x7D) // }
        bytes.remove(0x37) // 7
        XCTAssertEqual(compiled.uncoveredBytes(singleByteTokens: bytes), [0x37, 0x7D])
    }

    /// A non-empty string needs *some* one-byte string character; an
    /// optional-length one needs only the quote. (Object keys already contribute
    /// printable bytes, so this is observable on a root string schema.)
    func testCoverageRequiresSomeStringCharacterOnlyWhenAStringMustBeNonEmpty() throws {
        let quote: Set<UInt8> = [0x22]
        let mustHaveText = try CompiledSchema(.string(minLength: 1, maxLength: 4))
        let mayBeEmpty = try CompiledSchema(.string(minLength: 0, maxLength: 4))
        XCTAssertEqual(mustHaveText.uncoveredBytes(singleByteTokens: quote), [0x61])
        XCTAssertEqual(mustHaveText.uncoveredBytes(singleByteTokens: quote.union([0x78])), [])
        XCTAssertEqual(mayBeEmpty.uncoveredBytes(singleByteTokens: quote), [])
        XCTAssertEqual(mayBeEmpty.uncoveredBytes(singleByteTokens: []), [0x22])
    }
}

final class MatcherTests: XCTestCase {
    private func accepts(_ schema: Schema, _ text: String, file: StaticString = #filePath, line: UInt = #line) throws -> Bool {
        let constraint = try JSONSchemaConstraint(schema)
        guard let state = Fixtures.feed(constraint, text) else { return false }
        return constraint.isAccepting(state)
    }

    func testAcceptsCanonicalDocuments() throws {
        XCTAssertTrue(try accepts(Fixtures.ticket,
            #"{"title":"Crash","priority":"high","estimate_hours":6,"blocking":true,"labels":["crash"]}"#))
        XCTAssertTrue(try accepts(Fixtures.kitchenSink,
            #"{"id":-12,"name":"","tags":[],"grid":[[1,23],[0]],"mode":"ab","ok":false,"empty":{},"n":42}"#))
        XCTAssertTrue(try accepts(.integer(maxDigits: 3), "120"))
        XCTAssertTrue(try accepts(.integer(maxDigits: 3, allowNegative: true), "-0"))
    }

    func testRejectsNonCanonicalOrInvalidDocuments() throws {
        let rejected: [(Schema, String)] = [
            (Fixtures.ticket, #"{"title": "Crash","priority":"high","estimate_hours":6,"blocking":true,"labels":["crash"]}"#),
            (Fixtures.ticket, #"{"priority":"high","title":"Crash","estimate_hours":6,"blocking":true,"labels":["crash"]}"#),
            (Fixtures.ticket, #"{"title":"Crash","priority":"P1","estimate_hours":6,"blocking":true,"labels":["crash"]}"#),
            (Fixtures.ticket, #"{"title":"Crash","priority":"high","estimate_hours":6.5,"blocking":true,"labels":["crash"]}"#),
            (Fixtures.ticket, #"{"title":"Crash","priority":"high","estimate_hours":6,"blocking":"yes","labels":["crash"]}"#),
            (Fixtures.ticket, #"{"title":"Crash","priority":"high","estimate_hours":6,"blocking":true,"labels":["crash",]}"#),
            (Fixtures.ticket, #"{"title":"Crash","priority":"high","estimate_hours":6,"blocking":true,"labels":[]}"#),
            (Fixtures.ticket, #"{"title":"","priority":"high","estimate_hours":6,"blocking":true,"labels":["crash"]}"#),
            (.integer(maxDigits: 3), "012"),
            (.integer(maxDigits: 3), "1234"),
            (.integer(maxDigits: 3), "-1"),
            (.integer(maxDigits: 3, allowNegative: true), "-"),
            (.string(maxLength: 3), #""abcd""#),
            (.string(maxLength: 9), #""a\"b""#),
            (.enumeration(["abc"]), #""ab""#),
            (.array(of: .boolean, maxItems: 2), "[true,true,true]"),
            (.boolean, "tru"),
            (.boolean, "true "),
        ]
        for (schema, text) in rejected {
            XCTAssertFalse(try accepts(schema, text), text)
        }
    }

    func testIntegerEndsAtTheEnclosingDelimiter() throws {
        let constraint = try JSONSchemaConstraint(.array(of: .integer(maxDigits: 2), maxItems: 3))
        let state = try XCTUnwrap(Fixtures.feed(constraint, "[1,23"))
        XCTAssertFalse(constraint.isAccepting(state))
        XCTAssertNotNil(constraint.advance(state, byte: 0x2C)) // ,
        XCTAssertNotNil(constraint.advance(state, byte: 0x5D)) // ]
        XCTAssertNil(constraint.advance(state, byte: 0x34))    // a third digit
    }

    func testNothingFollowsACompleteDocument() throws {
        let constraint = try JSONSchemaConstraint(Fixtures.ticket)
        let state = try XCTUnwrap(Fixtures.feed(constraint,
            #"{"title":"A","priority":"low","estimate_hours":0,"blocking":false,"labels":["ui"]}"#))
        XCTAssertTrue(constraint.isAccepting(state))
        for byte in UInt8(0)...UInt8(255) { XCTAssertNil(constraint.advance(state, byte: byte)) }
    }

    /// Liveness + bound, checked on random walks: from every reachable state an
    /// accepting state is reachable within the remaining byte budget, and no
    /// walk exceeds `maxOutputBytes`. Walks pick uniformly among accepted bytes,
    /// so they wander into deep nesting and long strings.
    func testEveryReachableStateIsLiveAndBounded() throws {
        for schema in [Fixtures.ticket, Fixtures.kitchenSink] {
            let constraint = try JSONSchemaConstraint(schema)
            XCTAssertNil(firstLivenessViolation(constraint, seeds: 0..<12))
        }
    }

    /// Checker non-vacuity: a constraint with a trapdoor (`q` leads to a state
    /// that is neither accepting nor extendable) must be reported.
    func testLivenessCheckCatchesATrapdoor() throws {
        let trapdoor = try TrapdoorConstraint(inner: JSONSchemaConstraint(Fixtures.ticket))
        XCTAssertNotNil(firstLivenessViolation(trapdoor, seeds: 0..<12))
    }

    private func firstLivenessViolation<C: DecodingConstraint>(_ constraint: C, seeds: Range<UInt64>) -> String? {
        for seed in seeds {
            var rng = SplitMix64(seed: seed)
            var state = constraint.initialState
            var length = 0
            while true {
                guard canReachAccept(constraint, from: state, within: constraint.maxOutputBytes - length) else {
                    return "dead end after \(length) bytes, seed \(seed)"
                }
                let options = (UInt8(0)...UInt8(255)).compactMap { constraint.advance(state, byte: $0) }
                if options.isEmpty { break }
                state = options[Int(rng.next() % UInt64(options.count))]
                length += 1
                if length > constraint.maxOutputBytes { return "exceeded maxOutputBytes, seed \(seed)" }
            }
        }
        return nil
    }

    /// Completes a document by always taking the first accepted byte from a
    /// closer-first priority list (`"`, `]`, `}`, `0`, `,` ...) and then any
    /// byte in ascending order. Closers first means every frame finishes as soon
    /// as the schema lets it, so the walk ends; the assertion is that it ends in
    /// an *accepting* state within the remaining byte budget.
    private func canReachAccept<C: DecodingConstraint>(_ constraint: C, from start: C.State, within budget: Int) -> Bool {
        let priority: [UInt8] = Array("\"]}0,ft".utf8) + Array(UInt8(0)...UInt8(255))
        var state = start
        var used = 0
        while !constraint.isAccepting(state) {
            guard used < budget,
                  let next = priority.lazy.compactMap({ constraint.advance(state, byte: $0) }).first else { return false }
            state = next
            used += 1
        }
        return true
    }
}

/// The schema matcher plus a trapdoor: `q` from any state leads to `nil`, a
/// state that accepts nothing and is not accepting.
struct TrapdoorConstraint: DecodingConstraint {
    let inner: JSONSchemaConstraint
    var initialState: MatcherState? { inner.initialState }
    var maxOutputBytes: Int { inner.maxOutputBytes }
    func advance(_ state: MatcherState?, byte: UInt8) -> MatcherState?? {
        guard let state else { return nil }
        if byte == 0x71 { return .some(nil) }
        return inner.advance(state, byte: byte).map { .some($0) }
    }
    func isAccepting(_ state: MatcherState?) -> Bool { state.map(inner.isAccepting) ?? false }
    func uncoveredBytes(singleByteTokens: Set<UInt8>) -> [UInt8] { [] }
}
