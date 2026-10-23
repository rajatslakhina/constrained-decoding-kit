import Foundation
@testable import ConstrainedDecoding

/// The ticket-triage scenario the demo app ships with, duplicated here so the
/// golden-output test pins exactly what the README and the app show.
enum Fixtures {
    static let reference = """
    Sure! Here's the ticket as JSON. I'd call it "high" priority, it blocks the release (true), and my estimate is 6, maybe 6.5 hours:
    {"title": "Crash when rotating the photo editor", "priority": "P1", "estimate_hours": 6.5, "blocking": "yes", "labels": ["crash", "editor", "ipad",],}
    Let me know if you need anything else!
    """

    static let ticket = Schema.object([
        .init("title", .string(minLength: 1, maxLength: 48)),
        .init("priority", .enumeration(["low", "medium", "high", "critical"])),
        .init("estimate_hours", .integer(maxDigits: 3)),
        .init("blocking", .boolean),
        .init("labels", .array(of: .enumeration(["crash", "editor", "ipad", "performance", "ui"]), minItems: 1, maxItems: 3)),
    ])

    static let vocabulary = Vocabulary.demo(corpus: reference)

    /// A schema touching every node kind, nesting and the integer-at-end case.
    static let kitchenSink = Schema.object([
        .init("id", .integer(maxDigits: 4, allowNegative: true)),
        .init("name", .string(minLength: 0, maxLength: 5)),
        .init("tags", .array(of: .string(minLength: 1, maxLength: 3), minItems: 0, maxItems: 3)),
        .init("grid", .array(of: .array(of: .integer(maxDigits: 2), minItems: 1, maxItems: 2), minItems: 1, maxItems: 2)),
        .init("mode", .enumeration(["a", "ab", "abc"])),
        .init("ok", .boolean),
        .init("empty", .object([])),
        .init("n", .integer(maxDigits: 2)),
    ])

    static func feed(_ constraint: JSONSchemaConstraint, _ text: String) -> MatcherState? {
        var state: MatcherState? = constraint.initialState
        for byte in text.utf8 {
            guard let current = state else { return nil }
            state = constraint.advance(current, byte: byte)
        }
        return state
    }

    /// Random walk over bytes the constraint accepts, recording every state.
    static func randomWalkStates(_ constraint: JSONSchemaConstraint, seed: UInt64, maxSteps: Int = 400) -> [MatcherState] {
        var rng = SplitMix64(seed: seed)
        var state = constraint.initialState
        var visited = [state]
        for _ in 0..<maxSteps {
            let options = (UInt8(0)...UInt8(255)).compactMap { constraint.advance(state, byte: $0) }
            guard !options.isEmpty else { break }
            let pick = Int(rng.next() % UInt64(options.count))
            state = options[pick]
            visited.append(state)
        }
        return visited
    }
}

/// Logits that like whitespace, love long tokens and never want EOS — the
/// adversary the termination guarantee has to survive.
struct AdversarialProvider: LogitsProvider {
    func logits(after context: [Int], vocabulary: Vocabulary) async throws -> [Float] {
        var logits = vocabulary.tokens.map { token -> Float in
            let spaces = token.filter { $0 == 0x20 || $0 == 0x0A }.count
            return Float(token.count) + 10 * Float(spaces)
        }
        logits.append(-.infinity) // EOS
        return logits
    }
}

/// Returns logits of the wrong length.
struct WrongShapeProvider: LogitsProvider {
    func logits(after context: [Int], vocabulary: Vocabulary) async throws -> [Float] {
        [Float](repeating: 0, count: vocabulary.size - 1)
    }
}

/// Counts calls so the forced-step fast path can be checked.
actor CallCounter {
    private(set) var calls = 0
    func increment() { calls += 1 }
}

struct CountingProvider: LogitsProvider {
    let inner: ReferenceTextModel
    let counter: CallCounter
    func logits(after context: [Int], vocabulary: Vocabulary) async throws -> [Float] {
        await counter.increment()
        return try await inner.logits(after: context, vocabulary: vocabulary)
    }
}

/// Slow enough that a cancellation lands mid-decode.
struct SlowProvider: LogitsProvider {
    func logits(after context: [Int], vocabulary: Vocabulary) async throws -> [Float] {
        try await Task.sleep(nanoseconds: 20_000_000)
        return [Float](repeating: 0, count: vocabulary.size)
    }
}

/// A deliberately broken constraint: the real schema matcher plus a hole that
/// lets `#` through anywhere. Used to prove the validator is not vacuous.
struct LeakyConstraint: DecodingConstraint {
    let inner: JSONSchemaConstraint
    var initialState: MatcherState { inner.initialState }
    var maxOutputBytes: Int { inner.maxOutputBytes }
    func advance(_ state: MatcherState, byte: UInt8) -> MatcherState? {
        byte == 0x23 ? state : inner.advance(state, byte: byte)
    }
    func isAccepting(_ state: MatcherState) -> Bool { inner.isAccepting(state) }
    func uncoveredBytes(singleByteTokens: Set<UInt8>) -> [UInt8] { inner.uncoveredBytes(singleByteTokens: singleByteTokens) }
}

/// A constraint that lies about its coverage requirements, so the init-time
/// check passes and the runtime dead-end path is exercised.
struct UnderdeclaredConstraint: DecodingConstraint {
    let inner: JSONSchemaConstraint
    var initialState: MatcherState { inner.initialState }
    var maxOutputBytes: Int { inner.maxOutputBytes }
    func advance(_ state: MatcherState, byte: UInt8) -> MatcherState? { inner.advance(state, byte: byte) }
    func isAccepting(_ state: MatcherState) -> Bool { inner.isAccepting(state) }
    func uncoveredBytes(singleByteTokens: Set<UInt8>) -> [UInt8] { [] }
}

/// Strongly prefers the `#` token at step 3 (and avoids it otherwise) and EOS
/// whenever it is allowed; otherwise indifferent, so greedy takes the lowest id.
struct HashLovingProvider: LogitsProvider {
    func logits(after context: [Int], vocabulary: Vocabulary) async throws -> [Float] {
        var logits = vocabulary.tokens.map { token -> Float in
            guard token == [0x23] else { return 0 }
            return context.count == 3 ? 100 : -10
        }
        logits.append(50)
        return logits
    }
}

/// Always wants `#`, never EOS.
struct AlwaysHashProvider: LogitsProvider {
    func logits(after context: [Int], vocabulary: Vocabulary) async throws -> [Float] {
        var logits = vocabulary.tokens.map { $0 == [0x23] ? Float(100) : Float(0) }
        logits.append(-.infinity)
        return logits
    }
}

actor InFlightTracker {
    private var current = 0
    private(set) var maxInFlight = 0
    func enter() { current += 1; maxInFlight = max(maxInFlight, current) }
    func exit() { current -= 1 }
}

/// Suspends several times inside every call, so concurrent decodes interleave
/// at the decoder's suspension point, and records the overlap.
struct YieldingProvider: LogitsProvider {
    let inner: ReferenceTextModel
    let tracker: InFlightTracker
    func logits(after context: [Int], vocabulary: Vocabulary) async throws -> [Float] {
        await tracker.enter()
        for _ in 0..<3 { await Task.yield() }
        let logits = try await inner.logits(after: context, vocabulary: vocabulary)
        await tracker.exit()
        return logits
    }
}

/// A thread-safe call counter (test-only; a lock keeps it honest under Swift 6).
final class CallBudget: @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0
    let limit: Int
    init(limit: Int) { self.limit = limit }
    /// True while calls so far are within the limit.
    func spend() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        calls += 1
        return calls <= limit
    }
}

/// Behaves like the real matcher for `budget.limit` calls, then refuses every
/// byte: `advance` is not a pure function of its arguments.
struct ExpiringConstraint: DecodingConstraint {
    let inner: JSONSchemaConstraint
    let budget: CallBudget
    var initialState: MatcherState { inner.initialState }
    var maxOutputBytes: Int { inner.maxOutputBytes }
    func advance(_ state: MatcherState, byte: UInt8) -> MatcherState? {
        budget.spend() ? inner.advance(state, byte: byte) : nil
    }
    func isAccepting(_ state: MatcherState) -> Bool { inner.isAccepting(state) }
    func uncoveredBytes(singleByteTokens: Set<UInt8>) -> [UInt8] { inner.uncoveredBytes(singleByteTokens: singleByteTokens) }
}
