/// The model seam. Anything that turns a token context into next-token logits —
/// an MLX or llama.cpp runtime, a Core ML model, a remote endpoint that exposes
/// logits, or a deterministic test double — plugs in here.
///
/// Logits must have exactly `vocabulary.size` entries, with EOS last.
public protocol LogitsProvider: Sendable {
    func logits(after context: [Int], vocabulary: Vocabulary) async throws -> [Float]
}

public struct DecodingConfiguration: Sendable, Hashable {
    public var strategy: SamplingStrategy
    public var seed: UInt64
    /// An optional caller-imposed cap on decode steps (EOS included).
    ///
    /// `nil` (the default) means "the constraint's own bound",
    /// `constraint.maxOutputBytes + 1`, which is exactly the budget the
    /// termination guarantee needs. A smaller cap trades that guarantee for a
    /// latency ceiling: the decode can then stop early with
    /// ``Termination/tokenBudgetExhausted`` and an incomplete document.
    public var maxTokens: Int?

    /// Hard ceiling on any decode, whatever a constraint claims: the longest
    /// document `SchemaLimits` allows, plus EOS. A constraint whose bound
    /// saturates at `Int.max` (or a custom one that is simply wrong) still gets
    /// a finite budget, so `steps`, `bytes` and `context` stay bounded.
    public static let absoluteMaxTokens = SchemaLimits.maxEncodedBytes + 1

    /// The step budget a decode actually gets.
    public func effectiveBudget(constraintMaxOutputBytes: Int) -> Int {
        let bound = Saturating.add(max(0, constraintMaxOutputBytes), 1)
        let requested = maxTokens.map { min($0, bound) } ?? bound
        return min(requested, Self.absoluteMaxTokens)
    }

    public init(strategy: SamplingStrategy = .greedy, seed: UInt64 = 0, maxTokens: Int? = nil) {
        self.strategy = strategy
        self.seed = seed
        self.maxTokens = maxTokens
    }
}

/// Named to avoid shadowing `Swift.DecodingError` (Codable) in client code.
public enum ConstrainedDecodingError: Error, Equatable, Sendable {
    case invalidConfiguration(String)
    /// The vocabulary lacks one-byte tokens the constraint may need; decoding
    /// could reach a state no token can advance. Raised at init, not mid-decode.
    case vocabularyCannotExpress(bytes: [UInt8])
    case logitsShapeMismatch(expected: Int, actual: Int, step: Int)
    /// No token is allowed and the state is not accepting. Unreachable when the
    /// init-time coverage check is truthful; kept as a typed error, not a trap.
    case deadEnd(step: Int, emittedBytes: Int)
    /// A token from the mask failed to re-advance the constraint, meaning
    /// `advance` is not a pure function. A bug in the constraint, surfaced loudly.
    case inconsistentConstraint(step: Int, tokenID: Int)
}

public enum Termination: Sendable, Hashable {
    case endOfSequence
    case tokenBudgetExhausted
}

public struct DecodingStep: Sendable, Hashable {
    public let tokenID: Int
    public let isEndOfSequence: Bool
    public let allowedCount: Int
    /// `nil` when the step was forced (only one token allowed) and the model was not consulted.
    public let maskedMass: Double?
    public let maskCacheHit: Bool
    public let degenerate: Bool
}

public struct DecodingStats: Sendable, Hashable {
    public var modelCalls = 0
    public var forcedSteps = 0
    public var maskCacheHits = 0
    public var maskCacheMisses = 0
    /// `advance` calls spent computing masks (cache misses only).
    public var constraintAdvances = 0
    public var tokenBudget = 0
}

public struct DecodingResult: Sendable, Hashable {
    /// Emitted token ids, EOS excluded.
    public let tokens: [Int]
    public let bytes: [UInt8]
    public let steps: [DecodingStep]
    public let termination: Termination
    /// Whether the constraint accepts the final output. For a schema constraint
    /// this is `true` whenever `termination == .endOfSequence`.
    public let isAccepted: Bool
    public let stats: DecodingStats

    public var text: String { String(decoding: bytes, as: UTF8.self) }
}

/// Runs a model under a constraint, one token at a time.
///
/// **Concurrency model.** The decoder is an actor so that several decodes can
/// share one mask cache. Each `decode` call keeps its matcher state, context,
/// RNG and stats in *locals*; the only actor state is the cache, whose entries
/// are a pure function of their key. The one suspension point per step — the
/// model call — therefore cannot invalidate anything the call relies on: if
/// another decode inserts or evicts cache entries meanwhile, this one computes
/// its next mask from its own state and gets the same answer either way. The
/// test suite runs decodes concurrently on one decoder and requires results
/// identical to sequential runs on fresh decoders.
///
/// **Termination.** Every non-EOS token adds at least one byte and the
/// constraint accepts at most `maxOutputBytes`, so after that many tokens only
/// EOS can be allowed. The step budget defaults to `maxOutputBytes + 1`, is
/// never more than ``DecodingConfiguration/absoluteMaxTokens``, and is enforced
/// here — so a decode ends even if the constraint's bound is wrong or
/// saturated; with an honest bound and no smaller caller cap it ends in EOS.
public actor ConstrainedDecoder<Constraint: DecodingConstraint> {
    public nonisolated let constraint: Constraint
    public nonisolated let vocabulary: Vocabulary
    private let computer: MaskComputer<Constraint>
    private var cache: LRUCache<Constraint.State, TokenMask>

    public init(constraint: Constraint, vocabulary: Vocabulary, maskCacheCapacity: Int = 512) throws {
        let missing = constraint.uncoveredBytes(singleByteTokens: vocabulary.singleByteTokens)
        guard missing.isEmpty else { throw ConstrainedDecodingError.vocabularyCannotExpress(bytes: missing) }
        self.constraint = constraint
        self.vocabulary = vocabulary
        self.computer = MaskComputer(constraint: constraint, vocabulary: vocabulary)
        self.cache = LRUCache(capacity: maskCacheCapacity)
    }

    public var maskCacheCount: Int { cache.count }
    public var maskCacheEvictions: Int { cache.evictions }

    public func decode(using provider: some LogitsProvider,
                       configuration: DecodingConfiguration = DecodingConfiguration()) async throws -> DecodingResult {
        if case .temperature(let temperature) = configuration.strategy, !(temperature.isFinite && temperature > 0) {
            throw ConstrainedDecodingError.invalidConfiguration("temperature must be finite and > 0, got \(temperature)")
        }
        if let cap = configuration.maxTokens, cap <= 0 {
            throw ConstrainedDecodingError.invalidConfiguration("maxTokens must be > 0, got \(cap)")
        }

        let budget = configuration.effectiveBudget(constraintMaxOutputBytes: constraint.maxOutputBytes)
        var stats = DecodingStats()
        stats.tokenBudget = budget
        var state = constraint.initialState
        var context: [Int] = []
        var bytes: [UInt8] = []
        var steps: [DecodingStep] = []
        var rng = SplitMix64(seed: configuration.seed)

        for stepIndex in 0..<budget {
            try Task.checkCancellation()
            let (mask, hit) = maskFor(state, stats: &stats)
            guard let only = mask.allowed.first else {
                throw ConstrainedDecodingError.deadEnd(step: stepIndex, emittedBytes: bytes.count)
            }

            let chosen: Int
            if mask.count == 1 {
                // Forced token: the model cannot change the outcome, so do not pay for it.
                chosen = only
                stats.forcedSteps += 1
                steps.append(DecodingStep(tokenID: chosen, isEndOfSequence: chosen == vocabulary.eosID,
                                          allowedCount: 1, maskedMass: nil, maskCacheHit: hit, degenerate: false))
            } else {
                let logits = try await provider.logits(after: context, vocabulary: vocabulary)
                stats.modelCalls += 1
                guard logits.count == vocabulary.size else {
                    throw ConstrainedDecodingError.logitsShapeMismatch(expected: vocabulary.size, actual: logits.count, step: stepIndex)
                }
                guard let choice = Sampler.choose(logits: logits, allowed: mask.allowed,
                                                  strategy: configuration.strategy, rng: &rng) else {
                    throw ConstrainedDecodingError.deadEnd(step: stepIndex, emittedBytes: bytes.count)
                }
                chosen = choice.id
                steps.append(DecodingStep(tokenID: chosen, isEndOfSequence: chosen == vocabulary.eosID,
                                          allowedCount: mask.count, maskedMass: choice.maskedMass,
                                          maskCacheHit: hit, degenerate: choice.degenerate))
            }

            if chosen == vocabulary.eosID {
                return DecodingResult(tokens: context, bytes: bytes, steps: steps, termination: .endOfSequence,
                                      isAccepted: constraint.isAccepting(state), stats: stats)
            }
            guard let tokenBytes = vocabulary.bytes(of: chosen) else {
                throw ConstrainedDecodingError.inconsistentConstraint(step: stepIndex, tokenID: chosen)
            }
            var next = state
            for byte in tokenBytes {
                guard let advanced = constraint.advance(next, byte: byte) else {
                    throw ConstrainedDecodingError.inconsistentConstraint(step: stepIndex, tokenID: chosen)
                }
                next = advanced
            }
            state = next
            context.append(chosen)
            bytes.append(contentsOf: tokenBytes)
        }
        return DecodingResult(tokens: context, bytes: bytes, steps: steps, termination: .tokenBudgetExhausted,
                              isAccepted: constraint.isAccepting(state), stats: stats)
    }

    /// Synchronous on the actor: no suspension between lookup and insert.
    private func maskFor(_ state: Constraint.State, stats: inout DecodingStats) -> (TokenMask, Bool) {
        if let cached = cache.value(for: state) {
            stats.maskCacheHits += 1
            return (cached, true)
        }
        let (mask, advances) = computer.compute(state)
        stats.maskCacheMisses += 1
        stats.constraintAdvances = Saturating.add(stats.constraintAdvances, advances)
        cache.insert(mask, for: state)
        return (mask, false)
    }

    /// Test hook: the mask for a state without touching the cache.
    func uncachedMask(for state: Constraint.State) -> TokenMask { computer.compute(state).mask }
}
