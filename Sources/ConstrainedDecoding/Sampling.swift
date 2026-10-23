import Foundation

/// SplitMix64: a tiny, fast, fully deterministic PRNG. Determinism is a feature
/// here — the same seed must reproduce the same decode on every platform, which
/// is what makes the demo and the concurrency tests comparable run to run.
public struct SplitMix64: Sendable, Hashable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    public mutating func nextUnitInterval() -> Double {
        Double(next() >> 11) * 0x1.0p-53
    }

    /// A stateless hash of several words, for reproducible per-(step, token) noise.
    public static func mix(_ words: UInt64...) -> UInt64 {
        var generator = SplitMix64(seed: 0x51_7C_C1_B7_27_22_0A_95)
        var result: UInt64 = 0
        for word in words {
            generator = SplitMix64(seed: generator.next() ^ word)
            result ^= generator.next()
        }
        return result
    }
}

public enum SamplingStrategy: Sendable, Hashable {
    case greedy
    /// Softmax sampling at a finite, positive temperature.
    case temperature(Double)
}

/// Picks a token from the *allowed* set only, and measures how much the mask
/// distorted the model's distribution.
enum Sampler {
    struct Choice: Equatable {
        let id: Int
        /// Probability mass the mask removed from the distribution actually being
        /// sampled (the softmax at the sampling temperature; T = 1 for greedy), in `[0, 1]`.
        /// High values mean the constraint is fighting the model, which is where
        /// output quality degrades even though validity is guaranteed.
        let maskedMass: Double
        /// True when no allowed logit was finite, so the pick fell back to the
        /// lowest allowed id instead of following the model.
        let degenerate: Bool
    }

    /// - Precondition: `allowed` is non-empty and sorted; every id indexes `logits`
    ///   (the decoder checks both before calling).
    static func choose(logits: [Float], allowed: [Int], strategy: SamplingStrategy,
                       rng: inout SplitMix64) -> Choice? {
        guard let firstAllowed = allowed.first else { return nil }
        let values = logits.map(sanitize)
        guard allowed.allSatisfy(values.indices.contains) else { return nil }
        let allowedValues = allowed.map { values[$0] }

        let maskedMass: Double
        if case .temperature(let temperature) = strategy, temperature.isFinite, temperature > 0,
           let top = values.map({ $0 / temperature }).max(), top.isFinite {
            maskedMass = maskedProbabilityMass(all: values.map { $0 / temperature },
                                               allowed: allowedValues.map { $0 / temperature })
        } else {
            // Greedy, or a temperature so small the scaled logits overflow: the
            // T = 1 distribution is the meaningful (and finite) reference.
            maskedMass = maskedProbabilityMass(all: values, allowed: allowedValues)
        }
        guard let best = allowedValues.max(), best.isFinite else {
            return Choice(id: firstAllowed, maskedMass: maskedMass, degenerate: true)
        }

        switch strategy {
        case .greedy:
            return Choice(id: argmax(allowed, allowedValues), maskedMass: maskedMass, degenerate: false)
        case .temperature(let temperature):
            // Dividing by a tiny temperature can overflow to +inf; at that point the
            // distribution is effectively one-hot, so greedy is the correct limit.
            let scaled = allowedValues.map { $0 / temperature }
            guard let top = scaled.max(), top.isFinite else {
                return Choice(id: argmax(allowed, allowedValues), maskedMass: maskedMass, degenerate: false)
            }
            let weights = scaled.map { $0 == -.infinity ? 0 : exp($0 - top) }
            let total = weights.reduce(0, +)
            guard total > 0, total.isFinite else {
                return Choice(id: argmax(allowed, allowedValues), maskedMass: maskedMass, degenerate: false)
            }
            let target = rng.nextUnitInterval() * total
            var running = 0.0
            var lastPositive = firstAllowed
            for (offset, weight) in weights.enumerated() where weight > 0 {
                running += weight
                lastPositive = allowed[offset]
                if target < running { return Choice(id: allowed[offset], maskedMass: maskedMass, degenerate: false) }
            }
            // Floating-point rounding can leave `target` a hair above the final sum.
            return Choice(id: lastPositive, maskedMass: maskedMass, degenerate: false)
        }
    }

    /// NaN is treated as "impossible" (-inf) rather than poisoning the softmax;
    /// +inf is clamped to the largest finite value so it still wins argmax and
    /// softmax without turning `x - max` into `inf - inf`.
    static func sanitize(_ value: Float) -> Double {
        if value.isNaN { return -.infinity }
        if value == .infinity { return Double(Float.greatestFiniteMagnitude) }
        return Double(value)
    }

    /// Ties resolve to the lowest id, so greedy decoding is reproducible.
    private static func argmax(_ ids: [Int], _ values: [Double]) -> Int {
        var bestID = ids.first ?? 0
        var bestValue = -Double.infinity
        for (id, value) in zip(ids, values) where value > bestValue {
            bestID = id
            bestValue = value
        }
        return bestID
    }

    /// `1 - P(allowed)` under the unmasked softmax, computed in log space so
    /// large logits cannot overflow.
    static func maskedProbabilityMass(all: [Double], allowed: [Double]) -> Double {
        guard let logZ = logSumExp(all) else { return 0 }
        guard let logAllowed = logSumExp(allowed) else { return 1 }
        let kept = exp(logAllowed - logZ)
        guard kept.isFinite else { return 0 }
        return min(1, max(0, 1 - kept))
    }

    /// `nil` when every value is -inf (an empty distribution).
    static func logSumExp(_ values: [Double]) -> Double? {
        guard let top = values.max(), top.isFinite else { return nil }
        let sum = values.reduce(0.0) { $0 + ($1 == -.infinity ? 0 : exp($1 - top)) }
        return top + log(sum)
    }
}
