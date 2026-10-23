/// A deterministic stand-in for a small language model, used by the demo and tests.
///
/// It is a copy model over one reference document — an n-gram model "trained"
/// on a single text, the same idea as infini-gram-style suffix matching:
/// - a token scores higher the longer the suffix of the output it extends into
///   a substring of the reference (it has "seen" that continuation), plus a
///   small bonus per byte (longer merges are likelier, as with real BPE models),
/// - and gets a large bonus when the output so far is a prefix of the reference
///   and the token continues it exactly (left alone, the model reproduces the
///   reference verbatim).
///
/// Give it a chatty, almost-JSON reference and you get the behaviour that makes
/// constrained decoding necessary: unconstrained, it emits prose, a trailing
/// comma and a float where an integer belongs; under a mask, its preferences
/// still steer every choice the schema leaves open.
///
/// Not a real model. Its value is that every number in the demo is reproducible.
public struct ReferenceTextModel: LogitsProvider {
    public let reference: [UInt8]
    public private(set) var seed: UInt64
    public let noise: Float
    private let maxOverlap = 8
    private let maxSubstring: Int
    private let substrings: Set<[UInt8]>

    /// Reference text beyond this many bytes is ignored, which bounds the
    /// substring index at `2 × maxReferenceBytes × maxSubstring` entries (the
    /// original text and its compacted variant are both indexed).
    public static let maxReferenceBytes = 2_048

    public init(reference: String, seed: UInt64 = 0, noise: Float = 0.6, longestToken: Int = 24) {
        let bytes = Array(reference.utf8.prefix(Self.maxReferenceBytes))
        self.reference = bytes
        self.seed = seed
        self.noise = noise.isFinite ? max(0, noise) : 0
        let span = maxOverlap + max(1, min(longestToken, 64))
        self.maxSubstring = span
        // The model has "seen" JSON both pretty (`": "`) and compact (`":"`), as
        // real models have; without the compact variant it could never connect a
        // canonical `"title":"` to the value that followed `"title": "`.
        let compact = Array(reference.replacingAll(": ", with: ":").replacingAll(", ", with: ",").utf8
            .prefix(Self.maxReferenceBytes))
        var index = Set<[UInt8]>()
        for text in [bytes, compact] {
            for start in text.indices {
                let limit = min(text.count, start + span)
                var end = start + 1
                while end <= limit {
                    index.insert(Array(text[start..<end]))
                    end += 1
                }
            }
        }
        self.substrings = index
    }

    /// The same model with different sampling noise. The substring index is
    /// shared copy-on-write, so this is O(1) — build once, reseed per decode.
    public func reseeded(_ seed: UInt64) -> ReferenceTextModel {
        var copy = self
        copy.seed = seed
        return copy
    }

    public func logits(after context: [Int], vocabulary: Vocabulary) async throws -> [Float] {
        var offset = 0
        var tail: [UInt8] = []
        // "On script" while the output so far is a prefix of the reference. Only
        // then does the exact-continuation bonus apply; otherwise a constrained
        // output would pick up bonuses from whatever reference byte happens to
        // sit at the same offset.
        var onScript = true
        for id in context {
            guard let bytes = vocabulary.bytes(of: id) else { continue }
            if onScript {
                let end = Saturating.add(offset, bytes.count)
                onScript = end <= reference.count && reference[offset..<end].elementsEqual(bytes)
            }
            offset = Saturating.add(offset, bytes.count)
            tail.append(contentsOf: bytes)
            if tail.count > maxOverlap { tail.removeFirst(tail.count - maxOverlap) }
        }

        var logits = [Float](repeating: -4, count: vocabulary.size)
        for (id, token) in vocabulary.tokens.enumerated() {
            var score: Float = -4
            var overlap = min(maxOverlap, tail.count)
            while overlap >= 0 {
                let candidate = Array(tail.suffix(overlap)) + token
                if candidate.count <= maxSubstring, substrings.contains(candidate) {
                    score = Float(overlap) + 0.25 * Float(token.count)
                    break
                }
                overlap -= 1
            }
            if onScript, offset < reference.count,
               offset <= reference.count - token.count,
               reference[offset..<(offset + token.count)].elementsEqual(token) {
                score += 6
            }
            logits[id] = score + jitter(step: context.count, token: id)
        }

        // EOS: certain once the reference is exhausted, plausible right after a
        // sentence ends, unlikely otherwise.
        let eos: Float
        if onScript, offset >= reference.count {
            eos = 30
        } else if let last = tail.last, last == 0x2E || last == 0x21 || last == 0x7D {
            eos = 2
        } else {
            eos = -2
        }
        logits[vocabulary.eosID] = eos
        return logits
    }

    private func jitter(step: Int, token: Int) -> Float {
        let hash = SplitMix64.mix(seed, UInt64(truncatingIfNeeded: step), UInt64(truncatingIfNeeded: token))
        return noise * Float(hash >> 40) / Float(1 << 24)
    }
}

private extension String {
    /// Literal, non-overlapping, left-to-right replacement without Foundation.
    func replacingAll(_ target: String, with replacement: String) -> String {
        let source = Array(utf8)
        let pattern = Array(target.utf8)
        guard !pattern.isEmpty else { return self }
        var output: [UInt8] = []
        var index = 0
        while index < source.count {
            if source[index...].starts(with: pattern) {
                output.append(contentsOf: replacement.utf8)
                index += pattern.count
            } else {
                output.append(source[index])
                index += 1
            }
        }
        return String(decoding: output, as: UTF8.self)
    }
}
