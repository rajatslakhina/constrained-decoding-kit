import Foundation
import XCTest
@testable import ConstrainedDecoding

final class MaskingTests: XCTestCase {
    /// Two independent algorithms — trie walk with pruning and the naive
    /// per-token loop — must agree on every state a random walk reaches.
    func testTrieMaskMatchesNaiveOracle() throws {
        for schema in [Fixtures.ticket, Fixtures.kitchenSink] {
            let constraint = try JSONSchemaConstraint(schema)
            let computer = MaskComputer(constraint: constraint, vocabulary: Fixtures.vocabulary)
            var distinctMasks = Set<TokenMask>()
            for seed in UInt64(0)..<6 {
                for state in Fixtures.randomWalkStates(constraint, seed: seed) {
                    let fast = computer.compute(state).mask
                    XCTAssertEqual(fast, computer.computeNaive(state), "\(state)")
                    XCTAssertEqual(fast.allowed.contains(Fixtures.vocabulary.eosID), constraint.isAccepting(state))
                    distinctMasks.insert(fast)
                }
            }
            // Guards against both algorithms trivially returning the same constant.
            XCTAssertGreaterThan(distinctMasks.count, 10)
        }
    }

    /// The trie never costs more `advance` calls than the naive scan on any
    /// state (it explores each shared prefix once instead of once per token),
    /// and over random walks it is strictly cheaper overall. Inside a string
    /// almost every token is legal, so neither algorithm can be cheap there —
    /// which is the case the mask cache exists for.
    func testTriePruningNeverCostsMoreThanTheNaiveScan() throws {
        let constraint = try JSONSchemaConstraint(Fixtures.ticket)
        let computer = MaskComputer(constraint: constraint, vocabulary: Fixtures.vocabulary)
        var trieCost = 0
        var naiveCost = 0
        for seed in UInt64(0)..<4 {
            for state in Fixtures.randomWalkStates(constraint, seed: seed) {
                let trie = computer.compute(state).advances
                let naive = computer.computeNaiveCounted(state).advances
                XCTAssertLessThanOrEqual(trie, naive)
                trieCost += trie
                naiveCost += naive
            }
        }
        XCTAssertLessThan(trieCost, naiveCost)
        print("trie advances \(trieCost) vs naive \(naiveCost)")
    }

    /// Why the cache key must be the full stack. Within ONE constraint (so one
    /// decoder cache), the states after `["x` and after `["y","x` have the same
    /// top frame — inside the array's string, one character in — but `",` is
    /// legal in the first (a second item may follow) and illegal in the second
    /// (`maxItems: 2` is reached). A cache keyed on the top frame would serve
    /// one state the other's mask.
    func testSameTopFrameDifferentStacksHaveDifferentMasks() throws {
        let constraint = try JSONSchemaConstraint(.array(of: .string(maxLength: 3), maxItems: 2))
        let computer = MaskComputer(constraint: constraint, vocabulary: Fixtures.vocabulary)
        let first = try XCTUnwrap(Fixtures.feed(constraint, #"["x"#))
        let second = try XCTUnwrap(Fixtures.feed(constraint, #"["y","x"#))
        XCTAssertEqual(first.stack.last, second.stack.last, "precondition: identical top frames")
        XCTAssertNotEqual(first, second)
        let quoteComma = try XCTUnwrap(Fixtures.vocabulary.tokens.firstIndex(of: Array("\",".utf8)))
        XCTAssertTrue(computer.compute(first).mask.allowed.contains(quoteComma))
        XCTAssertFalse(computer.compute(second).mask.allowed.contains(quoteComma))
    }

    /// The decoder's cache, end to end: decoding through a shared decoder whose
    /// cache already holds the *first* state's mask must still refuse `",` in
    /// the second state (it would not if the key collapsed the two).
    func testDecoderCacheDistinguishesStatesWithTheSameTopFrame() async throws {
        let constraint = try JSONSchemaConstraint(.array(of: .string(maxLength: 3), maxItems: 2))
        let decoder = try ConstrainedDecoder(constraint: constraint, vocabulary: Fixtures.vocabulary)
        for seed in UInt64(0)..<10 {
            let result = try await decoder.decode(using: ReferenceTextModel(reference: #"["x","x","x","x"]"#, seed: seed),
                                                  configuration: .init(strategy: .temperature(1.5), seed: seed))
            XCTAssertEqual(SchemaValidator.validate(result.bytes, against: .array(of: .string(maxLength: 3), maxItems: 2)), .valid,
                           result.text)
        }
    }
}

final class LRUCacheTests: XCTestCase {
    func testEvictsLeastRecentlyUsed() {
        var cache = LRUCache<Int, String>(capacity: 2)
        cache.insert("a", for: 1)
        cache.insert("b", for: 2)
        XCTAssertEqual(cache.value(for: 1), "a")      // 1 becomes most recent
        cache.insert("c", for: 3)                      // evicts 2
        XCTAssertNil(cache.value(for: 2))
        XCTAssertEqual(cache.value(for: 1), "a")
        XCTAssertEqual(cache.value(for: 3), "c")
        XCTAssertEqual(cache.evictions, 1)
        XCTAssertEqual(cache.count, 2)
        XCTAssertEqual(cache.keysByRecency, [3, 1])
    }

    func testUpdateRefreshesRecencyWithoutGrowing() {
        var cache = LRUCache<Int, Int>(capacity: 3)
        for key in 0..<3 { cache.insert(key, for: key) }
        cache.insert(99, for: 0)
        XCTAssertEqual(cache.count, 3)
        XCTAssertEqual(cache.keysByRecency, [0, 2, 1])
        cache.insert(3, for: 3)                        // evicts 1, the oldest
        XCTAssertNil(cache.value(for: 1))
        XCTAssertEqual(cache.value(for: 0), 99)
    }

    func testCapacityZeroDisablesCaching() {
        var cache = LRUCache<Int, Int>(capacity: 0)
        cache.insert(1, for: 1)
        XCTAssertNil(cache.value(for: 1))
        XCTAssertEqual(cache.count, 0)
        var negative = LRUCache<Int, Int>(capacity: -5)
        negative.insert(1, for: 1)
        XCTAssertEqual(negative.count, 0)
    }

    func testCapacityOneAndChurnStayBounded() {
        var cache = LRUCache<Int, Int>(capacity: 1)
        for key in 0..<1_000 { cache.insert(key, for: key) }
        XCTAssertEqual(cache.count, 1)
        XCTAssertEqual(cache.value(for: 999), 999)
        XCTAssertEqual(cache.evictions, 999)
    }
}

final class SamplerTests: XCTestCase {
    func testGreedyBreaksTiesTowardTheLowestID() {
        var rng = SplitMix64(seed: 0)
        let choice = Sampler.choose(logits: [1, 5, 5, 0], allowed: [1, 2, 3], strategy: .greedy, rng: &rng)
        XCTAssertEqual(choice?.id, 1)
    }

    func testMaskedMassIsTheRemovedProbability() throws {
        var rng = SplitMix64(seed: 0)
        let half = try XCTUnwrap(Sampler.choose(logits: [0, 0], allowed: [1], strategy: .greedy, rng: &rng))
        XCTAssertEqual(half.maskedMass, 0.5, accuracy: 1e-12)
        let none = try XCTUnwrap(Sampler.choose(logits: [3, 3], allowed: [0, 1], strategy: .greedy, rng: &rng))
        XCTAssertEqual(none.maskedMass, 0, accuracy: 1e-12)
        // ln(3) vs 0: the masked token held 3/4 of the mass.
        let most = try XCTUnwrap(Sampler.choose(logits: [Float(log(3.0)), 0], allowed: [1], strategy: .greedy, rng: &rng))
        XCTAssertEqual(most.maskedMass, 0.75, accuracy: 1e-6)
    }

    /// Masked mass is measured on the distribution actually sampled: at T = 2
    /// the logits ln(9), 0 become ln(3), 0, so the masked token held 3/4, not 9/10.
    func testMaskedMassUsesTheSamplingTemperature() throws {
        var rng = SplitMix64(seed: 0)
        let tempered = try XCTUnwrap(Sampler.choose(logits: [Float(log(9.0)), 0], allowed: [1],
                                                    strategy: .temperature(2), rng: &rng))
        XCTAssertEqual(tempered.maskedMass, 0.75, accuracy: 1e-6)
        let greedy = try XCTUnwrap(Sampler.choose(logits: [Float(log(9.0)), 0], allowed: [1], strategy: .greedy, rng: &rng))
        XCTAssertEqual(greedy.maskedMass, 0.9, accuracy: 1e-6)
    }

    func testHugeLogitsDoNotOverflow() throws {
        var rng = SplitMix64(seed: 0)
        let choice = try XCTUnwrap(Sampler.choose(logits: [1e30, 1e30, .infinity], allowed: [0, 1],
                                                  strategy: .temperature(1e-300), rng: &rng))
        XCTAssertEqual(choice.id, 0)
        XCTAssertTrue(choice.maskedMass.isFinite)
        XCTAssertEqual(choice.maskedMass, 1, accuracy: 1e-9, "+inf on the masked token takes all the mass")
    }

    func testNaNAndNegativeInfinityAreTreatedAsImpossible() throws {
        var rng = SplitMix64(seed: 0)
        let picked = try XCTUnwrap(Sampler.choose(logits: [.nan, -1, -.infinity], allowed: [0, 1, 2],
                                                  strategy: .temperature(1), rng: &rng))
        XCTAssertEqual(picked.id, 1)
        let degenerate = try XCTUnwrap(Sampler.choose(logits: [.nan, -.infinity, 2], allowed: [0, 1],
                                                      strategy: .greedy, rng: &rng))
        XCTAssertEqual(degenerate.id, 0)
        XCTAssertTrue(degenerate.degenerate)
        XCTAssertEqual(degenerate.maskedMass, 1, accuracy: 1e-12)
    }

    func testRejectsEmptyOrOutOfRangeAllowedSets() {
        var rng = SplitMix64(seed: 0)
        XCTAssertNil(Sampler.choose(logits: [0, 0], allowed: [], strategy: .greedy, rng: &rng))
        XCTAssertNil(Sampler.choose(logits: [0, 0], allowed: [0, 2], strategy: .greedy, rng: &rng))
    }

    /// Temperature sampling follows the renormalised distribution over the
    /// allowed set: with logits ln(3), 0 and a third token masked, the first
    /// should win ~75% of 20,000 draws.
    func testTemperatureSamplingFollowsTheRenormalisedDistribution() {
        var rng = SplitMix64(seed: 42)
        var first = 0
        let draws = 20_000
        for _ in 0..<draws {
            if Sampler.choose(logits: [Float(log(3.0)), 0, 10], allowed: [0, 1], strategy: .temperature(1), rng: &rng)?.id == 0 {
                first += 1
            }
        }
        XCTAssertEqual(Double(first) / Double(draws), 0.75, accuracy: 0.015)
    }

    func testSplitMixIsDeterministicPerSeedAndDiffersAcrossSeeds() {
        var a = SplitMix64(seed: 7)
        var b = SplitMix64(seed: 7)
        var c = SplitMix64(seed: 8)
        let seqA = (0..<5).map { _ in a.next() }
        XCTAssertEqual(seqA, (0..<5).map { _ in b.next() })
        XCTAssertNotEqual(seqA, (0..<5).map { _ in c.next() })
        // Pinned first output: catches an accidental change to the mixing constants,
        // which would silently change every documented demo output.
        var pinned = SplitMix64(seed: 0)
        XCTAssertEqual(pinned.next(), 0xE220_A839_7B1D_CDAF)
    }
}

final class VocabularyTests: XCTestCase {
    func testValidation() {
        XCTAssertThrowsError(try Vocabulary(tokens: [])) { XCTAssertEqual($0 as? Vocabulary.Error, .empty) }
        XCTAssertThrowsError(try Vocabulary(tokens: ["a", ""])) { XCTAssertEqual($0 as? Vocabulary.Error, .emptyToken(index: 1)) }
        XCTAssertThrowsError(try Vocabulary(tokens: ["a", "b", "a"])) {
            XCTAssertEqual($0 as? Vocabulary.Error, .duplicateToken(text: "a", firstIndex: 0, secondIndex: 2))
        }
    }

    func testIDsAndEOS() throws {
        let vocabulary = try Vocabulary(tokens: ["a", "bc"])
        XCTAssertEqual(vocabulary.eosID, 2)
        XCTAssertEqual(vocabulary.size, 3)
        XCTAssertEqual(vocabulary.bytes(of: 1), Array("bc".utf8))
        XCTAssertNil(vocabulary.bytes(of: 2))
        XCTAssertNil(vocabulary.bytes(of: -1))
        XCTAssertNil(vocabulary.bytes(of: 99))
    }

    /// `demo(corpus:)` is public and must not trap on hostile input: a corpus
    /// with far more distinct words than `maxCorpusTokens` is truncated.
    func testDemoVocabularyIsBoundedForHugeCorpora() {
        let corpus = (0..<20_000).map { "w\($0)" }.joined(separator: " ")
        let vocabulary = Vocabulary.demo(corpus: corpus)
        XCTAssertLessThanOrEqual(vocabulary.tokens.count, 96 + 20 + Vocabulary.maxCorpusTokens)
        XCTAssertGreaterThan(vocabulary.tokens.count, Vocabulary.maxCorpusTokens)
    }

    func testDemoVocabularyCoversPrintableASCIIAndWhitespaceMerges() {
        let vocabulary = Fixtures.vocabulary
        XCTAssertTrue(Set(UInt8(0x20)...UInt8(0x7E)).isSubset(of: vocabulary.singleByteTokens))
        XCTAssertNotNil(vocabulary.tokens.firstIndex(of: Array("\": \"".utf8)), "a whitespace merge the grammar must reject")
        XCTAssertNotNil(vocabulary.tokens.firstIndex(of: Array("photo".utf8)))
    }
}
