import Foundation
import XCTest
@testable import ConstrainedDecoding

final class DecoderTests: XCTestCase {
    private func makeDecoder(_ schema: Schema = Fixtures.ticket, capacity: Int = 512) throws -> ConstrainedDecoder<JSONSchemaConstraint> {
        try ConstrainedDecoder(constraint: JSONSchemaConstraint(schema), vocabulary: Fixtures.vocabulary, maskCacheCapacity: capacity)
    }

    /// The exact output the README and the demo app show for greedy, seed 0.
    func testGreedyGoldenOutput() async throws {
        let decoder = try makeDecoder()
        let result = try await decoder.decode(using: ReferenceTextModel(reference: Fixtures.reference, seed: 0))
        XCTAssertEqual(result.text,
            #"{"title":"Crash when rotating the photo editor","priority":"high","estimate_hours":6,"blocking":true,"labels":["crash","editor","ipad"]}"#)
        XCTAssertEqual(result.termination, .endOfSequence)
        XCTAssertTrue(result.isAccepted)
        // Figures the README quotes.
        XCTAssertEqual(result.steps.count, 41)
        XCTAssertEqual(result.stats.forcedSteps, 7)
        XCTAssertEqual(result.stats.modelCalls, 34)
        // Masked mass per step — the figures the READMEs quote. Every step above
        // 99% is a step where the model wanted something illegal: prose at the
        // start (step 0), "P1" (high"), ".5" (the , after 6) and "yes" (true).
        let texts = result.steps.map { step in
            step.isEndOfSequence ? "<eos>" : String(decoding: Fixtures.vocabulary.bytes(of: step.tokenID) ?? [], as: UTF8.self)
        }
        let spikes = zip(texts, result.steps).enumerated()
            .filter { ($0.element.1.maskedMass ?? 0) > 0.99 }
            .map { "\($0.offset):\($0.element.0)" }
        XCTAssertEqual(spikes, ["0:{\"", "18:high\"", "24:,", "28:true"])
        XCTAssertEqual(texts[23], "6")
        // The free-text title (steps 3...13, "Crash" through "editor") stays under 6%.
        for index in 3...13 {
            XCTAssertLessThan(try XCTUnwrap(result.steps[index].maskedMass), 0.06, "step \(index) \(texts[index])")
        }
    }

    /// The unconstrained lane must actually fail, or the side-by-side proves nothing.
    func testUnconstrainedBaselineEmitsInvalidOutput() async throws {
        let decoder = try ConstrainedDecoder(constraint: UnconstrainedText(maxOutputBytes: 512), vocabulary: Fixtures.vocabulary)
        let result = try await decoder.decode(using: ReferenceTextModel(reference: Fixtures.reference, seed: 0))
        XCTAssertEqual(result.text, Fixtures.reference, "left alone, the model reproduces its reference verbatim")
        XCTAssertFalse(SchemaValidator.validate(result.bytes, against: Fixtures.ticket).isValid)
    }

    /// Pins the seed-sweep numbers the README quotes and the demo's "Run sweep"
    /// button computes: 20 seeds at T = 1.4, both lanes, same model and sampler.
    func testDocumentedSeedSweep() async throws {
        let constrained = try makeDecoder()
        let unconstrained = try ConstrainedDecoder(constraint: UnconstrainedText(maxOutputBytes: 512), vocabulary: Fixtures.vocabulary)
        var freeValid = 0
        var maskedValid = 0
        for seed in UInt64(0)..<20 {
            let model = ReferenceTextModel(reference: Fixtures.reference, seed: seed)
            let config = DecodingConfiguration(strategy: .temperature(1.4), seed: seed)
            let free = try await unconstrained.decode(using: model, configuration: config)
            let masked = try await constrained.decode(using: model, configuration: config)
            if SchemaValidator.validate(free.bytes, against: Fixtures.ticket).isValid { freeValid += 1 }
            if SchemaValidator.validate(masked.bytes, against: Fixtures.ticket).isValid { maskedValid += 1 }
        }
        XCTAssertEqual(freeValid, 0)
        XCTAssertEqual(maskedValid, 20)
    }

    /// The headline property: across seeds and samplers, every constrained output
    /// parses and satisfies the schema per the *independent* validator.
    func testConstrainedOutputIsAlwaysSchemaValid() async throws {
        let decoder = try makeDecoder()
        let strategies: [SamplingStrategy] = [.greedy, .temperature(0.7), .temperature(1.4), .temperature(3)]
        for strategy in strategies {
            for seed in UInt64(0)..<15 {
                let model = ReferenceTextModel(reference: Fixtures.reference, seed: seed)
                let result = try await decoder.decode(using: model, configuration: .init(strategy: strategy, seed: seed))
                XCTAssertEqual(result.termination, .endOfSequence)
                XCTAssertEqual(SchemaValidator.validate(result.bytes, against: Fixtures.ticket), .valid,
                               "\(strategy) seed \(seed): \(result.text)")
            }
        }
    }

    /// Validator non-vacuity: plant a hole in the matcher and the validator
    /// must catch the output — even though the broken constraint itself
    /// considers the document complete.
    func testValidatorCatchesALeakyConstraint() async throws {
        let leaky = try LeakyConstraint(inner: JSONSchemaConstraint(Fixtures.ticket))
        let decoder = try ConstrainedDecoder(constraint: leaky, vocabulary: Fixtures.vocabulary)
        let result = try await decoder.decode(using: HashLovingProvider())
        XCTAssertTrue(result.text.contains("#"), result.text)
        XCTAssertEqual(result.termination, .endOfSequence)
        XCTAssertTrue(result.isAccepted, "the broken constraint accepts its own output")
        XCTAssertFalse(SchemaValidator.validate(result.bytes, against: Fixtures.ticket).isValid)

        // Control: the real constraint under the same provider stays valid.
        let honest = try await makeDecoder().decode(using: HashLovingProvider())
        XCTAssertFalse(honest.text.contains("#"))
        XCTAssertTrue(SchemaValidator.validate(honest.bytes, against: Fixtures.ticket).isValid)
    }

    /// Termination does not depend on the model: an adversary that loves
    /// whitespace and long tokens and never wants EOS still ends in EOS,
    /// within `maxOutputBytes + 1` steps, with a valid document.
    func testTerminatesUnderAnAdversarialModel() async throws {
        for schema in [Fixtures.ticket, Fixtures.kitchenSink] {
            let decoder = try makeDecoder(schema)
            let constraint = try JSONSchemaConstraint(schema)
            let result = try await decoder.decode(using: AdversarialProvider())
            XCTAssertEqual(result.termination, .endOfSequence)
            XCTAssertLessThanOrEqual(result.steps.count, constraint.maxOutputBytes + 1)
            XCTAssertLessThanOrEqual(result.bytes.count, constraint.maxOutputBytes)
            // The adversary's favourites (whitespace merges like `": "`) were on offer and refused.
            XCTAssertNil(result.text.range(of: "\": "))
            XCTAssertEqual(SchemaValidator.validate(result.bytes, against: schema), .valid, result.text)
        }
    }

    /// The decoder's own step cap, exercised: a constraint whose `#` hole makes
    /// its `maxOutputBytes` a lie, and a model that wants `#` forever. Without
    /// the cap this would never return; with it, it stops at exactly
    /// `maxOutputBytes + 1` steps.
    func testDecoderStepCapHoldsEvenWhenTheConstraintBoundIsWrong() async throws {
        let leaky = try LeakyConstraint(inner: JSONSchemaConstraint(Fixtures.ticket))
        let decoder = try ConstrainedDecoder(constraint: leaky, vocabulary: Fixtures.vocabulary)
        let result = try await decoder.decode(using: AlwaysHashProvider())
        XCTAssertEqual(result.termination, .tokenBudgetExhausted)
        XCTAssertEqual(result.steps.count, leaky.maxOutputBytes + 1)
        XCTAssertEqual(result.stats.tokenBudget, leaky.maxOutputBytes + 1)
    }

    /// Default configuration = the constraint's bound, so a schema far wider
    /// than any fixed default (`maxLength: 4096`) still terminates in EOS
    /// against a model that never wants to stop.
    func testDefaultBudgetIsTheConstraintBound() async throws {
        let schema = Schema.string(maxLength: 4_096)
        let decoder = try makeDecoder(schema)
        let result = try await decoder.decode(using: AdversarialProvider())
        XCTAssertEqual(result.stats.tokenBudget, try JSONSchemaConstraint(schema).maxOutputBytes + 1)
        XCTAssertEqual(result.termination, .endOfSequence)
        XCTAssertEqual(SchemaValidator.validate(result.bytes, against: schema), .valid)
    }

    /// A saturated or dishonest bound still yields a finite budget, and the
    /// unconstrained baseline cannot be configured past the same ceiling.
    func testBudgetHasAnAbsoluteCeiling() {
        let defaults = DecodingConfiguration()
        XCTAssertEqual(defaults.effectiveBudget(constraintMaxOutputBytes: .max), DecodingConfiguration.absoluteMaxTokens)
        XCTAssertEqual(defaults.effectiveBudget(constraintMaxOutputBytes: 173), 174)
        XCTAssertEqual(defaults.effectiveBudget(constraintMaxOutputBytes: -5), 1)
        XCTAssertEqual(DecodingConfiguration(maxTokens: 10).effectiveBudget(constraintMaxOutputBytes: 173), 10)
        XCTAssertEqual(DecodingConfiguration(maxTokens: .max).effectiveBudget(constraintMaxOutputBytes: .max),
                       DecodingConfiguration.absoluteMaxTokens)
        XCTAssertEqual(UnconstrainedText(maxOutputBytes: .max).maxOutputBytes, SchemaLimits.maxEncodedBytes)
        XCTAssertEqual(UnconstrainedText(maxOutputBytes: -1).maxOutputBytes, 0)
    }

    func testCallerBudgetBelowTheBoundReportsExhaustion() async throws {
        let decoder = try makeDecoder()
        let result = try await decoder.decode(using: ReferenceTextModel(reference: Fixtures.reference),
                                              configuration: .init(maxTokens: 5))
        XCTAssertEqual(result.termination, .tokenBudgetExhausted)
        XCTAssertFalse(result.isAccepted)
        XCTAssertEqual(result.steps.count, 5)
        XCTAssertEqual(result.stats.tokenBudget, 5)
    }

    func testInvalidConfigurationsThrow() async throws {
        let decoder = try makeDecoder()
        let model = ReferenceTextModel(reference: Fixtures.reference)
        for config in [DecodingConfiguration(strategy: .temperature(0)), .init(strategy: .temperature(-1)),
                       .init(strategy: .temperature(.nan)), .init(strategy: .temperature(.infinity)), .init(maxTokens: 0)] {
            do {
                _ = try await decoder.decode(using: model, configuration: config)
                XCTFail("expected invalidConfiguration for \(config)")
            } catch let error as ConstrainedDecodingError {
                guard case .invalidConfiguration = error else { return XCTFail("\(error)") }
            }
        }
    }

    func testLogitsShapeMismatchIsATypedError() async throws {
        let decoder = try makeDecoder()
        do {
            _ = try await decoder.decode(using: WrongShapeProvider())
            XCTFail("expected a shape error")
        } catch let error as ConstrainedDecodingError {
            // Step 0 has several allowed tokens (`{`, `{"`), so the model is consulted at once.
            XCTAssertEqual(error, .logitsShapeMismatch(expected: Fixtures.vocabulary.size,
                                                       actual: Fixtures.vocabulary.size - 1, step: 0))
        }
    }

    func testVocabularyMissingAStructuralByteIsRejectedAtInit() throws {
        let tokens = Fixtures.vocabulary.tokens.filter { $0 != [0x7D] }
        let vocabulary = try Vocabulary(byteTokens: tokens)
        XCTAssertThrowsError(try ConstrainedDecoder(constraint: JSONSchemaConstraint(Fixtures.ticket), vocabulary: vocabulary)) {
            XCTAssertEqual($0 as? ConstrainedDecodingError, .vocabularyCannotExpress(bytes: [0x7D]))
        }
    }

    /// If a constraint under-declares its needs, the gap surfaces as a typed
    /// `deadEnd`, never a trap or an endless loop. Here every token containing
    /// `}` is removed, so nothing can close the object.
    func testUnderdeclaredCoverageSurfacesAsDeadEnd() async throws {
        let schema = Schema.object([.init("ok", .boolean)])
        let tokens = Fixtures.vocabulary.tokens.filter { !$0.contains(0x7D) }
        let vocabulary = try Vocabulary(byteTokens: tokens)
        let decoder = try ConstrainedDecoder(constraint: UnderdeclaredConstraint(inner: JSONSchemaConstraint(schema)),
                                             vocabulary: vocabulary)
        do {
            _ = try await decoder.decode(using: ReferenceTextModel(reference: Fixtures.reference))
            XCTFail("expected a dead end")
        } catch let error as ConstrainedDecodingError {
            guard case .deadEnd(_, let emitted) = error else { return XCTFail("\(error)") }
            // `{"ok":true` or `{"ok":false`: the dead end is exactly where `}` is needed.
            XCTAssertTrue([10, 11].contains(emitted), "emitted \(emitted)")
        }
    }

    /// Forced steps skip the model: calls == steps that had more than one option.
    func testForcedStepsDoNotCallTheModel() async throws {
        let counter = CallCounter()
        let provider = CountingProvider(inner: ReferenceTextModel(reference: Fixtures.reference), counter: counter)
        let result = try await makeDecoder().decode(using: provider)
        let calls = await counter.calls
        XCTAssertEqual(calls, result.stats.modelCalls)
        XCTAssertEqual(calls, result.steps.filter { $0.allowedCount > 1 }.count)
        XCTAssertEqual(result.stats.forcedSteps, result.steps.filter { $0.maskedMass == nil }.count)
        XCTAssertGreaterThan(result.stats.forcedSteps, 0)
    }

    /// Shared-cache reentrancy: eight decodes on one decoder, with a provider
    /// that yields inside every call and records how many decodes are mid-call
    /// at once. The test requires real overlap (max in flight > 1), a cache
    /// small enough to evict, and byte-identical results to sequential decodes
    /// on fresh, cache-less decoders.
    func testConcurrentDecodesOnASharedDecoderMatchSequentialRuns() async throws {
        let shared = try makeDecoder(capacity: 8)
        let tracker = InFlightTracker()
        let seeds = Array(UInt64(0)..<8)
        let base = ReferenceTextModel(reference: Fixtures.reference)
        let concurrent = try await withThrowingTaskGroup(of: (UInt64, DecodingResult).self) { group in
            for seed in seeds {
                group.addTask {
                    let model = YieldingProvider(inner: base.reseeded(seed), tracker: tracker)
                    return (seed, try await shared.decode(using: model, configuration: .init(strategy: .temperature(1.2), seed: seed)))
                }
            }
            var results: [UInt64: DecodingResult] = [:]
            for try await (seed, result) in group { results[seed] = result }
            return results
        }
        let overlap = await tracker.maxInFlight
        XCTAssertGreaterThan(overlap, 1, "decodes never overlapped, so this test proved nothing")
        let evictions = await shared.maskCacheEvictions
        XCTAssertGreaterThan(evictions, 0)
        for seed in seeds {
            let fresh = try makeDecoder(capacity: 0)
            let expected = try await fresh.decode(using: base.reseeded(seed),
                                                  configuration: .init(strategy: .temperature(1.2), seed: seed))
            XCTAssertEqual(concurrent[seed]?.bytes, expected.bytes, "seed \(seed)")
        }
    }

    /// The decoder's own cancellation check, isolated: the provider never
    /// suspends or checks cancellation, so only `Task.checkCancellation()` in
    /// the decode loop can stop this. The task is cancelled before decode starts.
    func testDecoderChecksCancellationEvenWhenTheModelIgnoresIt() async throws {
        let decoder = try makeDecoder(.string(maxLength: 4_096))
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await decoder.decode(using: AdversarialProvider())
        }
        do {
            _ = try await task.value
            XCTFail("expected cancellation")
        } catch is CancellationError {
            // expected
        }
    }

    /// A cooperative provider cancelled mid-decode also stops it.
    func testCancellationMidDecode() async throws {
        let decoder = try makeDecoder()
        let task = Task { try await decoder.decode(using: SlowProvider()) }
        try await Task.sleep(nanoseconds: 50_000_000)
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected cancellation")
        } catch is CancellationError {
            // expected
        }
    }

    /// A constraint whose `advance` is not a pure function (it starts refusing
    /// after a fixed number of calls) is caught by the decoder's re-advance
    /// check as a typed error instead of producing an invalid document.
    func testImpureConstraintIsATypedError() async throws {
        let inner = try JSONSchemaConstraint(Fixtures.ticket)
        let firstMaskCost = MaskComputer(constraint: inner, vocabulary: Fixtures.vocabulary)
            .compute(inner.initialState).advances
        let flaky = ExpiringConstraint(inner: inner, budget: CallBudget(limit: firstMaskCost))
        let decoder = try ConstrainedDecoder(constraint: flaky, vocabulary: Fixtures.vocabulary)
        do {
            _ = try await decoder.decode(using: ReferenceTextModel(reference: Fixtures.reference))
            XCTFail("expected inconsistentConstraint")
        } catch let error as ConstrainedDecodingError {
            guard case .inconsistentConstraint(let step, _) = error else { return XCTFail("\(error)") }
            XCTAssertEqual(step, 0)
        }
    }

    func testMaskCacheIsReusedAcrossDecodes() async throws {
        let decoder = try makeDecoder()
        let model = ReferenceTextModel(reference: Fixtures.reference)
        let first = try await decoder.decode(using: model)
        let second = try await decoder.decode(using: model)
        XCTAssertEqual(first.bytes, second.bytes)
        XCTAssertGreaterThan(first.stats.maskCacheMisses, 0)
        XCTAssertEqual(second.stats.maskCacheMisses, 0)
        XCTAssertEqual(second.stats.constraintAdvances, 0)
    }
}
