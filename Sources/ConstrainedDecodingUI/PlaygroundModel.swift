#if canImport(SwiftUI) && canImport(Observation)
import ConstrainedDecoding
import Observation

/// Everything the playground needs, supplied by the host app: the schema the
/// feature must produce, the text the stand-in model "learned", and a title.
public struct PlaygroundScenario: Sendable {
    public let title: String
    public let schema: Schema
    public let reference: String
    public let vocabulary: Vocabulary

    public init(title: String, schema: Schema, reference: String, vocabulary: Vocabulary? = nil) {
        self.title = title
        self.schema = schema
        self.reference = reference
        self.vocabulary = vocabulary ?? Vocabulary.demo(corpus: reference)
    }
}

/// One side of the side-by-side comparison.
public struct LaneReport: Sendable, Equatable {
    public let text: String
    public let outcome: SchemaValidator.Outcome
    public let termination: Termination
    public let tokens: Int
    public let stats: DecodingStats
    /// Per-step masked probability mass; `nil` entries are forced steps.
    public let maskedMass: [Double?]

    init(_ result: DecodingResult, schema: Schema) {
        text = result.text
        outcome = SchemaValidator.validate(result.bytes, against: schema)
        termination = result.termination
        tokens = result.tokens.count
        stats = result.stats
        maskedMass = result.steps.map(\.maskedMass)
    }

    public var meanMaskedMass: Double {
        let values = maskedMass.compactMap { $0 }
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    public var cacheHitRate: Double {
        let total = stats.maskCacheHits + stats.maskCacheMisses
        guard total > 0 else { return 0 }
        return Double(stats.maskCacheHits) / Double(total)
    }
}

/// Schema-valid outputs per lane across a seed sweep.
public struct SweepResult: Sendable, Equatable {
    public let unconstrainedValid: Int
    public let constrainedValid: Int
    public let total: Int
}

public enum PlaygroundSampling: String, CaseIterable, Identifiable, Sendable {
    case greedy = "Greedy"
    case warm = "T = 0.8"
    case hot = "T = 1.4"

    public var id: String { rawValue }

    var strategy: SamplingStrategy {
        switch self {
        case .greedy: .greedy
        case .warm: .temperature(0.8)
        case .hot: .temperature(1.4)
        }
    }
}

@MainActor
@Observable
public final class PlaygroundModel {
    public let scenario: PlaygroundScenario
    public var sampling: PlaygroundSampling = .greedy
    public var seed: Int = 0
    public private(set) var unconstrained: LaneReport?
    public private(set) var constrained: LaneReport?
    public private(set) var errorMessage: String?
    public private(set) var isRunning = false
    public private(set) var sweep: SweepResult?
    /// Seeds finished in the running sweep, for progress display.
    public private(set) var sweepProgress = 0
    public let sweepSeedCount = 20

    private let constrainedDecoder: ConstrainedDecoder<JSONSchemaConstraint>?
    private let unconstrainedDecoder: ConstrainedDecoder<UnconstrainedText>?
    /// Built once (its substring index is the expensive part) and reseeded per run.
    private let baseModel: ReferenceTextModel

    public init(scenario: PlaygroundScenario) {
        self.scenario = scenario
        self.baseModel = ReferenceTextModel(reference: scenario.reference)
        var constrained: ConstrainedDecoder<JSONSchemaConstraint>?
        var unconstrained: ConstrainedDecoder<UnconstrainedText>?
        var failure: String?
        do {
            let constraint = try JSONSchemaConstraint(scenario.schema)
            constrained = try ConstrainedDecoder(constraint: constraint, vocabulary: scenario.vocabulary)
            // Two decoder instances of the same engine (each owns its mask cache),
            // same model, seed and sampler: the only variable is the mask.
            unconstrained = try ConstrainedDecoder(
                constraint: UnconstrainedText(maxOutputBytes: max(512, constraint.maxOutputBytes)),
                vocabulary: scenario.vocabulary)
        } catch {
            constrained = nil
            unconstrained = nil
            failure = "Scenario rejected: \(error)"
        }
        self.constrainedDecoder = constrained
        self.unconstrainedDecoder = unconstrained
        self.errorMessage = failure
    }

    /// Seeds are clamped to `0...9_999` by the UI stepper; the conversion below
    /// never traps because a negative `Int` maps through its bit pattern.
    private var configuration: DecodingConfiguration {
        DecodingConfiguration(strategy: sampling.strategy, seed: UInt64(bitPattern: Int64(seed)))
    }

    public func run() async {
        guard let constrainedDecoder, let unconstrainedDecoder, !isRunning else { return }
        isRunning = true
        defer { isRunning = false }
        let model = baseModel.reseeded(configuration.seed)
        do {
            let config = configuration
            async let free = unconstrainedDecoder.decode(using: model, configuration: config)
            async let masked = constrainedDecoder.decode(using: model, configuration: config)
            let (freeResult, maskedResult) = try await (free, masked)
            unconstrained = LaneReport(freeResult, schema: scenario.schema)
            constrained = LaneReport(maskedResult, schema: scenario.schema)
            errorMessage = nil
        } catch {
            errorMessage = "Decode failed: \(error)"
        }
    }

    /// Decodes both lanes for ``sweepSeedCount`` seeds at T = 1.4 and counts
    /// schema-valid outputs.
    public func runSweep() async {
        guard let constrainedDecoder, let unconstrainedDecoder, !isRunning else { return }
        isRunning = true
        defer { isRunning = false }
        var freeValid = 0
        var maskedValid = 0
        let seeds = 0..<sweepSeedCount
        sweepProgress = 0
        do {
            for seed in seeds {
                let config = DecodingConfiguration(strategy: .temperature(1.4), seed: UInt64(seed))
                let model = baseModel.reseeded(UInt64(seed))
                let free = try await unconstrainedDecoder.decode(using: model, configuration: config)
                let masked = try await constrainedDecoder.decode(using: model, configuration: config)
                if SchemaValidator.validate(free.bytes, against: scenario.schema).isValid { freeValid += 1 }
                if SchemaValidator.validate(masked.bytes, against: scenario.schema).isValid { maskedValid += 1 }
                sweepProgress = seed + 1
            }
            sweep = SweepResult(unconstrainedValid: freeValid, constrainedValid: maskedValid, total: seeds.count)
            errorMessage = nil
        } catch {
            errorMessage = "Sweep failed: \(error)"
        }
    }
}
#endif
