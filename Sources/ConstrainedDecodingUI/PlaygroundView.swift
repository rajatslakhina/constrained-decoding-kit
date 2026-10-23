#if canImport(SwiftUI) && canImport(Observation)
import ConstrainedDecoding
import Foundation
import SwiftUI

/// Side-by-side: the same model, the same seed, the same sampler — with and
/// without the schema mask. Decodes once on appear so the comparison is
/// visible without a tap.
public struct ConstrainedDecodingPlayground: View {
    @State private var model: PlaygroundModel

    public init(scenario: PlaygroundScenario) {
        _model = State(initialValue: PlaygroundModel(scenario: scenario))
    }

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    controls
                    if let message = model.errorMessage {
                        Label(message, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red)
                            .font(.footnote)
                    }
                    LaneCard(title: "Unconstrained", subtitle: "model output as-is", report: model.unconstrained)
                    LaneCard(title: "Schema-constrained", subtitle: "token mask from the schema", report: model.constrained)
                    sweepSection
                }
                .padding()
            }
            .navigationTitle(model.scenario.title)
        }
        .task { await model.run() }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Sampling", selection: $model.sampling) {
                ForEach(PlaygroundSampling.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            HStack {
                Stepper("Seed \(model.seed)", value: $model.seed, in: 0...9_999)
                Button {
                    Task { await model.run() }
                } label: {
                    Label("Decode", systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isRunning)
            }
        }
    }

    private var sweepSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Seed sweep").font(.headline)
            Text("\(model.sweepSeedCount) seeds at T = 1.4. Counts outputs the independent validator accepts.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            if let sweep = model.sweep {
                HStack(spacing: 24) {
                    StatView(label: "Unconstrained valid", value: "\(sweep.unconstrainedValid)/\(sweep.total)")
                    StatView(label: "Constrained valid", value: "\(sweep.constrainedValid)/\(sweep.total)")
                }
            }
            if model.isRunning, model.sweepProgress > 0, model.sweepProgress < model.sweepSeedCount {
                ProgressView(value: Double(model.sweepProgress), total: Double(model.sweepSeedCount))
            }
            Button("Run sweep") { Task { await model.runSweep() } }
                .buttonStyle(.bordered)
                .disabled(model.isRunning)
        }
    }
}

private struct LaneCard: View {
    let title: String
    let subtitle: String
    let report: LaneReport?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading) {
                    Text(title).font(.headline)
                    Text(subtitle).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if let report { ValidityBadge(outcome: report.outcome) }
            }
            if let report {
                Text(report.text.isEmpty ? "(empty output)" : report.text)
                    .font(.system(.footnote, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
                if case .invalid(let reason, let offset) = report.outcome {
                    Text("Validator: \(reason) (byte \(offset))")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                HStack(spacing: 16) {
                    StatView(label: "Tokens", value: "\(report.tokens)")
                    StatView(label: "Model calls", value: "\(report.stats.modelCalls)")
                    StatView(label: "Forced", value: "\(report.stats.forcedSteps)")
                    StatView(label: "Mask cache", value: report.cacheHitRate.formatted(.percent.precision(.fractionLength(0))))
                }
                MaskedMassStrip(values: report.maskedMass)
            } else {
                ProgressView().frame(maxWidth: .infinity)
            }
        }
        .padding()
        .background(.background, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(.quaternary))
    }
}

/// One bar per decode step: how much probability the mask removed. Grey ticks
/// are forced steps, where the model was not consulted at all.
private struct MaskedMassStrip: View {
    let values: [Double?]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Probability mass removed by the mask, per step (line = 99%)")
                .font(.caption2)
                .foregroundStyle(.secondary)
            if values.isEmpty {
                Text("No steps").font(.caption2).foregroundStyle(.secondary)
            } else {
                GeometryReader { proxy in
                    let count = max(values.count, 1)
                    let width = max(proxy.size.width / CGFloat(count), 1)
                    ZStack(alignment: .top) {
                    HStack(alignment: .bottom, spacing: 0) {
                        ForEach(Array(values.enumerated()), id: \.offset) { _, value in
                            Rectangle()
                                .fill(value == nil ? Color.gray.opacity(0.35) : Color.orange)
                                .frame(width: width, height: barHeight(value, in: proxy.size.height))
                        }
                    }
                    .frame(maxHeight: .infinity, alignment: .bottom)
                    // 99% guide: bars that reach it are steps where the model's
                    // preferred continuation was almost entirely illegal.
                    Rectangle()
                        .fill(Color.red.opacity(0.6))
                        .frame(height: 1)
                        .offset(y: proxy.size.height * 0.01)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                    }
                }
                .frame(height: 44)
            }
        }
    }

    private func barHeight(_ value: Double?, in height: CGFloat) -> CGFloat {
        guard let value, value.isFinite else { return 3 }
        return max(1, CGFloat(min(1, max(0, value))) * height)
    }
}

private struct ValidityBadge: View {
    let outcome: SchemaValidator.Outcome

    var body: some View {
        let valid = outcome.isValid
        Label(valid ? "Valid" : "Invalid", systemImage: valid ? "checkmark.seal.fill" : "xmark.octagon.fill")
            .font(.caption.bold())
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .foregroundStyle(.white)
            .background(valid ? Color.green : Color.red, in: Capsule())
    }
}

private struct StatView: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value).font(.subheadline.monospacedDigit().bold())
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }
}
#endif
