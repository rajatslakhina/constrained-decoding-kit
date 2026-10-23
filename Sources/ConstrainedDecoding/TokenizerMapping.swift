/// Adapts a real model's tokenizer to the engine's vocabulary model.
///
/// The engine wants a clean vocabulary: every id is a non-empty, unique byte
/// string, and EOS is one extra id at the end. Real tokenizers break all of it:
/// - special/control tokens (`<|begin_of_text|>`, padding, tool markers) decode
///   to no bytes and must never be sampled into JSON,
/// - end-of-sequence ids sit in the *middle* of the id space, and there may be
///   several (`<|eot_id|>` and `<|end_of_text|>`),
/// - byte-fallback pieces (SentencePiece's `<0x41>`) duplicate ordinary pieces,
/// - and the model's logit width is the model's vocabulary size.
///
/// The mapping gives each distinct byte string one engine id, remembers every
/// model id that spells it, and translates in both directions: model logits to
/// engine logits (merging aliases with log-sum-exp, which is the exact
/// probability of "any spelling of these bytes"), and engine context ids back
/// to model ids for the next forward pass.
///
/// Out of scope, and the caller's job: turning a piece's *display* form into
/// bytes (GPT-2-style byte-level BPE maps bytes to printable code points such
/// as `Ġ` for space; SentencePiece uses `▁`). Pass real bytes in `Piece.bytes`.
public struct TokenizerMapping: Sendable {
    public enum Piece: Sendable, Hashable {
        case bytes([UInt8])
        /// Never allowed by any mask (BOS, padding, tool or control tokens).
        case special
        /// Any id that ends generation.
        case endOfSequence
    }

    public enum Error: Swift.Error, Equatable, Sendable {
        case noBytePieces
        case noEndOfSequence
        case logitsWidthMismatch(expected: Int, actual: Int)
        case unknownEngineID(Int)
    }

    public let vocabulary: Vocabulary
    public let modelVocabularySize: Int
    /// For each engine byte-token id, the model ids that spell it (non-empty).
    let aliases: [[Int]]
    let endOfSequenceIDs: [Int]

    public init(pieces: [Piece]) throws {
        var firstEngineID: [[UInt8]: Int] = [:]
        var tokens: [[UInt8]] = []
        var aliases: [[Int]] = []
        var eos: [Int] = []
        for (modelID, piece) in pieces.enumerated() {
            switch piece {
            case .bytes(let bytes) where !bytes.isEmpty:
                if let engineID = firstEngineID[bytes], aliases.indices.contains(engineID) {
                    aliases[engineID].append(modelID)
                } else {
                    firstEngineID[bytes] = tokens.count
                    tokens.append(bytes)
                    aliases.append([modelID])
                }
            case .bytes, .special:
                continue // empty pieces behave like specials: never sampled
            case .endOfSequence:
                eos.append(modelID)
            }
        }
        guard !tokens.isEmpty else { throw Error.noBytePieces }
        guard !eos.isEmpty else { throw Error.noEndOfSequence }
        self.vocabulary = try Vocabulary(byteTokens: tokens)
        self.modelVocabularySize = pieces.count
        self.aliases = aliases
        self.endOfSequenceIDs = eos
    }

    /// The model id to feed back for an engine id (the first spelling seen).
    public func modelID(forEngineID id: Int) -> Int? {
        if id == vocabulary.eosID { return endOfSequenceIDs.first }
        return aliases.indices.contains(id) ? aliases[id].first : nil
    }

    /// Collapses model-width logits to engine-width logits.
    public func engineLogits(fromModel logits: [Float]) throws -> [Float] {
        guard logits.count == modelVocabularySize else {
            throw Error.logitsWidthMismatch(expected: modelVocabularySize, actual: logits.count)
        }
        var result = aliases.map { merge($0, logits) }
        result.append(merge(endOfSequenceIDs, logits))
        return result
    }

    /// log Σ exp(logit) over the aliases, computed stably. NaN counts as -inf.
    private func merge(_ ids: [Int], _ logits: [Float]) -> Float {
        let values = ids.compactMap { logits.indices.contains($0) ? Sampler.sanitize(logits[$0]) : nil }
        guard let merged = Sampler.logSumExp(values) else { return -.infinity }
        return Float(merged)
    }
}

/// A model backend that speaks model ids and returns model-width logits —
/// the shape of an MLX, llama.cpp or Core ML forward pass.
public protocol ModelBackend: Sendable {
    func logits(afterModelTokens context: [Int]) async throws -> [Float]
}

/// Bridges a ``ModelBackend`` into a ``LogitsProvider`` through a ``TokenizerMapping``.
public struct MappedModel<Backend: ModelBackend>: LogitsProvider {
    public let mapping: TokenizerMapping
    public let backend: Backend

    public init(mapping: TokenizerMapping, backend: Backend) {
        self.mapping = mapping
        self.backend = backend
    }

    public func logits(after context: [Int], vocabulary: Vocabulary) async throws -> [Float] {
        var modelContext: [Int] = []
        modelContext.reserveCapacity(context.count)
        for id in context {
            guard let modelID = mapping.modelID(forEngineID: id) else { throw TokenizerMapping.Error.unknownEngineID(id) }
            modelContext.append(modelID)
        }
        return try mapping.engineLogits(fromModel: try await backend.logits(afterModelTokens: modelContext))
    }
}
