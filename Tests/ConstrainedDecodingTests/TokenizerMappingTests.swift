import Foundation
import XCTest
@testable import ConstrainedDecoding

/// A toy "real" tokenizer with the awkward parts real ones have: a BOS special
/// at id 0, an empty piece, an EOS in the middle, a byte-fallback duplicate of
/// `{`, and a second end-of-turn id at the end.
enum ToyTokenizer {
    static var pieces: [TokenizerMapping.Piece] {
        var pieces: [TokenizerMapping.Piece] = [.special, .bytes([])]           // 0 BOS, 1 empty
        pieces += (UInt8(0x20)...UInt8(0x7E)).map { .bytes([$0]) }              // 2...96
        pieces.append(.endOfSequence)                                            // 97 mid-vocab EOS
        pieces.append(.bytes([0x7B]))                                            // 98 byte-fallback `{`
        pieces.append(.bytes(Array("true".utf8)))                                // 99
        pieces.append(.endOfSequence)                                            // 100 end-of-turn
        return pieces
    }
    static let braceID = 2 + Int(0x7B - 0x20)
    static let fallbackBraceID = 98
}

/// Answers with model-width logits and records the model ids it was fed.
actor ModelLog { var contexts: [[Int]] = []; func record(_ c: [Int]) { contexts.append(c) } }

struct ToyBackend: ModelBackend {
    let width: Int
    let log: ModelLog
    func logits(afterModelTokens context: [Int]) async throws -> [Float] {
        await log.record(context)
        var logits = [Float](repeating: 0, count: width)
        if width > 99 { logits[99] = 5 }                 // likes "true"
        if width > ToyTokenizer.fallbackBraceID { logits[ToyTokenizer.fallbackBraceID] = 3 }
        return logits
    }
}

final class TokenizerMappingTests: XCTestCase {
    func testMapsSpecialsDuplicatesAndMidVocabularyEOS() throws {
        let mapping = try TokenizerMapping(pieces: ToyTokenizer.pieces)
        XCTAssertEqual(mapping.modelVocabularySize, 101)
        XCTAssertEqual(mapping.vocabulary.tokens.count, 96, "95 printable bytes + `true`; the duplicate `{` merges")
        let brace = try XCTUnwrap(mapping.vocabulary.tokens.firstIndex(of: [0x7B]))
        XCTAssertEqual(mapping.modelID(forEngineID: brace), ToyTokenizer.braceID)
        XCTAssertEqual(mapping.modelID(forEngineID: mapping.vocabulary.eosID), 97)
        XCTAssertNil(mapping.modelID(forEngineID: 10_000))
    }

    /// Duplicate spellings merge by log-sum-exp; both EOS ids merge into the
    /// engine's single EOS; specials and empty pieces vanish.
    func testLogitsMergeIsLogSumExp() throws {
        let mapping = try TokenizerMapping(pieces: ToyTokenizer.pieces)
        var model = [Float](repeating: -.infinity, count: 101)
        model[ToyTokenizer.braceID] = 0
        model[ToyTokenizer.fallbackBraceID] = 0
        model[97] = 1
        model[100] = 1
        model[0] = 50 // BOS: must not leak anywhere
        let engine = try mapping.engineLogits(fromModel: model)
        XCTAssertEqual(engine.count, mapping.vocabulary.size)
        let brace = try XCTUnwrap(mapping.vocabulary.tokens.firstIndex(of: [0x7B]))
        XCTAssertEqual(Double(engine[brace]), log(2.0), accuracy: 1e-6)
        XCTAssertEqual(Double(engine[mapping.vocabulary.eosID]), 1 + log(2.0), accuracy: 1e-6)
        XCTAssertFalse(engine.contains(50))
    }

    func testRejectsBadInputs() throws {
        XCTAssertThrowsError(try TokenizerMapping(pieces: [.special, .endOfSequence])) {
            XCTAssertEqual($0 as? TokenizerMapping.Error, .noBytePieces)
        }
        XCTAssertThrowsError(try TokenizerMapping(pieces: [.bytes([0x41])])) {
            XCTAssertEqual($0 as? TokenizerMapping.Error, .noEndOfSequence)
        }
        let mapping = try TokenizerMapping(pieces: ToyTokenizer.pieces)
        XCTAssertThrowsError(try mapping.engineLogits(fromModel: [0, 0])) {
            XCTAssertEqual($0 as? TokenizerMapping.Error, .logitsWidthMismatch(expected: 101, actual: 2))
        }
    }

    /// End to end: a schema-constrained decode through the mapping feeds the
    /// backend model ids (never engine ids), and the output is valid.
    func testEndToEndDecodeThroughTheMapping() async throws {
        let mapping = try TokenizerMapping(pieces: ToyTokenizer.pieces)
        let log = ModelLog()
        let model = MappedModel(mapping: mapping, backend: ToyBackend(width: 101, log: log))
        let schema = Schema.object([.init("ok", .boolean)])
        let decoder = try ConstrainedDecoder(constraint: JSONSchemaConstraint(schema), vocabulary: mapping.vocabulary)
        let result = try await decoder.decode(using: model)
        XCTAssertEqual(result.text, #"{"ok":true}"#)
        let contexts = await log.contexts
        XCTAssertFalse(contexts.isEmpty)
        let fed = Set(contexts.flatMap { $0 })
        XCTAssertTrue(fed.isSubset(of: Set(0..<101)))
        XCTAssertFalse(fed.contains(0), "BOS is a special and is never produced")
        XCTAssertTrue(fed.contains(ToyTokenizer.braceID), "engine `{` is fed back as its first model spelling")
    }

    func testWrongBackendWidthSurfacesAsAnError() async throws {
        let mapping = try TokenizerMapping(pieces: ToyTokenizer.pieces)
        let model = MappedModel(mapping: mapping, backend: ToyBackend(width: 50, log: ModelLog()))
        let decoder = try ConstrainedDecoder(constraint: JSONSchemaConstraint(.boolean), vocabulary: mapping.vocabulary)
        do {
            _ = try await decoder.decode(using: model)
            XCTFail("expected a width error")
        } catch let error as TokenizerMapping.Error {
            XCTAssertEqual(error, .logitsWidthMismatch(expected: 101, actual: 50))
        }
    }
}
