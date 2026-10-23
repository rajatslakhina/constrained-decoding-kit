/// A tokenizer vocabulary: token id → bytes, plus an implicit end-of-sequence id.
///
/// Ids `0..<tokens.count` are byte tokens; ``eosID`` (`== tokens.count`) is EOS.
/// Logits from a model must therefore have exactly ``size`` entries.
public struct Vocabulary: Sendable, Hashable {
    public let tokens: [[UInt8]]

    public var eosID: Int { tokens.count }
    /// Number of logits a model must return: byte tokens plus EOS.
    public var size: Int { tokens.count + 1 }

    public static let maxTokens = 1 << 20
    /// Cap on tokens ``demo(corpus:)`` derives from its corpus.
    public static let maxCorpusTokens = 8_192

    public enum Error: Swift.Error, Equatable, Sendable {
        case empty
        case tooLarge(count: Int)
        case emptyToken(index: Int)
        /// Duplicate byte strings make "which id produced this text" ambiguous.
        case duplicateToken(text: String, firstIndex: Int, secondIndex: Int)
    }

    public init(tokens: [String]) throws {
        try self.init(byteTokens: tokens.map { Array($0.utf8) })
    }

    public init(byteTokens: [[UInt8]]) throws {
        guard !byteTokens.isEmpty else { throw Error.empty }
        guard byteTokens.count <= Self.maxTokens else { throw Error.tooLarge(count: byteTokens.count) }
        var firstIndex: [[UInt8]: Int] = [:]
        for (index, token) in byteTokens.enumerated() {
            guard !token.isEmpty else { throw Error.emptyToken(index: index) }
            if let first = firstIndex[token] {
                throw Error.duplicateToken(text: String(decoding: token, as: UTF8.self), firstIndex: first, secondIndex: index)
            }
            firstIndex[token] = index
        }
        self.tokens = byteTokens
    }

    /// Bytes of a token, or `nil` for EOS or an out-of-range id.
    public func bytes(of id: Int) -> [UInt8]? {
        tokens.indices.contains(id) ? tokens[id] : nil
    }

    public var singleByteTokens: Set<UInt8> {
        Set(tokens.lazy.filter { $0.count == 1 }.compactMap(\.first))
    }

    /// A small BPE-flavoured vocabulary: every printable ASCII byte as its own
    /// token (so nothing is ever unspellable), JSON punctuation merges — including
    /// whitespace-bearing ones like `": "` that the canonical grammar must reject —
    /// and whole words lifted from `corpus`, the way a real tokenizer learns merges
    /// from its training text.
    ///
    /// At most ``maxCorpusTokens`` tokens are taken from `corpus`; the rest of
    /// the corpus is ignored, which keeps the result bounded for any input.
    public static func demo(corpus: String) -> Vocabulary {
        var ordered: [[UInt8]] = []
        var seen = Set<[UInt8]>()
        var fromCorpus = 0
        var collectingCorpus = false
        func add(_ bytes: [UInt8]) {
            guard !bytes.isEmpty, bytes.count <= 24 else { return }
            if collectingCorpus {
                guard fromCorpus < maxCorpusTokens else { return }
            }
            guard seen.insert(bytes).inserted else { return }
            if collectingCorpus { fromCorpus += 1 }
            ordered.append(bytes)
        }
        for byte in UInt8(0x20)...UInt8(0x7E) { add([byte]) }
        add([0x0A])
        let merges = ["{\"", "\":", "\":\"", "\",\"", "\",", "\": \"", "\": ", ", \"", "\"}", "\"]", "[\"",
                      "\"]}", "],", "},", "true", "false", "null", ", ", ": ", "  "]
        for merge in merges { add(Array(merge.utf8)) }
        // Words and the punctuation glued to them, e.g. `"crash"` and `crash`.
        collectingCorpus = true
        var word: [UInt8] = []
        for byte in corpus.utf8 {
            let isWordByte = (byte >= 0x30 && byte <= 0x39) || (byte >= 0x41 && byte <= 0x5A)
                || (byte >= 0x61 && byte <= 0x7A) || byte == 0x5F
            if isWordByte {
                word.append(byte)
            } else {
                add(word)
                if !word.isEmpty, byte == JSONBytes.quote { add(word + [byte]); add([byte] + word) }
                word.removeAll(keepingCapacity: true)
            }
        }
        add(word)
        // Provably safe: `ordered` is non-empty (95 single bytes), deduplicated by
        // `seen`, free of empty tokens (`add` skips them), and at most
        // 96 + 20 + maxCorpusTokens = 8_308 entries, far below `maxTokens` (2^20).
        // Every condition `init(byteTokens:)` checks therefore holds.
        // swiftlint:disable:next force_try
        return try! Vocabulary(byteTokens: ordered)
    }
}

/// A byte trie over the vocabulary. Mask computation walks it once per state
/// and prunes a whole subtree the moment the constraint rejects a byte, so a
/// rejected `{` costs one `advance` call instead of one per token starting with `{`.
struct VocabularyTrie: Sendable {
    struct Node: Sendable {
        var edges: [(byte: UInt8, child: Int)] = []
        var token: Int?
    }

    private(set) var nodes: [Node]

    init(_ vocabulary: Vocabulary) {
        var built = [Node()]
        var children: [[UInt8: Int]] = [[:]]
        for (id, bytes) in vocabulary.tokens.enumerated() {
            var current = 0
            for byte in bytes {
                if let next = children[current][byte] {
                    current = next
                } else {
                    built.append(Node())
                    children.append([:])
                    let next = built.count - 1
                    children[current][byte] = next
                    current = next
                }
            }
            built[current].token = id
        }
        for index in built.indices {
            built[index].edges = children[index].map { (byte: $0.key, child: $0.value) }.sorted { $0.byte < $1.byte }
        }
        self.nodes = built
    }
}
