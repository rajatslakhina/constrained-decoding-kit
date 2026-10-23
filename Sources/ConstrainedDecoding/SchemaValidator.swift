/// An independent checker: does this text parse as JSON and satisfy the schema?
///
/// It shares no code with ``JSONSchemaConstraint`` on purpose. It is a
/// conventional recursive-descent parser that accepts *any* valid JSON
/// (whitespace, escapes, any key order, non-ASCII text), so it is strictly more
/// permissive than the matcher's canonical form. That asymmetry is the test
/// oracle: everything the matcher lets through must pass here, and a matcher
/// with a deliberately planted hole must fail here (both are tested).
public enum SchemaValidator {
    public enum Outcome: Sendable, Hashable {
        case valid
        case invalid(reason: String, byteOffset: Int)

        public var isValid: Bool { self == .valid }
    }

    /// Recursion is driven by the schema, so depth is bounded by schema depth;
    /// this cap only guards against a pathologically deep hand-built schema.
    static let maxDepth = 64

    public static func validate(_ text: String, against schema: Schema) -> Outcome {
        validate(Array(text.utf8), against: schema)
    }

    public static func validate(_ bytes: [UInt8], against schema: Schema) -> Outcome {
        var parser = Parser(bytes: bytes)
        do {
            parser.skipWhitespace()
            try parser.value(schema, path: "$", depth: 0)
            parser.skipWhitespace()
            guard parser.atEnd else { throw Issue(reason: "unexpected content after the document", offset: parser.offset) }
            return .valid
        } catch let issue as Issue {
            return .invalid(reason: issue.reason, byteOffset: issue.offset)
        } catch {
            return .invalid(reason: "\(error)", byteOffset: parser.offset)
        }
    }

    private struct Issue: Error {
        let reason: String
        let offset: Int
    }

    private struct Parser {
        let bytes: [UInt8]
        var offset = 0

        init(bytes: [UInt8]) { self.bytes = bytes }

        var atEnd: Bool { offset >= bytes.count }
        var peek: UInt8? { bytes.indices.contains(offset) ? bytes[offset] : nil }

        mutating func skipWhitespace() {
            while let byte = peek, byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D { offset += 1 }
        }

        func fail(_ reason: String) -> Issue { Issue(reason: reason, offset: offset) }

        mutating func expect(_ byte: UInt8, _ what: String) throws {
            guard peek == byte else { throw fail("expected \(what)") }
            offset += 1
        }

        mutating func value(_ schema: Schema, path: String, depth: Int) throws {
            guard depth < SchemaValidator.maxDepth else { throw fail("\(path): nesting too deep") }
            switch schema {
            case .object(let properties):
                try object(properties, path: path, depth: depth)
            case .string(let minLength, let maxLength):
                let (_, count) = try string(path: path)
                guard count >= minLength, count <= maxLength else {
                    throw fail("\(path): string length \(count) outside \(minLength)...\(maxLength)")
                }
            case .integer(let maxDigits, let allowNegative):
                try integer(maxDigits: maxDigits, allowNegative: allowNegative, path: path)
            case .boolean:
                if bytes[offset...].starts(with: Array("true".utf8)) { offset += 4 }
                else if bytes[offset...].starts(with: Array("false".utf8)) { offset += 5 }
                else { throw fail("\(path): expected true or false") }
            case .enumeration(let values):
                let start = offset
                let (decoded, _) = try string(path: path)
                guard values.contains(decoded) else {
                    throw Issue(reason: "\(path): \"\(decoded)\" is not one of \(values)", offset: start)
                }
            case .array(let item, let minItems, let maxItems):
                try expect(0x5B, "[ at \(path)")
                skipWhitespace()
                var count = 0
                if peek == 0x5D {
                    offset += 1
                } else {
                    while true {
                        skipWhitespace()
                        try value(item, path: "\(path)[\(count)]", depth: depth + 1)
                        count += 1
                        skipWhitespace()
                        if peek == 0x2C { offset += 1; continue }
                        try expect(0x5D, ", or ] at \(path)")
                        break
                    }
                }
                guard count >= minItems, count <= maxItems else {
                    throw fail("\(path): \(count) items outside \(minItems)...\(maxItems)")
                }
            }
        }

        mutating func object(_ properties: [Schema.Property], path: String, depth: Int) throws {
            try expect(0x7B, "{ at \(path)")
            var seen = Set<String>()
            skipWhitespace()
            if peek == 0x7D {
                offset += 1
            } else {
                while true {
                    skipWhitespace()
                    let keyOffset = offset
                    let (key, _) = try string(path: path)
                    guard let property = properties.first(where: { $0.name == key }) else {
                        throw Issue(reason: "\(path): unexpected key \"\(key)\"", offset: keyOffset)
                    }
                    guard seen.insert(key).inserted else {
                        throw Issue(reason: "\(path): duplicate key \"\(key)\"", offset: keyOffset)
                    }
                    skipWhitespace()
                    try expect(0x3A, ": after key at \(path)")
                    skipWhitespace()
                    try value(property.schema, path: "\(path).\(key)", depth: depth + 1)
                    skipWhitespace()
                    if peek == 0x2C { offset += 1; continue }
                    try expect(0x7D, ", or } at \(path)")
                    break
                }
            }
            let missing = properties.map(\.name).filter { !seen.contains($0) }
            guard missing.isEmpty else { throw fail("\(path): missing \(missing)") }
        }

        /// Returns the decoded string and its length in Unicode scalars.
        mutating func string(path: String) throws -> (String, Int) {
            try expect(0x22, "\" at \(path)")
            var raw: [UInt8] = []
            while true {
                guard let byte = peek else { throw fail("\(path): unterminated string") }
                offset += 1
                switch byte {
                case 0x22:
                    var scalars = String.UnicodeScalarView()
                    var decoder = UTF8()
                    var iterator = raw.makeIterator()
                    decoding: while true {
                        switch decoder.decode(&iterator) {
                        case .scalarValue(let scalar): scalars.append(scalar)
                        case .emptyInput: break decoding
                        case .error: throw fail("\(path): invalid UTF-8 in string")
                        }
                    }
                    return (String(scalars), scalars.count)
                case 0x5C:
                    guard let escape = peek else { throw fail("\(path): unterminated escape") }
                    offset += 1
                    switch escape {
                    case 0x22, 0x5C, 0x2F: raw.append(escape)
                    case 0x62: raw.append(0x08)
                    case 0x66: raw.append(0x0C)
                    case 0x6E: raw.append(0x0A)
                    case 0x72: raw.append(0x0D)
                    case 0x74: raw.append(0x09)
                    case 0x75:
                        // Exactly four hex digits; `UInt32(_:radix:)` alone would also take "+041".
                        guard offset + 4 <= bytes.count,
                              bytes[offset..<offset + 4].allSatisfy({ ($0 >= 0x30 && $0 <= 0x39) || ($0 >= 0x41 && $0 <= 0x46) || ($0 >= 0x61 && $0 <= 0x66) }),
                              let code = UInt32(String(decoding: bytes[offset..<offset + 4], as: UTF8.self), radix: 16)
                        else { throw fail("\(path): bad \\u escape") }
                        offset += 4
                        // Lone surrogates are not scalars; substitute U+FFFD rather than reject.
                        let scalar = Unicode.Scalar(code) ?? "\u{FFFD}"
                        raw.append(contentsOf: Array(String(scalar).utf8))
                    default:
                        throw fail("\(path): invalid escape")
                    }
                case 0x00..<0x20:
                    throw fail("\(path): raw control character in string")
                default:
                    raw.append(byte)
                }
            }
        }

        mutating func integer(maxDigits: Int, allowNegative: Bool, path: String) throws {
            if peek == 0x2D {
                guard allowNegative else { throw fail("\(path): negative not allowed") }
                offset += 1
            }
            var digits = 0
            let first = peek
            while let byte = peek, byte >= 0x30, byte <= 0x39 { offset += 1; digits += 1 }
            guard digits > 0 else { throw fail("\(path): expected an integer") }
            guard !(first == 0x30 && digits > 1) else { throw fail("\(path): leading zero") }
            guard digits <= maxDigits else { throw fail("\(path): \(digits) digits exceeds \(maxDigits)") }
            if let byte = peek, byte == 0x2E || byte == 0x65 || byte == 0x45 {
                throw fail("\(path): expected an integer, found a fraction or exponent")
            }
        }
    }
}
