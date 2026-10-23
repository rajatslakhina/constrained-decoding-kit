import Foundation
import XCTest
@testable import ConstrainedDecoding

final class ValidatorTests: XCTestCase {
    func testAcceptsAnyValidEncodingNotJustCanonical() {
        let pretty = """
        {
          "labels" : [ "ui" ],
          "blocking": false,
          "estimate_hours": 0,
          "priority": "medium",
          "title": "Tab \\"bar\\" \\u00e9 caf\u{E9} \\n"
        }
        """
        XCTAssertEqual(SchemaValidator.validate(pretty, against: Fixtures.ticket), .valid)
    }

    func testRejectsEachKindOfViolation() {
        let base = #"{"title":"T","priority":"low","estimate_hours":1,"blocking":true,"labels":["ui"]}"#
        func variant(_ from: String, _ to: String) -> String { base.replacingOccurrences(of: from, with: to) }
        let invalid: [String] = [
            "Sure! " + base,
            base + " thanks",
            variant(#""labels":["ui"]"#, #""labels":["ui",]"#),
            variant(#""labels":["ui"]"#, #""labels":[]"#),
            variant(#""labels":["ui"]"#, #""labels":["ui","ui","ui","ui"]"#),
            variant(#""priority":"low""#, #""priority":"P1""#),
            variant(#""estimate_hours":1"#, #""estimate_hours":1.5"#),
            variant(#""estimate_hours":1"#, #""estimate_hours":01"#),
            variant(#""estimate_hours":1"#, #""estimate_hours":1e3"#),
            variant(#""estimate_hours":1"#, #""estimate_hours":-1"#),
            variant(#""estimate_hours":1"#, #""estimate_hours":1234"#),
            variant(#""blocking":true"#, #""blocking":"yes""#),
            variant(#""title":"T""#, #""title":"""#),
            variant(#""title":"T""#, #""title":"T","title":"U""#),
            variant(#""title":"T","#, ""),
            variant("}", #","extra":1}"#),
            variant(#""title":"T""#, "\"title\":\"a\u{01}b\""),
            variant(#""title":"T""#, #""title":"bad \q""#),
            variant(#""title":"T""#, #""title":"unterminated"#),
            variant(#""title":"T""#, #""title":"\u+041""#),
        ]
        for text in invalid {
            XCTAssertFalse(SchemaValidator.validate(text, against: Fixtures.ticket).isValid, text)
        }
    }

    func testStringLengthCountsUnicodeScalarsNotBytes() {
        let schema = Schema.string(minLength: 0, maxLength: 3)
        XCTAssertTrue(SchemaValidator.validate("\"\u{E9}\u{E9}\u{E9}\"", against: schema).isValid) // 6 bytes, 3 scalars
        XCTAssertFalse(SchemaValidator.validate("\"\u{E9}\u{E9}\u{E9}\u{E9}\"", against: schema).isValid)
    }

    func testRejectsInvalidUTF8() {
        let bytes: [UInt8] = [0x22, 0xC3, 0x28, 0x22] // truncated 2-byte sequence
        XCTAssertFalse(SchemaValidator.validate(bytes, against: .string(maxLength: 10)).isValid)
    }

    func testReportsTheOffsetOfTheFirstProblem() {
        guard case .invalid(_, let offset) = SchemaValidator.validate("Sure! {}", against: .object([])) else {
            return XCTFail("expected invalid")
        }
        XCTAssertEqual(offset, 0)
    }
}
