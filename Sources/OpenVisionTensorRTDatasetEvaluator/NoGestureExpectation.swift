/// A frame window in which the dataset explicitly says that no gesture was
/// performed.
///
/// This is distinct from an absent annotation. An absent annotation is
/// unscored; an explicit no-gesture window passes only when the recognizer
/// commits to nothing in the evaluated run.
struct NoGestureExpectation: Sendable {
    enum ParseError: Error, Equatable {
        case fieldCount(Int)
        case emptyRecordID
    }

    enum ResolutionError: Error, Equatable {
        case unknownRecordID(String)
        case invertedWindow(first: String, last: String)
    }

    struct Resolved: Sendable {
        let expectation: NoGestureExpectation
        let firstFrameIndex: Int
        let lastFrameIndex: Int

        func contains(frameIndex: Int) -> Bool {
            frameIndex >= firstFrameIndex && frameIndex <= lastFrameIndex
        }
    }

    let firstRecordID: String
    let lastRecordID: String

    /// Parses `#expectNone <first-record-id> <last-record-id>`.
    static func parse(
        fields: [Substring]
    ) throws(ParseError) -> Self {
        guard fields.count == 3 else {
            throw .fieldCount(fields.count)
        }
        guard !fields[1].isEmpty, !fields[2].isEmpty else {
            throw .emptyRecordID
        }
        return Self(
            firstRecordID: String(fields[1]),
            lastRecordID: String(fields[2])
        )
    }

    func resolved(
        frameIndexByRecordID: [String: Int]
    ) throws(ResolutionError) -> Resolved {
        guard let first = frameIndexByRecordID[firstRecordID] else {
            throw .unknownRecordID(firstRecordID)
        }
        guard let last = frameIndexByRecordID[lastRecordID] else {
            throw .unknownRecordID(lastRecordID)
        }
        guard first <= last else {
            throw .invertedWindow(first: firstRecordID, last: lastRecordID)
        }
        return Resolved(
            expectation: self,
            firstFrameIndex: first,
            lastFrameIndex: last
        )
    }

    var json: String {
        "{\"firstRecordID\":" + jsonString(firstRecordID)
            + ",\"lastRecordID\":" + jsonString(lastRecordID) + "}"
    }
}
