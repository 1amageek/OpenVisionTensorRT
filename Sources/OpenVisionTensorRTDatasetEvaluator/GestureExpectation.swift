import ActionRecognition

/// A gesture the dataset says the actor performed, and the window of frames it
/// was performed in.
///
/// Without expectations a run can only report that the pipeline executed.
/// Reporting that as a pass conflates "the code ran" with "the answer was
/// right", which are different claims and fail for different reasons.
struct GestureExpectation: Sendable {
    /// Direction stated in the gesture's own vocabulary, so a manifest cannot
    /// annotate a swipe as `clockwise`. The gesture is therefore implied by the
    /// direction rather than named separately, which removes the possibility of
    /// a manifest declaring a gesture and a direction that contradict.
    enum Direction: Sendable, Hashable {
        case horizontal(HorizontalDirection)
        case rotary(RotationDirection)

        init?(rawValue: String) {
            switch rawValue {
            case "left": self = .horizontal(.left)
            case "right": self = .horizontal(.right)
            case "clockwise": self = .rotary(.clockwise)
            case "counterclockwise": self = .rotary(.counterclockwise)
            default: return nil
            }
        }

        var rawValue: String {
            switch self {
            case .horizontal(.left): "left"
            case .horizontal(.right): "right"
            case .rotary(.clockwise): "clockwise"
            case .rotary(.counterclockwise): "counterclockwise"
            }
        }

        var identifier: GestureIdentifier {
            switch self {
            case .horizontal: .horizontalSwipe
            case .rotary: .rotaryManipulation
            }
        }
    }

    enum ParseError: Error, Equatable {
        case fieldCount(Int)
        case unknownDirection(String)
        case emptyRecordID
    }

    enum ResolutionError: Error, Equatable {
        /// A record named by an expectation is absent from the manifest.
        /// Dropping the expectation instead would silently shrink what the run
        /// is scored against.
        case unknownRecordID(String)
        case invertedWindow(first: String, last: String)
    }

    /// An expectation with its record identifiers resolved to positions in the
    /// evaluated sequence, which is the form scoring can compare against.
    struct Resolved: Sendable {
        let expectation: GestureExpectation
        /// Inclusive frame range the dataset attributes the gesture to.
        let firstFrameIndex: Int
        let lastFrameIndex: Int

        var identifier: GestureIdentifier { expectation.identifier }
        var direction: Direction { expectation.direction }

        func contains(frameIndex: Int) -> Bool {
            frameIndex >= firstFrameIndex && frameIndex <= lastFrameIndex
        }
    }

    let direction: Direction
    let firstRecordID: String
    let lastRecordID: String

    var identifier: GestureIdentifier { direction.identifier }

    /// Parses the fields of an `#expect` manifest line, which carries a
    /// direction and the first and last record the gesture spans.
    static func parse(
        fields: [Substring]
    ) throws(ParseError) -> Self {
        guard fields.count == 4 else {
            throw .fieldCount(fields.count)
        }
        guard let direction = Direction(rawValue: String(fields[1])) else {
            throw .unknownDirection(String(fields[1]))
        }
        guard !fields[2].isEmpty, !fields[3].isEmpty else {
            throw .emptyRecordID
        }
        return Self(
            direction: direction,
            firstRecordID: String(fields[2]),
            lastRecordID: String(fields[3])
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
        "{\"identifier\":" + jsonString(identifier.rawValue)
            + ",\"direction\":" + jsonString(direction.rawValue)
            + ",\"firstRecordID\":" + jsonString(firstRecordID)
            + ",\"lastRecordID\":" + jsonString(lastRecordID) + "}"
    }
}
