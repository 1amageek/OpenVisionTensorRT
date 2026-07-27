import ActionRecognition

/// A gesture the recognizer committed to, together with the span of frames it
/// spent reaching that commitment.
///
/// Only `.ended` carries a direction, so only `.ended` is a commitment. The
/// span is kept alongside it because an annotation covers an interval while a
/// commitment is instantaneous, and the two can only be compared through the
/// interval the recognizer was actually tracking.
struct GestureCommitment: Sendable {
    /// Collects commitments as frames are processed.
    ///
    /// The span cannot be recovered from the terminal observation alone — the
    /// observation says what was decided, not when the recognizer started
    /// gathering the evidence — so the first frame of each attempt is recorded
    /// as it goes past.
    struct Collector: Sendable {
        private struct Attempt: Hashable {
            let actorSequence: UInt64
            let correlationSequence: UInt64
        }

        private var firstFrameIndexByAttempt: [Attempt: Int] = [:]
        private var commitments: [GestureCommitment] = []
        private var ambiguousCommitmentCount = 0

        mutating func consume(
            decisions: [RecognitionDecision],
            frameIndex: Int
        ) {
            for decision in decisions {
                switch decision {
                case .recognized(let observation):
                    record(observation, frameIndex: frameIndex, isDecided: true)
                case .ambiguous(_, let candidates):
                    for candidate in candidates {
                        record(candidate, frameIndex: frameIndex, isDecided: false)
                    }
                case .noMatch:
                    continue
                }
            }
        }

        private mutating func record(
            _ observation: RecognitionObservation,
            frameIndex: Int,
            isDecided: Bool
        ) {
            guard case .gesture(let gesture) = observation else { return }
            let attempt = Attempt(
                actorSequence: gesture.actorID.sequence,
                correlationSequence: gesture.correlationID.sequence
            )
            if firstFrameIndexByAttempt[attempt] == nil {
                firstFrameIndexByAttempt[attempt] = frameIndex
            }
            guard let direction = Self.direction(of: gesture) else { return }
            // An ambiguous decision is the session saying it could not choose.
            // Scoring it as a detection would credit a non-answer, so it is
            // counted separately and reported rather than dropped.
            guard isDecided else {
                ambiguousCommitmentCount += 1
                return
            }
            commitments.append(
                GestureCommitment(
                    identifier: gesture.identifier,
                    direction: direction,
                    actorSequence: attempt.actorSequence,
                    correlationSequence: attempt.correlationSequence,
                    firstFrameIndex: firstFrameIndexByAttempt[attempt] ?? frameIndex,
                    commitmentFrameIndex: frameIndex
                )
            )
        }

        private static func direction(
            of gesture: GestureObservation
        ) -> GestureExpectation.Direction? {
            switch gesture.parameters {
            case .horizontalSwipe(let parameters):
                parameters.direction.map(GestureExpectation.Direction.horizontal)
            case .rotaryManipulation(let parameters):
                parameters.direction.map(GestureExpectation.Direction.rotary)
            case .cancel:
                nil
            }
        }

        var result: (
            commitments: [GestureCommitment],
            ambiguousCommitmentCount: Int
        ) {
            (commitments, ambiguousCommitmentCount)
        }
    }

    let identifier: GestureIdentifier
    let direction: GestureExpectation.Direction
    let actorSequence: UInt64
    let correlationSequence: UInt64
    /// Frame the attempt was first reported on.
    let firstFrameIndex: Int
    /// Frame the direction was committed on.
    let commitmentFrameIndex: Int

    /// Overlap of the tracked span with an annotated window, as a fraction of
    /// their union.
    ///
    /// Reported on every match rather than used as a gate by default: the
    /// recognizer commits as soon as it has enough evidence, so its span is
    /// shorter than a hand-annotated segment by construction, and a threshold
    /// invented here would reject correct detections.
    func intersectionOverUnion(
        with expectation: GestureExpectation.Resolved
    ) -> Double {
        let intersectionStart = max(firstFrameIndex, expectation.firstFrameIndex)
        let intersectionEnd = min(
            commitmentFrameIndex,
            expectation.lastFrameIndex
        )
        let intersection = max(0, intersectionEnd - intersectionStart + 1)
        let predicted = commitmentFrameIndex - firstFrameIndex + 1
        let expected = expectation.lastFrameIndex
            - expectation.firstFrameIndex + 1
        let union = predicted + expected - intersection
        guard union > 0 else { return 0 }
        return Double(intersection) / Double(union)
    }

    var json: String {
        "{\"identifier\":" + jsonString(identifier.rawValue)
            + ",\"direction\":" + jsonString(direction.rawValue)
            + ",\"actorSequence\":" + String(actorSequence)
            + ",\"correlationSequence\":" + String(correlationSequence)
            + ",\"firstFrameIndex\":" + String(firstFrameIndex)
            + ",\"commitmentFrameIndex\":" + String(commitmentFrameIndex) + "}"
    }
}
