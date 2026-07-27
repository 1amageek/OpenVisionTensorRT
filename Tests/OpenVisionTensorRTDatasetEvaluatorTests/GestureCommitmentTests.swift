import ActionRecognition
import OpenVision
import Testing

@testable import OpenVisionTensorRTDatasetEvaluator

/// Checks what the evaluator counts as an answer.
///
/// A gesture is only an answer once it names a direction, which the recognizers
/// do exactly once, on `.ended`. Everything before that is the recognizer
/// gathering evidence, and an ambiguous decision is it saying it could not
/// choose. Scoring either as a detection would credit work in progress and a
/// non-answer respectively.
@Suite("Commitment collection")
struct GestureCommitmentTests {
    // MARK: - Through the recognizer

    /// The whole path the evaluator uses: a wrist swept to the right through a
    /// real `RecognitionSession`, its decisions collected frame by frame, and
    /// the result scored against what the sweep was.
    @Test("A swept wrist scores against the direction it was swept")
    func sweptWristScoresAgainstItsDirection() async throws {
        let collected = try await collect(
            xs: [0.30, 0.40, 0.50, 0.505, 0.50, 0.505, 0.50, 0.50, 0.50, 0.50]
        )
        let commitment = try #require(collected.commitments.first)
        #expect(collected.commitments.count == 1)
        #expect(commitment.identifier == .horizontalSwipe)
        #expect(commitment.direction == .horizontal(.right))
        // The span is the frames the recognizer spent on the motion, so it can
        // be compared against an annotated interval at all.
        #expect(commitment.firstFrameIndex < commitment.commitmentFrameIndex)

        let matching = ExpectationScore(
            expectations: [try window("right", 0, 9)],
            commitments: collected.commitments,
            ambiguousCommitmentCount: collected.ambiguousCommitmentCount,
            minimumIntersectionOverUnion: nil
        )
        #expect(matching.isPassing)

        // The same run against the opposite annotation. If the evaluator still
        // passed here it would not be reading the direction at all.
        let opposed = ExpectationScore(
            expectations: [try window("left", 0, 9)],
            commitments: collected.commitments,
            ambiguousCommitmentCount: collected.ambiguousCommitmentCount,
            minimumIntersectionOverUnion: nil
        )
        #expect(opposed.count(of: .directionError) == 1)
        #expect(!opposed.isPassing)
    }

    /// A hand that never travels far enough produces no commitment, so the
    /// evaluator has nothing to score and says so rather than passing.
    @Test("A hand that does not travel commits to nothing")
    func stillHandCommitsToNothing() async throws {
        let collected = try await collect(
            xs: [0.50, 0.502, 0.50, 0.501, 0.50, 0.502, 0.50, 0.501, 0.50, 0.50]
        )
        #expect(collected.commitments.isEmpty)

        let score = ExpectationScore(
            expectations: [try window("right", 0, 9)],
            commitments: collected.commitments,
            ambiguousCommitmentCount: collected.ambiguousCommitmentCount,
            minimumIntersectionOverUnion: nil
        )
        #expect(score.count(of: .falseNegative) == 1)
        #expect(!score.isPassing)
    }

    // MARK: - Collector

    /// The span comes from the frames the attempt was reported on, which the
    /// terminal observation cannot supply: it says what was decided, not when
    /// the recognizer started looking.
    @Test("The span runs from the first frame of the attempt")
    func spanStartsAtFirstReportedFrame() throws {
        var collector = GestureCommitment.Collector()
        collector.consume(
            decisions: [.recognized(try gesture(phase: .began, direction: nil))],
            frameIndex: 3
        )
        collector.consume(
            decisions: [
                .recognized(try gesture(phase: .changed, direction: nil))
            ],
            frameIndex: 4
        )
        collector.consume(
            decisions: [
                .recognized(try gesture(phase: .ended, direction: .right))
            ],
            frameIndex: 6
        )
        let result = collector.result
        let commitment = try #require(result.commitments.first)
        #expect(result.commitments.count == 1)
        #expect(commitment.firstFrameIndex == 3)
        #expect(commitment.commitmentFrameIndex == 6)
    }

    /// An ambiguous decision is the session reporting that it could not choose.
    /// Counting it as a detection would let a run pass on answers nobody gave.
    @Test("An ambiguous candidate is counted, never committed")
    func ambiguousCandidateIsNotACommitment() throws {
        var collector = GestureCommitment.Collector()
        collector.consume(
            decisions: [
                .ambiguous(
                    actorID: actorID,
                    candidates: [try gesture(phase: .ended, direction: .right)]
                )
            ],
            frameIndex: 5
        )
        let result = collector.result
        #expect(result.commitments.isEmpty)
        #expect(result.ambiguousCommitmentCount == 1)
    }

    /// A withdrawn gesture publishes no direction, so there is nothing to
    /// score — the same rule that keeps a consumer from acting on it.
    @Test("A cancelled gesture is not a commitment")
    func cancelledGestureIsNotACommitment() throws {
        var collector = GestureCommitment.Collector()
        collector.consume(
            decisions: [
                .recognized(try gesture(phase: .began, direction: nil)),
                .recognized(
                    .gesture(
                        try GestureObservation(
                            actorID: actorID,
                            identifier: .cancel,
                            phase: .cancelled,
                            confidence: 0.9,
                            parameters: .cancel(reason: .explicitPose),
                            startTimestamp: try timestamp(frame: 1),
                            latestTimestamp: try timestamp(frame: 2),
                            correlationID: correlationID,
                            recognizerRevision: 1
                        )
                    )
                )
            ],
            frameIndex: 2
        )
        #expect(collector.result.commitments.isEmpty)
        #expect(collector.result.ambiguousCommitmentCount == 0)
    }

    @Test("A frame with no match contributes nothing")
    func noMatchContributesNothing() {
        var collector = GestureCommitment.Collector()
        collector.consume(decisions: [.noMatch(actorID: actorID)], frameIndex: 0)
        #expect(collector.result.commitments.isEmpty)
        #expect(collector.result.ambiguousCommitmentCount == 0)
    }

    // MARK: - Construction

    private let sessionID = RecognitionSessionID(high: 7, low: 11)

    private var actorID: ActorID {
        ActorID(sessionID: sessionID, sequence: 1)
    }

    private var correlationID: GestureCorrelationID {
        GestureCorrelationID(sessionID: sessionID, sequence: 1)
    }

    private func window(
        _ direction: String,
        _ first: Int,
        _ last: Int
    ) throws -> GestureExpectation.Resolved {
        let firstID = "record-\(first)"
        let lastID = "record-\(last)"
        let parsed = try GestureExpectation.parse(
            fields: ["#expect"[...], direction[...], firstID[...], lastID[...]]
        )
        var frameIndexByRecordID: [String: Int] = [:]
        frameIndexByRecordID[firstID] = first
        frameIndexByRecordID[lastID] = last
        return try parsed.resolved(frameIndexByRecordID: frameIndexByRecordID)
    }

    private func gesture(
        phase: GesturePhase,
        direction: HorizontalDirection?
    ) throws -> RecognitionObservation {
        .gesture(
            try GestureObservation(
                actorID: actorID,
                identifier: .horizontalSwipe,
                phase: phase,
                confidence: 0.9,
                parameters: .horizontalSwipe(
                    HorizontalSwipeParameters(
                        direction: direction,
                        displacement: 0.2,
                        velocity: 0.6
                    )
                ),
                startTimestamp: try timestamp(frame: 1),
                latestTimestamp: try timestamp(frame: 2),
                correlationID: correlationID,
                recognizerRevision: 1
            )
        )
    }

    // MARK: - Drive

    /// Feeds a wrist travelling along `xs` through a real session and collects
    /// its decisions the way the dataset evaluator does.
    private func collect(
        xs: [Float]
    ) async throws -> (
        commitments: [GestureCommitment],
        ambiguousCommitmentCount: Int
    ) {
        let session = RecognitionSession(
            id: sessionID,
            budget: try .lumeProofV1()
        )
        var collector = GestureCommitment.Collector()
        for (index, x) in xs.enumerated() {
            let update = try makeUpdate(frame: index + 1, x: x)
            let recognition = try await session.process(update)
            collector.consume(
                decisions: recognition.decisions,
                frameIndex: index
            )
        }
        await session.shutdown()
        return collector.result
    }

    private func makeUpdate(
        frame: Int,
        x: Float
    ) throws -> HumanBodyPoseTrackingUpdate {
        let stamp = try timestamp(frame: frame)
        let frameID = VisionFrameID(
            source: "commitment",
            sequence: UInt64(frame)
        )
        let provenance = VisionObservationProvenance(
            frameID: frameID,
            timestamp: stamp,
            coordinateSpace: .normalizedImage(source: "commitment"),
            calibration: nil,
            transformRevision: nil
        )
        let hand = try HumanHandPoseObservation(
            id: VisionObservationID(high: UInt64(frame), low: 90),
            confidence: 0.95,
            timeRange: nil,
            originatingRequestDescriptor: nil,
            chirality: .right,
            joints: [
                .wrist: try Joint(
                    location: NormalizedPoint(x: x, y: 0.5),
                    jointName: HumanHandPoseObservation.JointName.wrist.rawValue,
                    confidence: 0.95
                ),
                .middleMCP: try Joint(
                    location: NormalizedPoint(x: x + 0.08, y: 0.5),
                    jointName: HumanHandPoseObservation.JointName.middleMCP
                        .rawValue,
                    confidence: 0.95
                ),
            ],
            provenance: provenance
        )
        let pose = try HumanBodyPoseObservation(
            id: VisionObservationID(high: UInt64(frame), low: 1),
            confidence: 0.95,
            timeRange: nil,
            originatingRequestDescriptor: nil,
            joints: [
                .rightWrist: try Joint(
                    location: NormalizedPoint(x: x, y: 0.5),
                    jointName: HumanBodyPoseObservation.JointName.rightWrist
                        .rawValue,
                    confidence: 0.95
                )
            ],
            rightHand: hand,
            provenance: provenance
        )
        let reference = try VisionTrackReference(
            id: VisionTrackID(
                sessionID: VisionTrackingSessionID(high: 1, low: 2),
                epoch: 1,
                sequence: 1
            ),
            source: "commitment",
            state: frame == 1 ? .new : .continued,
            firstObservationTimestamp: try timestamp(frame: 1),
            latestObservationTimestamp: stamp,
            confidence: 0.95,
            observationCount: UInt64(frame),
            missedAnalysisCountBeforeObservation: 0,
            predecessorTrackID: nil
        )
        return HumanBodyPoseTrackingUpdate(
            frameID: frameID,
            timestamp: stamp,
            wasAnalyzed: true,
            observations: [
                TrackedHumanBodyPoseObservation(pose: pose, track: reference)
            ],
            endedTrackIDs: []
        )
    }

    private func timestamp(frame: Int) throws -> VisionTimestamp {
        try VisionTimestamp(
            time: CMTime(value: Int64(frame), timescale: 30),
            clockDomain: VisionClockDomain(
                id: "commitment",
                epoch: 1,
                kind: .deviceMonotonic
            )
        )
    }
}
