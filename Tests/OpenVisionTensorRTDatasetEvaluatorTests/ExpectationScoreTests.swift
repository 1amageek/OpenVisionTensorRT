import ActionRecognition
import Testing

@testable import OpenVisionTensorRTDatasetEvaluator

/// Checks what the evaluator is willing to call a pass.
///
/// The evaluator previously reported `passed` whenever a gesture reached
/// `.ended`, which credits the pipeline for running rather than for being
/// right — a run that recognized every motion backwards passed. Each case below
/// pins one way of being wrong to its own outcome, so a failure names the
/// defect instead of just denying the pass.
@Suite("Expectation scoring")
struct ExpectationScoreTests {
    // MARK: - Manifest parsing

    /// The gesture is implied by the direction, so a manifest has no way to
    /// state a swipe and a rotary direction in the same breath.
    @Test("A direction names its own gesture")
    func directionImpliesIdentifier() throws {
        let swipe = try GestureExpectation.parse(
            fields: fields("#expect", "left", "a", "b")
        )
        #expect(swipe.identifier == .horizontalSwipe)
        #expect(swipe.direction == .horizontal(.left))

        let rotary = try GestureExpectation.parse(
            fields: fields("#expect", "clockwise", "a", "b")
        )
        #expect(rotary.identifier == .rotaryManipulation)
        #expect(rotary.direction == .rotary(.clockwise))
    }

    @Test("An unreadable direction is rejected, not guessed")
    func unknownDirectionIsRejected() {
        #expect(throws: GestureExpectation.ParseError.unknownDirection("up")) {
            try GestureExpectation.parse(
                fields: fields("#expect", "up", "a", "b")
            )
        }
    }

    @Test("A malformed line is rejected by field count")
    func malformedLineIsRejected() {
        #expect(throws: GestureExpectation.ParseError.fieldCount(3)) {
            try GestureExpectation.parse(fields: fields("#expect", "left", "a"))
        }
    }

    /// Resolving against a record the manifest does not contain fails loudly.
    /// Dropping the expectation instead would shrink what the run is scored
    /// against while still reporting a pass.
    @Test("An expectation naming an absent record fails resolution")
    func absentRecordFailsResolution() throws {
        let parsed = try GestureExpectation.parse(
            fields: fields("#expect", "left", "a", "missing")
        )
        #expect(
            throws: GestureExpectation.ResolutionError.unknownRecordID("missing")
        ) {
            try parsed.resolved(frameIndexByRecordID: ["a": 0])
        }
    }

    @Test("A backwards window fails resolution")
    func invertedWindowFailsResolution() throws {
        let parsed = try GestureExpectation.parse(
            fields: fields("#expect", "left", "late", "early")
        )
        #expect(
            throws: GestureExpectation.ResolutionError.invertedWindow(
                first: "late",
                last: "early"
            )
        ) {
            try parsed.resolved(
                frameIndexByRecordID: ["early": 2, "late": 9]
            )
        }
    }

    // MARK: - Outcomes

    @Test("The annotated gesture, recognized as annotated, passes")
    func correctDetectionPasses() throws {
        let score = ExpectationScore(
            expectations: [try window("right", 0, 20)],
            commitments: [commitment(.horizontal(.right), first: 4, ended: 9)],
            ambiguousCommitmentCount: 0,
            minimumIntersectionOverUnion: nil
        )
        #expect(score.count(of: .truePositive) == 1)
        #expect(score.falsePositives.isEmpty)
        #expect(score.isPassing)
        #expect(score.status == "passed")
    }

    /// The recorded G05 and G06 failure: the right gesture, read backwards.
    /// This has to be its own outcome — the arbitration picked the correct
    /// recognizer and only the sign is wrong, which is a different repair from
    /// picking the wrong recognizer.
    @Test("A reversed direction is a direction error, not a pass")
    func reversedDirectionFails() throws {
        let score = ExpectationScore(
            expectations: [try window("right", 0, 20)],
            commitments: [commitment(.horizontal(.left), first: 4, ended: 9)],
            ambiguousCommitmentCount: 0,
            minimumIntersectionOverUnion: nil
        )
        #expect(score.count(of: .directionError) == 1)
        #expect(score.count(of: .truePositive) == 0)
        #expect(!score.isPassing)
        #expect(score.status == "failedExpectations")
    }

    /// The recorded B0B failure: a rotary claimed a motion the dataset
    /// annotates as a swipe.
    @Test("The wrong gesture in the right window is a label error")
    func wrongGestureFails() throws {
        let score = ExpectationScore(
            expectations: [try window("right", 0, 20)],
            commitments: [commitment(.rotary(.clockwise), first: 4, ended: 9)],
            ambiguousCommitmentCount: 0,
            minimumIntersectionOverUnion: nil
        )
        #expect(score.count(of: .labelError) == 1)
        #expect(!score.isPassing)
    }

    @Test("A commitment outside every window is a false positive")
    func commitmentOutsideWindowFails() throws {
        let score = ExpectationScore(
            expectations: [try window("right", 0, 20)],
            commitments: [commitment(.horizontal(.right), first: 40, ended: 45)],
            ambiguousCommitmentCount: 0,
            minimumIntersectionOverUnion: nil
        )
        #expect(score.falsePositives.count == 1)
        #expect(score.count(of: .falseNegative) == 1)
        #expect(!score.isPassing)
    }

    @Test("An unmet expectation is a false negative")
    func missedGestureFails() throws {
        let score = ExpectationScore(
            expectations: [try window("right", 0, 20)],
            commitments: [],
            ambiguousCommitmentCount: 0,
            minimumIntersectionOverUnion: nil
        )
        #expect(score.count(of: .falseNegative) == 1)
        #expect(score.matches.first?.commitment == nil)
        #expect(!score.isPassing)
    }

    /// The defect this scoring exists to remove: a run that recognized
    /// something, against a manifest that never said what was performed, is
    /// unscored. Calling it a pass would let an unannotated clip certify the
    /// recognizer.
    @Test("A run with nothing to check against does not pass")
    func unscoredRunDoesNotPass() {
        let score = ExpectationScore(
            expectations: [],
            commitments: [commitment(.horizontal(.right), first: 4, ended: 9)],
            ambiguousCommitmentCount: 0,
            minimumIntersectionOverUnion: nil
        )
        #expect(score.status == "completedNoExpectation")
        #expect(!score.isPassing)
        // The commitment is still reported, so an unscored run cannot hide
        // what the recognizer did.
        #expect(score.falsePositives.count == 1)
    }

    /// One motion, one answer. A recognizer that fires twice inside a single
    /// annotated gesture is producing a spurious commitment, and counting the
    /// second one as another hit would reward the flicker.
    @Test("A second commitment in one window cannot be a second hit")
    func repeatedCommitmentInWindowIsFalsePositive() throws {
        let score = ExpectationScore(
            expectations: [try window("right", 0, 20)],
            commitments: [
                commitment(.horizontal(.right), first: 2, ended: 6),
                commitment(.horizontal(.right), first: 8, ended: 12)
            ],
            ambiguousCommitmentCount: 0,
            minimumIntersectionOverUnion: nil
        )
        #expect(score.count(of: .truePositive) == 1)
        #expect(score.falsePositives.count == 1)
        #expect(!score.isPassing)
    }

    /// Commitments are claimed in the order they were made, whatever order the
    /// collector happened to accumulate them in, so a run with two annotated
    /// gestures scores the same either way.
    @Test("Claiming follows commitment order, not array order")
    func claimingFollowsFrameOrder() throws {
        let score = ExpectationScore(
            expectations: [
                try window("right", 0, 9),
                try window("left", 10, 19)
            ],
            commitments: [
                commitment(.horizontal(.left), first: 11, ended: 15),
                commitment(.horizontal(.right), first: 1, ended: 5)
            ],
            ambiguousCommitmentCount: 0,
            minimumIntersectionOverUnion: nil
        )
        #expect(score.count(of: .truePositive) == 2)
        #expect(score.isPassing)
    }

    // MARK: - Overlap

    /// Overlap is always measured and reported. It gates only when the manifest
    /// asks it to, because the recognizer commits as soon as it has enough
    /// evidence and its span is shorter than a hand-annotated segment by
    /// construction — a threshold invented here would reject correct answers.
    @Test("Overlap is measured but does not gate unless declared")
    func overlapIsMeasuredNotAssumed() throws {
        let expectation = try window("right", 0, 19)
        let short = commitment(.horizontal(.right), first: 10, ended: 12)
        // 3 overlapping frames over a union of 20.
        #expect(short.intersectionOverUnion(with: expectation) == 0.15)

        let ungated = ExpectationScore(
            expectations: [expectation],
            commitments: [short],
            ambiguousCommitmentCount: 0,
            minimumIntersectionOverUnion: nil
        )
        #expect(ungated.count(of: .truePositive) == 1)
        #expect(ungated.matches.first?.intersectionOverUnion == 0.15)
        #expect(ungated.isPassing)

        let gated = ExpectationScore(
            expectations: [expectation],
            commitments: [short],
            ambiguousCommitmentCount: 0,
            minimumIntersectionOverUnion: 0.5
        )
        #expect(gated.count(of: .temporalError) == 1)
        #expect(!gated.isPassing)
    }

    @Test("A commitment sharing no frame with its window overlaps by zero")
    func disjointSpansOverlapByZero() throws {
        let expectation = try window("right", 10, 20)
        let before = commitment(.horizontal(.right), first: 0, ended: 4)
        #expect(before.intersectionOverUnion(with: expectation) == 0)
    }

    // MARK: - Reporting

    /// A decision the session declined to make is neither a hit nor a miss, but
    /// it cannot vanish: a run full of ambiguity that scores clean is still a
    /// run the recognizer could not read.
    @Test("Ambiguous decisions are carried into the report")
    func ambiguityIsReported() throws {
        let score = ExpectationScore(
            expectations: [try window("right", 0, 20)],
            commitments: [commitment(.horizontal(.right), first: 4, ended: 9)],
            ambiguousCommitmentCount: 3,
            minimumIntersectionOverUnion: nil
        )
        #expect(score.ambiguousCommitmentCount == 3)
        #expect(score.json.contains("\"ambiguousCommitmentCount\":3"))
        #expect(score.json.contains("\"status\":\"passed\""))
    }

    // MARK: - Construction

    private func fields(_ values: String...) -> [Substring] {
        values.map { $0[...] }
    }

    private func window(
        _ direction: String,
        _ first: Int,
        _ last: Int
    ) throws -> GestureExpectation.Resolved {
        let firstID = "record-\(first)"
        let lastID = "record-\(last)"
        let parsed = try GestureExpectation.parse(
            fields: fields("#expect", direction, firstID, lastID)
        )
        var frameIndexByRecordID: [String: Int] = [:]
        frameIndexByRecordID[firstID] = first
        frameIndexByRecordID[lastID] = last
        return try parsed.resolved(frameIndexByRecordID: frameIndexByRecordID)
    }

    private func commitment(
        _ direction: GestureExpectation.Direction,
        first: Int,
        ended: Int
    ) -> GestureCommitment {
        GestureCommitment(
            identifier: direction.identifier,
            direction: direction,
            actorSequence: 1,
            correlationSequence: UInt64(first),
            firstFrameIndex: first,
            commitmentFrameIndex: ended
        )
    }
}
