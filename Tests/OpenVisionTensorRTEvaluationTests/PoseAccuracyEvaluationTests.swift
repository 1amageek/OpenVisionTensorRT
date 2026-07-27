import Testing
@testable import OpenVisionTensorRTEvaluation

@Suite("Pose accuracy evaluation")
struct PoseAccuracyEvaluationTests {
    @Test("Exact joints produce complete PCK")
    func exactJoints() throws {
        let truth = fixtureTruth
        let result = try PoseAccuracyEvaluator.evaluate(
            groundTruth: truth,
            predictedPoses: [
                EvaluationPredictedPose(
                    joints: truth.map {
                        EvaluationPredictedJoint(name: $0.name, x: $0.x, y: $0.y)
                    }
                )
            ]
        )

        #expect(result.matchedJointCount == 3)
        #expect(result.jointRecall == 1)
        #expect(result.pckAtFivePercent == 1)
        #expect(result.pckAtTenPercent == 1)
        #expect(result.normalizedMeanError == 0)
        #expect(result.matchedPredictedPoseIndex == 0)
    }

    @Test("Missing joints remain accuracy failures")
    func missingJoints() throws {
        let result = try PoseAccuracyEvaluator.evaluate(
            groundTruth: fixtureTruth,
            predictedPoses: [
                EvaluationPredictedPose(
                    joints: [
                        EvaluationPredictedJoint(name: "nose", x: 0.5, y: 0.8)
                    ]
                )
            ]
        )

        #expect(result.matchedJointCount == 1)
        #expect(result.jointRecall == 1.0 / 3.0)
        #expect(result.pckAtFivePercent == 1.0 / 3.0)
        #expect(result.normalizedMeanError == 2.0 / 3.0)
    }

    @Test("Matching chooses the pose with the greatest joint evidence")
    func poseAssignment() throws {
        let result = try PoseAccuracyEvaluator.evaluate(
            groundTruth: fixtureTruth,
            predictedPoses: [
                EvaluationPredictedPose(
                    joints: [
                        EvaluationPredictedJoint(name: "nose", x: 0.5, y: 0.8)
                    ]
                ),
                EvaluationPredictedPose(
                    joints: fixtureTruth.map {
                        EvaluationPredictedJoint(name: $0.name, x: $0.x, y: $0.y)
                    }
                ),
            ]
        )

        #expect(result.matchedPredictedPoseIndex == 1)
    }

    @Test("Ground-truth parser preserves lower-left coordinates")
    func parser() throws {
        let parsed = try EvaluationGroundTruthParser.parse(
            "nose,0.500000000,0.800000000;leftHip,0.400000000,0.200000000"
        )

        #expect(parsed == [
            EvaluationGroundTruthJoint(name: "nose", x: 0.5, y: 0.8),
            EvaluationGroundTruthJoint(name: "leftHip", x: 0.4, y: 0.2),
        ])
    }

    @Test("Ambiguous and invalid ground truth fails explicitly")
    func invalidGroundTruth() throws {
        #expect(throws: EvaluationGroundTruthParserError.self) {
            try EvaluationGroundTruthParser.parse("nose,1.1,0.5")
        }
        #expect(throws: PoseAccuracyEvaluationError.self) {
            try PoseAccuracyEvaluator.evaluate(
                groundTruth: [
                    EvaluationGroundTruthJoint(name: "nose", x: 0.5, y: 0.5),
                    EvaluationGroundTruthJoint(name: "nose", x: 0.6, y: 0.5),
                ],
                predictedPoses: []
            )
        }
    }

    private var fixtureTruth: [EvaluationGroundTruthJoint] {
        [
            EvaluationGroundTruthJoint(name: "nose", x: 0.5, y: 0.8),
            EvaluationGroundTruthJoint(name: "leftHip", x: 0.4, y: 0.2),
            EvaluationGroundTruthJoint(name: "rightHip", x: 0.6, y: 0.2),
        ]
    }
}
