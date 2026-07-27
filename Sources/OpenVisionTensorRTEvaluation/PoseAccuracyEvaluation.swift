package struct EvaluationGroundTruthJoint: Sendable, Equatable {
    package let name: String
    package let x: Float
    package let y: Float

    package init(name: String, x: Float, y: Float) {
        self.name = name
        self.x = x
        self.y = y
    }
}

package struct EvaluationPredictedJoint: Sendable, Equatable {
    package let name: String
    package let x: Float
    package let y: Float

    package init(name: String, x: Float, y: Float) {
        self.name = name
        self.x = x
        self.y = y
    }
}

package struct EvaluationPredictedPose: Sendable, Equatable {
    package let joints: [EvaluationPredictedJoint]

    package init(joints: [EvaluationPredictedJoint]) {
        self.joints = joints
    }
}

package struct PoseAccuracyEvaluation: Sendable, Equatable {
    package let groundTruthJointCount: Int
    package let matchedJointCount: Int
    package let jointRecall: Double
    package let pckAtFivePercent: Double
    package let pckAtTenPercent: Double
    package let normalizedMeanError: Double
    package let matchedPredictedPoseIndex: Int?

    package var json: String {
        "{\"groundTruthJointCount\":" + String(groundTruthJointCount)
            + ",\"matchedJointCount\":" + String(matchedJointCount)
            + ",\"jointRecall\":" + String(jointRecall)
            + ",\"pckAtFivePercent\":" + String(pckAtFivePercent)
            + ",\"pckAtTenPercent\":" + String(pckAtTenPercent)
            + ",\"normalizedMeanError\":" + String(normalizedMeanError)
            + ",\"matchedPredictedPoseIndex\":"
            + (matchedPredictedPoseIndex.map(String.init) ?? "null")
            + "}"
    }
}

package enum PoseAccuracyEvaluationError: Error, Equatable {
    case emptyGroundTruth
    case duplicateGroundTruthJoint(String)
    case duplicatePredictedJoint(poseIndex: Int, name: String)
    case invalidCoordinate(name: String)
    case degenerateGroundTruthExtent
}

package enum PoseAccuracyEvaluator {
    package static func evaluate(
        groundTruth: [EvaluationGroundTruthJoint],
        predictedPoses: [EvaluationPredictedPose]
    ) throws(PoseAccuracyEvaluationError) -> PoseAccuracyEvaluation {
        guard !groundTruth.isEmpty else {
            throw .emptyGroundTruth
        }
        var truthByName: [String: EvaluationGroundTruthJoint] = [:]
        truthByName.reserveCapacity(groundTruth.count)
        for joint in groundTruth {
            guard joint.x.isFinite, joint.y.isFinite,
                  (0...1).contains(joint.x), (0...1).contains(joint.y) else {
                throw .invalidCoordinate(name: joint.name)
            }
            guard truthByName.updateValue(joint, forKey: joint.name) == nil else {
                throw .duplicateGroundTruthJoint(joint.name)
            }
        }

        var minimumX = groundTruth[0].x
        var maximumX = groundTruth[0].x
        var minimumY = groundTruth[0].y
        var maximumY = groundTruth[0].y
        for joint in groundTruth.dropFirst() {
            minimumX = min(minimumX, joint.x)
            maximumX = max(maximumX, joint.x)
            minimumY = min(minimumY, joint.y)
            maximumY = max(maximumY, joint.y)
        }
        let width = maximumX - minimumX
        let height = maximumY - minimumY
        let normalizationExtent = Double(width * width + height * height)
            .squareRoot()
        guard normalizationExtent.isFinite, normalizationExtent > 0 else {
            throw .degenerateGroundTruthExtent
        }

        var best: Candidate?
        for (poseIndex, pose) in predictedPoses.enumerated() {
            var predictedByName: [String: EvaluationPredictedJoint] = [:]
            predictedByName.reserveCapacity(pose.joints.count)
            for joint in pose.joints {
                guard joint.x.isFinite, joint.y.isFinite else {
                    throw .invalidCoordinate(name: joint.name)
                }
                guard predictedByName.updateValue(joint, forKey: joint.name) == nil else {
                    throw .duplicatePredictedJoint(
                        poseIndex: poseIndex,
                        name: joint.name
                    )
                }
            }
            var distances: [Double] = []
            distances.reserveCapacity(groundTruth.count)
            for truth in groundTruth {
                guard let predicted = predictedByName[truth.name] else {
                    continue
                }
                let deltaX = Double(predicted.x - truth.x)
                let deltaY = Double(predicted.y - truth.y)
                distances.append(
                    (deltaX * deltaX + deltaY * deltaY).squareRoot()
                        / normalizationExtent
                )
            }
            let candidate = Candidate(poseIndex: poseIndex, distances: distances)
            if candidate.isBetter(than: best) {
                best = candidate
            }
        }

        let distances = best?.distances ?? []
        let denominator = Double(groundTruth.count)
        let matchedCount = distances.count
        let missingCount = groundTruth.count - matchedCount
        return PoseAccuracyEvaluation(
            groundTruthJointCount: groundTruth.count,
            matchedJointCount: matchedCount,
            jointRecall: Double(matchedCount) / denominator,
            pckAtFivePercent: Double(distances.filter { $0 <= 0.05 }.count)
                / denominator,
            pckAtTenPercent: Double(distances.filter { $0 <= 0.10 }.count)
                / denominator,
            normalizedMeanError: (
                distances.reduce(0, +) + Double(missingCount)
            ) / denominator,
            matchedPredictedPoseIndex: best?.poseIndex
        )
    }

    private struct Candidate {
        let poseIndex: Int
        let distances: [Double]

        private var meanDistance: Double {
            distances.isEmpty
                ? .infinity
                : distances.reduce(0, +) / Double(distances.count)
        }

        func isBetter(than other: Candidate?) -> Bool {
            guard let other else { return true }
            if distances.count != other.distances.count {
                return distances.count > other.distances.count
            }
            if meanDistance != other.meanDistance {
                return meanDistance < other.meanDistance
            }
            return poseIndex < other.poseIndex
        }
    }
}
