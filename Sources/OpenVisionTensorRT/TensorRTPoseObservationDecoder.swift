import OpenCoreMedia
import OpenVision

struct TensorRTPoseObservationDecoder: Sendable {
    let minimumConfidence: Float

    func observations(
        batch: TensorRTDecodedPoseBatch,
        regions: [TensorRTPoseRegion],
        timing: CMSampleTimingInfo,
        provenance: VisionObservationProvenance,
        request: DetectHumanBodyPoseRequest,
        executionID: VisionExecutionID
    ) throws(VisionError) -> [HumanBodyPoseObservation] {
        guard
            batch.regionCount == regions.count,
            batch.jointCount == 133
        else {
            throw OpenVisionTensorRTProvider.backendFailure(
                operation: "decodeShape",
                code: 40
            )
        }
        var observations: [HumanBodyPoseObservation] = []
        observations.reserveCapacity(batch.regionCount)
        let timeRange = CMTimeRange(
            start: timing.presentationTimeStamp,
            duration: timing.duration
        )
        for regionIndex in 0 ..< batch.regionCount {
            let decoded = batch.joints(forRegion: regionIndex)
            let body = try bodyJoints(decoded)
            guard !body.isEmpty else {
                continue
            }
            let leftHand = request.detectsHands
                ? try handObservation(
                    chirality: .left,
                    firstJointIndex: 91,
                    decoded: decoded,
                    regionIndex: regionIndex,
                    executionID: executionID,
                    timeRange: timeRange,
                    request: request,
                    provenance: provenance
                )
                : nil
            let rightHand = request.detectsHands
                ? try handObservation(
                    chirality: .right,
                    firstJointIndex: 112,
                    decoded: decoded,
                    regionIndex: regionIndex,
                    executionID: executionID,
                    timeRange: timeRange,
                    request: request,
                    provenance: provenance
                )
                : nil
            let poseConfidence = averageConfidence(body.values)
            let confidence = min(
                regions[regionIndex].confidence,
                poseConfidence
            )
            observations.append(
                try HumanBodyPoseObservation(
                    id: observationID(
                        executionID: executionID,
                        regionIndex: regionIndex,
                        kind: 0
                    ),
                    confidence: confidence,
                    timeRange: timeRange,
                    originatingRequestDescriptor:
                        request.descriptor,
                    joints: body,
                    leftHand: leftHand,
                    rightHand: rightHand,
                    provenance: provenance
                )
            )
        }
        return observations
    }

    private func bodyJoints(
        _ decoded: ArraySlice<TensorRTDecodedPoseJoint>
    ) throws(VisionError)
        -> [HumanBodyPoseObservation.JointName: Joint]
    {
        let mappings: [
            (HumanBodyPoseObservation.JointName, Int)
        ] = [
            (.nose, 0),
            (.leftEye, 1),
            (.rightEye, 2),
            (.leftEar, 3),
            (.rightEar, 4),
            (.leftShoulder, 5),
            (.rightShoulder, 6),
            (.leftElbow, 7),
            (.rightElbow, 8),
            (.leftWrist, 9),
            (.rightWrist, 10),
            (.leftHip, 11),
            (.rightHip, 12),
            (.leftKnee, 13),
            (.rightKnee, 14),
            (.leftAnkle, 15),
            (.rightAnkle, 16)
        ]
        var joints:
            [HumanBodyPoseObservation.JointName: Joint] = [:]
        joints.reserveCapacity(mappings.count + 2)
        for (name, index) in mappings {
            guard let joint = try joint(
                name: name.rawValue,
                decoded: decoded,
                index: index
            ) else {
                continue
            }
            joints[name] = joint
        }
        if let leftShoulder = joints[.leftShoulder],
           let rightShoulder = joints[.rightShoulder] {
            joints[.neck] = try midpoint(
                name:
                    HumanBodyPoseObservation.JointName
                    .neck.rawValue,
                first: leftShoulder,
                second: rightShoulder
            )
        }
        if let leftHip = joints[.leftHip],
           let rightHip = joints[.rightHip] {
            joints[.root] = try midpoint(
                name:
                    HumanBodyPoseObservation.JointName
                    .root.rawValue,
                first: leftHip,
                second: rightHip
            )
        }
        return joints
    }

    private func handObservation(
        chirality: HumanHandPoseObservation.Chirality,
        firstJointIndex: Int,
        decoded: ArraySlice<TensorRTDecodedPoseJoint>,
        regionIndex: Int,
        executionID: VisionExecutionID,
        timeRange: CMTimeRange,
        request: DetectHumanBodyPoseRequest,
        provenance: VisionObservationProvenance
    ) throws(VisionError) -> HumanHandPoseObservation? {
        let names: [HumanHandPoseObservation.JointName] = [
            .wrist,
            .thumbCMC,
            .thumbMP,
            .thumbIP,
            .thumbTip,
            .indexMCP,
            .indexPIP,
            .indexDIP,
            .indexTip,
            .middleMCP,
            .middlePIP,
            .middleDIP,
            .middleTip,
            .ringMCP,
            .ringPIP,
            .ringDIP,
            .ringTip,
            .littleMCP,
            .littlePIP,
            .littleDIP,
            .littleTip
        ]
        var joints:
            [HumanHandPoseObservation.JointName: Joint] = [:]
        joints.reserveCapacity(names.count)
        for (offset, name) in names.enumerated() {
            guard let joint = try joint(
                name: name.rawValue,
                decoded: decoded,
                index: firstJointIndex + offset
            ) else {
                continue
            }
            joints[name] = joint
        }
        guard !joints.isEmpty else {
            return nil
        }
        return try HumanHandPoseObservation(
            id: observationID(
                executionID: executionID,
                regionIndex: regionIndex,
                kind: chirality == .left ? 1 : 2
            ),
            confidence: averageConfidence(joints.values),
            timeRange: timeRange,
            originatingRequestDescriptor: request.descriptor,
            chirality: chirality,
            joints: joints,
            provenance: provenance
        )
    }

    private func joint(
        name: String,
        decoded: ArraySlice<TensorRTDecodedPoseJoint>,
        index: Int
    ) throws(VisionError) -> Joint? {
        let absoluteIndex = decoded.startIndex + index
        guard decoded.indices.contains(absoluteIndex) else {
            throw OpenVisionTensorRTProvider.backendFailure(
                operation: "jointIndex",
                code: 41
            )
        }
        let value = decoded[absoluteIndex]
        guard value.confidence >= minimumConfidence else {
            return nil
        }
        let location: NormalizedPoint
        do {
            location = try NormalizedPoint(
                x: value.normalizedX,
                y: value.normalizedY
            )
        } catch let error {
            throw .invalidGeometry(error)
        }
        return try Joint(
            location: location,
            jointName: name,
            confidence: value.confidence
        )
    }

    private func midpoint(
        name: String,
        first: Joint,
        second: Joint
    ) throws(VisionError) -> Joint {
        let location: NormalizedPoint
        do {
            location = try NormalizedPoint(
                x: (first.location.x + second.location.x) * 0.5,
                y: (first.location.y + second.location.y) * 0.5
            )
        } catch let error {
            throw .invalidGeometry(error)
        }
        return try Joint(
            location: location,
            jointName: name,
            confidence: min(first.confidence, second.confidence)
        )
    }

    private func averageConfidence<Values: Collection>(
        _ joints: Values
    ) -> Float where Values.Element == Joint {
        var sum: Float = 0
        var count: Float = 0
        for joint in joints {
            sum += joint.confidence
            count += 1
        }
        return count > 0 ? sum / count : 0
    }

    private func observationID(
        executionID: VisionExecutionID,
        regionIndex: Int,
        kind: UInt64
    ) -> VisionObservationID {
        VisionObservationID(
            high: executionID.sequence,
            low: UInt64(regionIndex) * 4 + kind
        )
    }
}
