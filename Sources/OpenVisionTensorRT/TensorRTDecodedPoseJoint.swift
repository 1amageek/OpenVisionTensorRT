import CTensorRTShim

public struct TensorRTDecodedPoseJoint: Sendable, Hashable {
    public let normalizedX: Float
    public let normalizedY: Float
    public let confidence: Float

    init(_ joint: OVTRTPoseJoint) {
        normalizedX = joint.normalizedX
        normalizedY = joint.normalizedY
        confidence = joint.confidence
    }

    init(
        normalizedX: Float,
        normalizedY: Float,
        confidence: Float
    ) {
        self.normalizedX = normalizedX
        self.normalizedY = normalizedY
        self.confidence = confidence
    }
}
