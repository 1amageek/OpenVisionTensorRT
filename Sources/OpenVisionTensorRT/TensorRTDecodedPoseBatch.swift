public struct TensorRTDecodedPoseBatch: Sendable {
    public let joints: [TensorRTDecodedPoseJoint]
    public let regionCount: Int
    public let jointCount: Int
    public let report: TensorRTPosePipelineReport

    public func joints(forRegion index: Int)
        -> ArraySlice<TensorRTDecodedPoseJoint>
    {
        guard (0 ..< regionCount).contains(index) else {
            return []
        }
        let lowerBound = index * jointCount
        return joints[
            lowerBound ..< lowerBound + jointCount
        ]
    }
}
