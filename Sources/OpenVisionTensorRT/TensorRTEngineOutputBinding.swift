import OpenVision

public struct TensorRTEngineOutputBinding: Sendable, Hashable {
    public let semanticTensorID: VisionModelTensorID
    public let engineTensorName: String
    /// Persistent TensorRT output capacity when its allocation callback can
    /// request an upper bound larger than the final semantic tensor shape.
    ///
    /// This is a total element count, not a byte count or one-axis dimension.
    /// The engine still validates the final shape against the semantic model.
    public let executionElementCapacity: Int?

    public init(
        semanticTensorID: VisionModelTensorID,
        engineTensorName: String,
        executionElementCapacity: Int? = nil
    ) throws(TensorRTStageEngineArtifactError) {
        guard !semanticTensorID.rawValue.isEmpty else {
            throw .emptySemanticTensorID
        }
        guard !engineTensorName.isEmpty else {
            throw .emptyEngineTensorName
        }
        if let executionElementCapacity {
            guard executionElementCapacity > 0 else {
                throw .invalidExecutionElementCapacity(
                    executionElementCapacity
                )
            }
        }
        self.semanticTensorID = semanticTensorID
        self.engineTensorName = engineTensorName
        self.executionElementCapacity =
            executionElementCapacity
    }
}
