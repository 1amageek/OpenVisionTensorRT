import OpenVision

public struct TensorRTEngineOutputBinding: Sendable, Hashable {
    public let semanticTensorID: VisionModelTensorID
    public let engineTensorName: String

    public init(
        semanticTensorID: VisionModelTensorID,
        engineTensorName: String
    ) throws(TensorRTStageEngineArtifactError) {
        guard !semanticTensorID.rawValue.isEmpty else {
            throw .emptySemanticTensorID
        }
        guard !engineTensorName.isEmpty else {
            throw .emptyEngineTensorName
        }
        self.semanticTensorID = semanticTensorID
        self.engineTensorName = engineTensorName
    }
}
