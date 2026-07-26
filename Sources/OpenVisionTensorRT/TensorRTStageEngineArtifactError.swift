import OpenVision

public enum TensorRTStageEngineArtifactError:
    Error,
    Sendable,
    Equatable
{
    case emptyStageID
    case missingStage(VisionModelStageID)
    case emptyInputTensorName
    case emptySemanticTensorID
    case emptyEngineTensorName
    case duplicateSemanticTensorID(VisionModelTensorID)
    case duplicateEngineTensorName(String)
    case inputOutputNameCollision(String)
    case unknownSemanticTensorID(VisionModelTensorID)
    case missingSemanticTensorID(VisionModelTensorID)
}
