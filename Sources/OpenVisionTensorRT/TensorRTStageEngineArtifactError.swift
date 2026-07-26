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
    case invalidExecutionElementCapacity(Int)
    case duplicateSemanticTensorID(VisionModelTensorID)
    case duplicateEngineTensorName(String)
    case inputOutputNameCollision(String)
    case unknownSemanticTensorID(VisionModelTensorID)
    case missingSemanticTensorID(VisionModelTensorID)
}
