public enum TensorRTEngineError:
    Error,
    Sendable,
    Equatable
{
    case unavailable(TensorRTEngineLoadReport)
    case loadingFailed(
        status: TensorRTRuntimeStatus,
        report: TensorRTEngineLoadReport
    )
    case inspectionFailed(
        status: TensorRTRuntimeStatus,
        tensorIndex: Int
    )
    case unsupportedTensorElementType(tensorIndex: Int)
    case incompatibleArtifact(TensorRTEngineCompatibilityError)
    case alreadyShutDown
}
