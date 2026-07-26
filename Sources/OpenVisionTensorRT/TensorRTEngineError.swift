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
    case executionAlreadyPrepared
    case executionNotPrepared
    case outputInUse
    case invalidBatchSize(Int)
    case invalidOutputCount(Int)
    case invalidOutputCapacity(String)
    case invalidInput(TensorRTDeviceInputError)
    case executionPreparationFailed(
        status: TensorRTRuntimeStatus,
        report: TensorRTEngineExecutionReport
    )
    case executionFailed(
        status: TensorRTRuntimeStatus,
        report: TensorRTEngineExecutionReport
    )
    case outputInspectionFailed(
        status: TensorRTRuntimeStatus,
        outputIndex: Int
    )
    case executionCleanupFailed(
        status: TensorRTRuntimeStatus,
        report: TensorRTEngineExecutionReport
    )
    case alreadyShutDown
}
