import OpenVision

public enum RG10PreprocessorError:
    Error,
    Sendable,
    Equatable
{
    case unavailable(RG10PreprocessingReport)
    case invalidInput(VisionError)
    case incompatibleInput(RG10InputMismatch)
    case failed(
        status: TensorRTRuntimeStatus,
        report: RG10PreprocessingReport
    )
    case creationFailed(
        status: TensorRTRuntimeStatus,
        operation: RG10PreprocessingReport,
        cleanup: RG10PreprocessingReport
    )
    case deferredCleanupPending(RG10DeferredCleanupResult)
    case outputInUse
    case invalidTensorDescriptor
    case alreadyShutDown
}
