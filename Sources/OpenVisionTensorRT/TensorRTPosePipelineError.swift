public enum TensorRTPosePipelineError:
    Error,
    Sendable,
    Equatable
{
    case invalidConfiguration(String)
    case deferredCleanupPending(
        TensorRTPoseDeferredCleanupResult
    )
    case unavailable(TensorRTPosePipelineReport)
    case creationFailed(
        status: TensorRTRuntimeStatus,
        operation: TensorRTPosePipelineReport,
        cleanup: TensorRTPosePipelineReport
    )
    case failed(
        status: TensorRTRuntimeStatus,
        report: TensorRTPosePipelineReport
    )
    case outputInUse
    case decodePending
    case noDecodePending
    case invalidDetectorOutput
    case invalidPoseOutput
    case inputReleased
    case alreadyShutDown
}
