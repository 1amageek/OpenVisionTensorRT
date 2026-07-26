public struct TensorRTPoseDeferredCleanupFailure:
    Sendable,
    Equatable
{
    public let status: TensorRTRuntimeStatus
    public let report: TensorRTPosePipelineReport

    init(
        status: TensorRTRuntimeStatus,
        report: TensorRTPosePipelineReport
    ) {
        self.status = status
        self.report = report
    }
}
