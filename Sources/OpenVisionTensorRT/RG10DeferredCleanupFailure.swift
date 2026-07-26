public struct RG10DeferredCleanupFailure:
    Sendable,
    Equatable
{
    public let status: TensorRTRuntimeStatus
    public let report: RG10PreprocessingReport

    init(
        status: TensorRTRuntimeStatus,
        report: RG10PreprocessingReport
    ) {
        self.status = status
        self.report = report
    }
}
