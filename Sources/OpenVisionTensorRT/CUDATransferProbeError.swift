public enum CUDATransferProbeError:
    Error,
    Sendable,
    Equatable
{
    case unavailable(CUDATransferProbeResult)
    case failed(
        status: TensorRTRuntimeStatus,
        result: CUDATransferProbeResult
    )
}
