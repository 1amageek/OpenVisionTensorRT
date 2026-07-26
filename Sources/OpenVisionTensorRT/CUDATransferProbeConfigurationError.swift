public enum CUDATransferProbeConfigurationError:
    Error,
    Sendable,
    Equatable
{
    case invalidByteCount(UInt64)
    case invalidWarmupIterationCount(UInt32)
    case invalidMeasuredIterationCount(UInt32)
}
