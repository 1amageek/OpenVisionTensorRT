public struct CUDATransferProbeConfiguration:
    Sendable,
    Hashable
{
    public static let maximumByteCount: UInt64 =
        512 * 1024 * 1024
    public static let maximumIterationCount: UInt32 = 10_000

    public let byteCount: UInt64
    public let warmupIterationCount: UInt32
    public let measuredIterationCount: UInt32

    public init(
        byteCount: UInt64,
        warmupIterationCount: UInt32,
        measuredIterationCount: UInt32
    ) throws(CUDATransferProbeConfigurationError) {
        guard byteCount > 0 else {
            throw .invalidByteCount(byteCount)
        }
        guard byteCount <= Self.maximumByteCount else {
            throw .invalidByteCount(byteCount)
        }
        guard
            warmupIterationCount <= Self.maximumIterationCount
        else {
            throw .invalidWarmupIterationCount(
                warmupIterationCount
            )
        }
        guard
            measuredIterationCount > 0,
            measuredIterationCount <= Self.maximumIterationCount
        else {
            throw .invalidMeasuredIterationCount(
                measuredIterationCount
            )
        }

        self.byteCount = byteCount
        self.warmupIterationCount = warmupIterationCount
        self.measuredIterationCount = measuredIterationCount
    }

    public static func rg10FullHD()
        throws(CUDATransferProbeConfigurationError)
        -> CUDATransferProbeConfiguration
    {
        try CUDATransferProbeConfiguration(
            byteCount: 1920 * 1080 * 2,
            warmupIterationCount: 10,
            measuredIterationCount: 100
        )
    }
}
