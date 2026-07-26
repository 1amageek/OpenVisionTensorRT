import CTensorRTShim

public struct TensorRTEngineLoadReport: Sendable, Hashable {
    public let tensorRTVersion: Int32
    public let cudaRuntimeVersion: Int32
    public let computeCapabilityMajor: Int32
    public let computeCapabilityMinor: Int32
    public let systemErrorCode: Int32
    public let failureStage: TensorRTEngineLoadStage
    public let ioTensorCount: Int
    public let artifactByteCount: UInt64
    public let checksumVerified: Bool

    init(_ result: OVTRTEngineLoadResult) {
        tensorRTVersion = result.tensorRTVersion
        cudaRuntimeVersion = result.cudaRuntimeVersion
        computeCapabilityMajor = result.computeCapabilityMajor
        computeCapabilityMinor = result.computeCapabilityMinor
        systemErrorCode = result.systemErrorCode
        failureStage = TensorRTEngineLoadStage(result.failureStage)
        ioTensorCount = Int(result.ioTensorCount)
        artifactByteCount = result.artifactByteCount
        checksumVerified = result.checksumVerified != 0
    }
}
