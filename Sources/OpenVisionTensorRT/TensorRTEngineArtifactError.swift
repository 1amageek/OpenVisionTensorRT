public enum TensorRTEngineArtifactError:
    Error,
    Sendable,
    Equatable
{
    case emptyChecksum
    case invalidChecksum(String)
    case invalidTensorRTVersion(Int32)
    case invalidCUDARuntimeVersion(Int32)
    case invalidComputeCapability(major: Int32, minor: Int32)
    case unsupportedPrecision(TensorRTPrecision)
}
