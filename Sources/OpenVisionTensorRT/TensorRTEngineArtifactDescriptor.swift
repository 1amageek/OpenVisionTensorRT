import OpenVision

public struct TensorRTEngineArtifactDescriptor: Sendable, Hashable {
    public let semanticModel: VisionModelDescriptor
    public let checksum: String
    public let tensorRTVersion: Int32
    public let cudaRuntimeVersion: Int32
    public let computeCapabilityMajor: Int32
    public let computeCapabilityMinor: Int32
    public let precision: TensorRTPrecision

    public init(
        semanticModel: VisionModelDescriptor,
        checksum: String,
        tensorRTVersion: Int32,
        cudaRuntimeVersion: Int32,
        computeCapabilityMajor: Int32,
        computeCapabilityMinor: Int32,
        precision: TensorRTPrecision
    ) throws(TensorRTEngineArtifactError) {
        guard !checksum.isEmpty else {
            throw .emptyChecksum
        }
        guard tensorRTVersion > 0 else {
            throw .invalidTensorRTVersion(tensorRTVersion)
        }
        guard cudaRuntimeVersion > 0 else {
            throw .invalidCUDARuntimeVersion(cudaRuntimeVersion)
        }
        guard
            computeCapabilityMajor > 0,
            computeCapabilityMinor >= 0
        else {
            throw .invalidComputeCapability(
                major: computeCapabilityMajor,
                minor: computeCapabilityMinor
            )
        }

        self.semanticModel = semanticModel
        self.checksum = checksum
        self.tensorRTVersion = tensorRTVersion
        self.cudaRuntimeVersion = cudaRuntimeVersion
        self.computeCapabilityMajor = computeCapabilityMajor
        self.computeCapabilityMinor = computeCapabilityMinor
        self.precision = precision
    }
}
