import OpenVision

public struct TensorRTEngineArtifactDescriptor: Sendable, Hashable {
    public let semanticModel: VisionModelManifest
    public let checksum: String
    public let tensorRTVersion: Int32
    public let cudaRuntimeVersion: Int32
    public let computeCapabilityMajor: Int32
    public let computeCapabilityMinor: Int32
    public let precision: TensorRTPrecision

    public init(
        semanticModel: VisionModelManifest,
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
        guard
            checksum.utf8.count == 64,
            checksum.utf8.allSatisfy({
                (48 ... 57).contains($0) ||
                (97 ... 102).contains($0)
            })
        else {
            throw .invalidChecksum(checksum)
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
        let semanticPrecision: VisionModelPrecision
        switch precision {
        case .float32:
            semanticPrecision = .float32
        case .float16:
            semanticPrecision = .float16
        case .int8:
            semanticPrecision = .int8
        }
        guard semanticModel.quality.permittedPrecisions.contains(
            semanticPrecision
        ) else {
            throw .unsupportedPrecision(precision)
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
