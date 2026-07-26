import CTensorRTShim

public struct TensorRTPosePipelineReport:
    Sendable,
    Hashable
{
    public let selectedRegionCount: Int
    public let regionSelectionKernelLaunchCount: Int
    public let regionAffineKernelLaunchCount: Int
    public let simCCDecodeKernelLaunchCount: Int
    public let compactDeviceToHostCopyCount: Int
    public let explicitFrameDeviceAllocationCount: Int
    public let persistentDeviceAllocationByteCount: UInt64
    public let poseInputByteCount: UInt64
    public let compactReadbackByteCount: UInt64
    public let failureStage: TensorRTPosePipelineStage
    public let cleanupFailureStage: TensorRTPosePipelineStage
    public let cudaErrorCode: Int32
    public let cudaDriverErrorCode: Int32
    public let nvrtcErrorCode: Int32

    init(_ result: OVTRTPosePipelineResult) {
        selectedRegionCount = Int(result.selectedRegionCount)
        regionSelectionKernelLaunchCount =
            Int(result.regionSelectionKernelLaunchCount)
        regionAffineKernelLaunchCount =
            Int(result.regionAffineKernelLaunchCount)
        simCCDecodeKernelLaunchCount =
            Int(result.simCCDecodeKernelLaunchCount)
        compactDeviceToHostCopyCount =
            Int(result.compactDeviceToHostCopyCount)
        explicitFrameDeviceAllocationCount =
            Int(result.explicitFrameDeviceAllocationCount)
        persistentDeviceAllocationByteCount =
            result.persistentDeviceAllocationByteCount
        poseInputByteCount = result.poseInputByteCount
        compactReadbackByteCount = result.compactReadbackByteCount
        failureStage = TensorRTPosePipelineStage(
            rawValue: Int32(result.failureStage.rawValue)
        ) ?? .unknown
        cleanupFailureStage = TensorRTPosePipelineStage(
            rawValue: Int32(result.cleanupFailureStage.rawValue)
        ) ?? .unknown
        cudaErrorCode = result.cudaErrorCode
        cudaDriverErrorCode = result.cudaDriverErrorCode
        nvrtcErrorCode = result.nvrtcErrorCode
    }

    init(
        selectedRegionCount: Int,
        regionSelectionKernelLaunchCount: Int,
        regionAffineKernelLaunchCount: Int,
        simCCDecodeKernelLaunchCount: Int,
        compactDeviceToHostCopyCount: Int,
        explicitFrameDeviceAllocationCount: Int,
        persistentDeviceAllocationByteCount: UInt64,
        poseInputByteCount: UInt64,
        compactReadbackByteCount: UInt64,
        failureStage: TensorRTPosePipelineStage = .none,
        cleanupFailureStage: TensorRTPosePipelineStage = .none,
        cudaErrorCode: Int32 = 0,
        cudaDriverErrorCode: Int32 = 0,
        nvrtcErrorCode: Int32 = 0
    ) {
        self.selectedRegionCount = selectedRegionCount
        self.regionSelectionKernelLaunchCount =
            regionSelectionKernelLaunchCount
        self.regionAffineKernelLaunchCount =
            regionAffineKernelLaunchCount
        self.simCCDecodeKernelLaunchCount =
            simCCDecodeKernelLaunchCount
        self.compactDeviceToHostCopyCount =
            compactDeviceToHostCopyCount
        self.explicitFrameDeviceAllocationCount =
            explicitFrameDeviceAllocationCount
        self.persistentDeviceAllocationByteCount =
            persistentDeviceAllocationByteCount
        self.poseInputByteCount = poseInputByteCount
        self.compactReadbackByteCount = compactReadbackByteCount
        self.failureStage = failureStage
        self.cleanupFailureStage = cleanupFailureStage
        self.cudaErrorCode = cudaErrorCode
        self.cudaDriverErrorCode = cudaDriverErrorCode
        self.nvrtcErrorCode = nvrtcErrorCode
    }
}
