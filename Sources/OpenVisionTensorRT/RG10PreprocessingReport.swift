import CTensorRTShim

public struct RG10PreprocessingReport: Sendable, Hashable {
    public let inputByteCount: UInt64
    public let outputElementCount: UInt64
    public let fullFrameHostToDeviceCopyCount: UInt32
    public let kernelLaunchCount: UInt32
    public let deviceToHostVerificationCopyCount: UInt32
    public let frameSizedDeviceAllocationCount: UInt32
    public let explicitFrameSizedDeviceAllocationCountAfterPreparation:
        UInt32
    public let nvrtcCompilationCount: UInt32
    public let cudaErrorCode: Int32
    public let cudaDriverErrorCode: Int32
    public let nvrtcErrorCode: Int32
    public let cleanupCUDAErrorCode: Int32
    public let cleanupCUDADriverErrorCode: Int32
    public let cleanupNVRTCErrorCode: Int32
    public let cleanupDynamicLoaderErrorCode: Int32
    public let failureStage: RG10PreprocessingStage
    public let cleanupFailureStage: RG10PreprocessingStage
    public let gpuMilliseconds: Double
    public let libraryOpenFailures: RG10LibraryOpenFailures
    public let sourceReadCompleted: Bool
    public let sourceReadFencePassed: Bool
    public let outputReadyEventPassed: Bool
    public let verificationPassed: Bool

    public var satisfiesFrameContract: Bool {
        fullFrameHostToDeviceCopyCount == 1 &&
            kernelLaunchCount == 1 &&
            explicitFrameSizedDeviceAllocationCountAfterPreparation == 0 &&
            failureStage == .none &&
            cleanupFailureStage == .none &&
            sourceReadCompleted &&
            sourceReadFencePassed &&
            outputReadyEventPassed
    }

    init(
        preparation: OVTRTRG10PreprocessingResult,
        submission: OVTRTRG10PreprocessingResult,
        completion: OVTRTRG10PreprocessingResult
    ) {
        inputByteCount = submission.inputByteCount
        outputElementCount = completion.outputElementCount
        fullFrameHostToDeviceCopyCount =
            submission.fullFrameHostToDeviceCopyCount
        kernelLaunchCount = submission.kernelLaunchCount
        deviceToHostVerificationCopyCount =
            completion.deviceToHostVerificationCopyCount
        frameSizedDeviceAllocationCount =
            preparation.frameSizedDeviceAllocationCount
        explicitFrameSizedDeviceAllocationCountAfterPreparation =
            max(
                submission
                    .explicitFrameSizedDeviceAllocationCountAfterPreparation,
                completion
                    .explicitFrameSizedDeviceAllocationCountAfterPreparation
            )
        nvrtcCompilationCount = preparation.nvrtcCompilationCount
        cudaErrorCode =
            completion.cudaErrorCode != 0
            ? completion.cudaErrorCode
            : submission.cudaErrorCode
        cudaDriverErrorCode =
            submission.cudaDriverErrorCode != 0
            ? submission.cudaDriverErrorCode
            : completion.cudaDriverErrorCode
        nvrtcErrorCode = preparation.nvrtcErrorCode
        cleanupCUDAErrorCode =
            submission.cleanupCUDAErrorCode != 0
            ? submission.cleanupCUDAErrorCode
            : completion.cleanupCUDAErrorCode
        cleanupCUDADriverErrorCode =
            submission.cleanupCUDADriverErrorCode != 0
            ? submission.cleanupCUDADriverErrorCode
            : completion.cleanupCUDADriverErrorCode
        cleanupNVRTCErrorCode =
            submission.cleanupNVRTCErrorCode != 0
            ? submission.cleanupNVRTCErrorCode
            : completion.cleanupNVRTCErrorCode
        cleanupDynamicLoaderErrorCode =
            submission.cleanupDynamicLoaderErrorCode != 0
            ? submission.cleanupDynamicLoaderErrorCode
            : completion.cleanupDynamicLoaderErrorCode
        failureStage =
            completion.failureStage.rawValue != 0
            ? RG10PreprocessingStage(completion.failureStage)
            : RG10PreprocessingStage(submission.failureStage)
        cleanupFailureStage =
            completion.cleanupFailureStage.rawValue != 0
            ? RG10PreprocessingStage(completion.cleanupFailureStage)
            : RG10PreprocessingStage(submission.cleanupFailureStage)
        gpuMilliseconds = completion.gpuMilliseconds
        libraryOpenFailures = RG10LibraryOpenFailures(
            rawValue:
                preparation.libraryOpenFailureMask |
                submission.libraryOpenFailureMask |
                completion.libraryOpenFailureMask
        )
        sourceReadCompleted =
            submission.sourceReadCompleted == 1
        sourceReadFencePassed =
            submission.sourceReadFencePassed == 1
        outputReadyEventPassed =
            completion.outputReadyEventPassed == 1
        verificationPassed =
            completion.verificationPassed == 1
    }

    init(_ raw: OVTRTRG10PreprocessingResult) {
        self.init(
            preparation: raw,
            submission: raw,
            completion: raw
        )
    }
}
