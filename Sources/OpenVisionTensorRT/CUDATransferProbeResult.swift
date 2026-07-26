import CTensorRTShim

public struct CUDATransferProbeResult:
    Sendable,
    Hashable
{
    public let byteCount: UInt64
    public let warmupIterationCount: UInt32
    public let measuredIterationCount: UInt32
    public let hostToDeviceCopyCount: UInt32
    public let deviceToHostVerificationCopyCount: UInt32
    public let hostFrameAllocationCount: UInt32
    public let deviceFrameAllocationCount: UInt32
    public let frameSizedAllocationCountAfterWarmup: UInt32
    public let cudaErrorCode: Int32
    public let cleanupCUDAErrorCode: Int32
    public let failureStage: CUDATransferStage
    public let cleanupFailureStage: CUDATransferStage
    public let p50Milliseconds: Double
    public let p95Milliseconds: Double
    public let p50GigabytesPerSecond: Double
    public let p95GigabytesPerSecond: Double
    public let hostRegistrationPassed: Bool
    public let sourceAddressPreserved: Bool
    public let inputConsumedEventPassed: Bool
    public let verificationPassed: Bool

    public var isTransferContractSatisfied: Bool {
        let expectedCopyCount =
            UInt64(warmupIterationCount) +
            UInt64(measuredIterationCount)
        return
            UInt64(hostToDeviceCopyCount) == expectedCopyCount &&
            deviceToHostVerificationCopyCount == 1 &&
            hostFrameAllocationCount == 2 &&
            deviceFrameAllocationCount == 1 &&
            frameSizedAllocationCountAfterWarmup == 0 &&
            failureStage == .none &&
            cleanupFailureStage == .none &&
            hostRegistrationPassed &&
            sourceAddressPreserved &&
            inputConsumedEventPassed &&
            verificationPassed
    }

    init(_ result: OVTRTCUDATransferProbeResult) {
        byteCount = result.byteCount
        warmupIterationCount = result.warmupIterationCount
        measuredIterationCount = result.measuredIterationCount
        hostToDeviceCopyCount = result.hostToDeviceCopyCount
        deviceToHostVerificationCopyCount =
            result.deviceToHostVerificationCopyCount
        hostFrameAllocationCount =
            result.hostFrameAllocationCount
        deviceFrameAllocationCount =
            result.deviceFrameAllocationCount
        frameSizedAllocationCountAfterWarmup =
            result.frameSizedAllocationCountAfterWarmup
        cudaErrorCode = result.cudaErrorCode
        cleanupCUDAErrorCode = result.cleanupCUDAErrorCode
        failureStage = CUDATransferStage(result.failureStage)
        cleanupFailureStage = CUDATransferStage(
            result.cleanupFailureStage
        )
        p50Milliseconds = result.p50Milliseconds
        p95Milliseconds = result.p95Milliseconds
        p50GigabytesPerSecond =
            result.p50GigabytesPerSecond
        p95GigabytesPerSecond =
            result.p95GigabytesPerSecond
        hostRegistrationPassed =
            result.hostRegistrationPassed == 1
        sourceAddressPreserved =
            result.sourceAddressPreserved == 1
        inputConsumedEventPassed =
            result.inputConsumedEventPassed == 1
        verificationPassed = result.verificationPassed == 1
    }
}
