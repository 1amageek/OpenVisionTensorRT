#ifndef C_TENSOR_RT_SHIM_H
#define C_TENSOR_RT_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum OVTRTStatus {
    OVTRTStatusSuccess = 0,
    OVTRTStatusUnavailable = 1,
    OVTRTStatusInvalidArgument = 2,
    OVTRTStatusAllocationFailed = 3,
    OVTRTStatusCUDARuntimeFailure = 4,
    OVTRTStatusTensorRTRuntimeFailure = 5,
    OVTRTStatusTransferVerificationFailure = 6
} OVTRTStatus;

typedef struct OVTRTProbeResult {
    int32_t tensorRTVersion;
    int32_t cudaRuntimeVersion;
    int32_t cudaDriverVersion;
    int32_t cudaDeviceCount;
} OVTRTProbeResult;

typedef struct OVTRTRuntime OVTRTRuntime;

typedef enum OVTRTCUDATransferStage {
    OVTRTCUDATransferStageNone = 0,
    OVTRTCUDATransferStageLibraryOpen = 1,
    OVTRTCUDATransferStageSymbolLoad = 2,
    OVTRTCUDATransferStageHostAllocation = 3,
    OVTRTCUDATransferStageHostRegistration = 4,
    OVTRTCUDATransferStageStreamCreation = 5,
    OVTRTCUDATransferStageEventCreation = 6,
    OVTRTCUDATransferStageDeviceAllocation = 7,
    OVTRTCUDATransferStageHostToDevice = 8,
    OVTRTCUDATransferStageEventRecord = 9,
    OVTRTCUDATransferStageEventSynchronization = 10,
    OVTRTCUDATransferStageEventTiming = 11,
    OVTRTCUDATransferStageDeviceToHostVerification = 12,
    OVTRTCUDATransferStageContentVerification = 13,
    OVTRTCUDATransferStageEventDestruction = 14,
    OVTRTCUDATransferStageDeviceDeallocation = 15,
    OVTRTCUDATransferStageStreamDestruction = 16,
    OVTRTCUDATransferStageHostUnregistration = 17,
    OVTRTCUDATransferStageStreamSynchronization = 18
} OVTRTCUDATransferStage;

typedef struct OVTRTCUDATransferProbeConfiguration {
    uint64_t byteCount;
    uint32_t warmupIterationCount;
    uint32_t measuredIterationCount;
} OVTRTCUDATransferProbeConfiguration;

typedef struct OVTRTCUDATransferProbeResult {
    uint64_t byteCount;
    uint64_t sourceAddressBefore;
    uint64_t sourceAddressAfter;
    uint32_t warmupIterationCount;
    uint32_t measuredIterationCount;
    uint32_t hostToDeviceCopyCount;
    uint32_t deviceToHostVerificationCopyCount;
    uint32_t hostFrameAllocationCount;
    uint32_t deviceFrameAllocationCount;
    uint32_t frameSizedAllocationCountAfterWarmup;
    int32_t cudaErrorCode;
    int32_t cleanupCUDAErrorCode;
    OVTRTCUDATransferStage failureStage;
    OVTRTCUDATransferStage cleanupFailureStage;
    double p50Milliseconds;
    double p95Milliseconds;
    double p50GigabytesPerSecond;
    double p95GigabytesPerSecond;
    uint8_t hostRegistrationPassed;
    uint8_t sourceAddressPreserved;
    uint8_t inputConsumedEventPassed;
    uint8_t verificationPassed;
} OVTRTCUDATransferProbeResult;

OVTRTStatus ovtrt_probe(OVTRTProbeResult *result);

OVTRTStatus ovtrt_cuda_transfer_probe(
    OVTRTCUDATransferProbeConfiguration const *configuration,
    OVTRTCUDATransferProbeResult *result
);

OVTRTStatus ovtrt_runtime_create(OVTRTRuntime **runtime);

void ovtrt_runtime_destroy(OVTRTRuntime *runtime);

#ifdef __cplusplus
}
#endif

#endif
