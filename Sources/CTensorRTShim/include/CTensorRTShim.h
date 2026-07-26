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
    OVTRTStatusTensorRTRuntimeFailure = 5
} OVTRTStatus;

typedef struct OVTRTProbeResult {
    int32_t tensorRTVersion;
    int32_t cudaRuntimeVersion;
    int32_t cudaDriverVersion;
    int32_t cudaDeviceCount;
} OVTRTProbeResult;

typedef struct OVTRTRuntime OVTRTRuntime;

OVTRTStatus ovtrt_probe(OVTRTProbeResult *result);

OVTRTStatus ovtrt_runtime_create(OVTRTRuntime **runtime);

void ovtrt_runtime_destroy(OVTRTRuntime *runtime);

#ifdef __cplusplus
}
#endif

#endif
