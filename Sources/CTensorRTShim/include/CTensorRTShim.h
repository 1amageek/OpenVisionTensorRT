#ifndef C_TENSOR_RT_SHIM_H
#define C_TENSOR_RT_SHIM_H

#include <stddef.h>
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
    OVTRTStatusTransferVerificationFailure = 6,
    OVTRTStatusPreprocessingFailure = 7,
    OVTRTStatusNVRTCFailure = 8,
    OVTRTStatusCUDADriverFailure = 9,
    OVTRTStatusResourceBusy = 10,
    OVTRTStatusEngineArtifactFailure = 11,
    OVTRTStatusEngineChecksumMismatch = 12,
    OVTRTStatusEngineDeserializationFailure = 13,
    OVTRTStatusEngineExecutionSetupFailure = 14,
    OVTRTStatusEngineExecutionFailure = 15,
    OVTRTStatusOutputCapacityExceeded = 16
} OVTRTStatus;

typedef struct OVTRTProbeResult {
    int32_t tensorRTVersion;
    int32_t cudaRuntimeVersion;
    int32_t cudaDriverVersion;
    int32_t cudaDeviceCount;
} OVTRTProbeResult;

typedef struct OVTRTRuntime OVTRTRuntime;
typedef struct OVTRTRG10Preprocessor OVTRTRG10Preprocessor;
typedef struct OVTRTEngine OVTRTEngine;

typedef enum OVTRTEngineLoadStage {
    OVTRTEngineLoadStageNone = 0,
    OVTRTEngineLoadStageConfiguration = 1,
    OVTRTEngineLoadStageLibraryOpen = 2,
    OVTRTEngineLoadStageSymbolLoad = 3,
    OVTRTEngineLoadStageFileOpen = 4,
    OVTRTEngineLoadStageFileStat = 5,
    OVTRTEngineLoadStageFileMapping = 6,
    OVTRTEngineLoadStageChecksum = 7,
    OVTRTEngineLoadStageRuntimeCreation = 8,
    OVTRTEngineLoadStageDeserialization = 9,
    OVTRTEngineLoadStageTensorInspection = 10
} OVTRTEngineLoadStage;

typedef enum OVTRTTensorIOMode {
    OVTRTTensorIOModeInput = 0,
    OVTRTTensorIOModeOutput = 1,
    OVTRTTensorIOModeUnknown = 2
} OVTRTTensorIOMode;

typedef enum OVTRTTensorElementType {
    OVTRTTensorElementTypeFloat32 = 0,
    OVTRTTensorElementTypeFloat16 = 1,
    OVTRTTensorElementTypeInt8 = 2,
    OVTRTTensorElementTypeInt32 = 3,
    OVTRTTensorElementTypeBool = 4,
    OVTRTTensorElementTypeUInt8 = 5,
    OVTRTTensorElementTypeFP8 = 6,
    OVTRTTensorElementTypeBF16 = 7,
    OVTRTTensorElementTypeInt64 = 8,
    OVTRTTensorElementTypeInt4 = 9,
    OVTRTTensorElementTypeFP4 = 10,
    OVTRTTensorElementTypeUnknown = 255
} OVTRTTensorElementType;

typedef enum OVTRTShapeSelector {
    OVTRTShapeSelectorDeclared = 0,
    OVTRTShapeSelectorMinimum = 1,
    OVTRTShapeSelectorOptimum = 2,
    OVTRTShapeSelectorMaximum = 3
} OVTRTShapeSelector;

typedef struct OVTRTEngineLoadResult {
    int32_t tensorRTVersion;
    int32_t cudaRuntimeVersion;
    int32_t computeCapabilityMajor;
    int32_t computeCapabilityMinor;
    int32_t systemErrorCode;
    OVTRTEngineLoadStage failureStage;
    uint32_t ioTensorCount;
    uint64_t artifactByteCount;
    uint8_t checksumVerified;
} OVTRTEngineLoadResult;

typedef struct OVTRTEngineTensorInfo {
    OVTRTTensorIOMode ioMode;
    OVTRTTensorElementType elementType;
    int32_t rank;
} OVTRTEngineTensorInfo;

typedef enum OVTRTEngineExecutionStage {
    OVTRTEngineExecutionStageNone = 0,
    OVTRTEngineExecutionStageConfiguration = 1,
    OVTRTEngineExecutionStageContextCreation = 2,
    OVTRTEngineExecutionStageStreamCreation = 3,
    OVTRTEngineExecutionStageEventCreation = 4,
    OVTRTEngineExecutionStageShapeConfiguration = 5,
    OVTRTEngineExecutionStageOutputAllocation = 6,
    OVTRTEngineExecutionStageTensorBinding = 7,
    OVTRTEngineExecutionStageEnqueue = 8,
    OVTRTEngineExecutionStageSynchronization = 9,
    OVTRTEngineExecutionStageOutputInspection = 10,
    OVTRTEngineExecutionStageCleanup = 11
} OVTRTEngineExecutionStage;

typedef struct OVTRTEngineExecutionResult {
    OVTRTEngineExecutionStage failureStage;
    uint32_t outputTensorCount;
    uint32_t persistentDeviceAllocationCount;
    uint32_t frameDeviceAllocationCount;
    uint32_t batchSize;
    uint64_t persistentDeviceAllocationByteCount;
    uint64_t inputByteCount;
    uint64_t outputByteCount;
    uint64_t submissionCount;
    float inferenceMilliseconds;
} OVTRTEngineExecutionResult;

typedef struct OVTRTEngineOutputView {
    void *deviceAddress;
    uint64_t byteCount;
    uint64_t elementCount;
    OVTRTTensorElementType elementType;
    int32_t rank;
} OVTRTEngineOutputView;

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

typedef enum OVTRTRG10Orientation {
    OVTRTRG10OrientationUp = 1,
    OVTRTRG10OrientationUpMirrored = 2,
    OVTRTRG10OrientationDown = 3,
    OVTRTRG10OrientationDownMirrored = 4,
    OVTRTRG10OrientationLeftMirrored = 5,
    OVTRTRG10OrientationRight = 6,
    OVTRTRG10OrientationRightMirrored = 7,
    OVTRTRG10OrientationLeft = 8
} OVTRTRG10Orientation;

typedef enum OVTRTRG10ResizePolicy {
    OVTRTRG10ResizePolicyScaleFill = 0,
    OVTRTRG10ResizePolicyScaleFit = 1,
    OVTRTRG10ResizePolicyCenterCrop = 2
} OVTRTRG10ResizePolicy;

typedef enum OVTRTTensorLayout {
    OVTRTTensorLayoutNCHW = 0,
    OVTRTTensorLayoutNHWC = 1
} OVTRTTensorLayout;

typedef enum OVTRTTensorChannelOrder {
    OVTRTTensorChannelOrderRGB = 0,
    OVTRTTensorChannelOrderBGR = 1
} OVTRTTensorChannelOrder;

typedef enum OVTRTRG10PreprocessingStage {
    OVTRTRG10PreprocessingStageNone = 0,
    OVTRTRG10PreprocessingStageConfiguration = 1,
    OVTRTRG10PreprocessingStageLibraryOpen = 2,
    OVTRTRG10PreprocessingStageSymbolLoad = 3,
    OVTRTRG10PreprocessingStageStreamCreation = 4,
    OVTRTRG10PreprocessingStageEventCreation = 5,
    OVTRTRG10PreprocessingStageInputAllocation = 6,
    OVTRTRG10PreprocessingStageOutputAllocation = 7,
    OVTRTRG10PreprocessingStageConfigurationAllocation = 8,
    OVTRTRG10PreprocessingStageConfigurationTransfer = 9,
    OVTRTRG10PreprocessingStageNVRTCProgramCreation = 10,
    OVTRTRG10PreprocessingStageNVRTCCompilation = 11,
    OVTRTRG10PreprocessingStagePTXAccess = 12,
    OVTRTRG10PreprocessingStageDriverInitialization = 13,
    OVTRTRG10PreprocessingStageModuleLoad = 14,
    OVTRTRG10PreprocessingStageKernelLookup = 15,
    OVTRTRG10PreprocessingStageHostToDevice = 16,
    OVTRTRG10PreprocessingStageKernelLaunch = 17,
    OVTRTRG10PreprocessingStageOutputEvent = 18,
    OVTRTRG10PreprocessingStageOutputSynchronization = 19,
    OVTRTRG10PreprocessingStageEventTiming = 20,
    OVTRTRG10PreprocessingStageOutputReadback = 21,
    OVTRTRG10PreprocessingStageContentVerification = 22,
    OVTRTRG10PreprocessingStageStreamSynchronization = 23,
    OVTRTRG10PreprocessingStageModuleUnload = 24,
    OVTRTRG10PreprocessingStageDeviceDeallocation = 25,
    OVTRTRG10PreprocessingStageEventDestruction = 26,
    OVTRTRG10PreprocessingStageStreamDestruction = 27,
    OVTRTRG10PreprocessingStageNVRTCProgramDestruction = 28,
    OVTRTRG10PreprocessingStageLibraryClose = 29
} OVTRTRG10PreprocessingStage;

typedef enum OVTRTRG10LibraryOpenFailure {
    OVTRTRG10LibraryOpenFailureNone = 0,
    OVTRTRG10LibraryOpenFailureCUDARuntime = 1 << 0,
    OVTRTRG10LibraryOpenFailureCUDADriver = 1 << 1,
    OVTRTRG10LibraryOpenFailureNVRTC = 1 << 2
} OVTRTRG10LibraryOpenFailure;

typedef struct OVTRTRG10PreprocessingConfiguration {
    uint32_t sourceWidth;
    uint32_t sourceHeight;
    uint32_t sourceBytesPerRow;
    uint64_t sourceByteCount;
    uint32_t outputWidth;
    uint32_t outputHeight;
    OVTRTRG10ResizePolicy resizePolicy;
    OVTRTTensorLayout tensorLayout;
    OVTRTTensorChannelOrder channelOrder;
    float blackLevelR;
    float blackLevelGreenR;
    float blackLevelGreenB;
    float blackLevelB;
    float whiteLevel;
    float gainR;
    float gainGreenR;
    float gainGreenB;
    float gainB;
    float colorMatrix00;
    float colorMatrix01;
    float colorMatrix02;
    float colorMatrix10;
    float colorMatrix11;
    float colorMatrix12;
    float colorMatrix20;
    float colorMatrix21;
    float colorMatrix22;
    float letterboxR;
    float letterboxG;
    float letterboxB;
    float normalizationScaleR;
    float normalizationScaleG;
    float normalizationScaleB;
    float normalizationBiasR;
    float normalizationBiasG;
    float normalizationBiasB;
    uint8_t applySRGBTransfer;
} OVTRTRG10PreprocessingConfiguration;

typedef struct OVTRTRG10PreprocessingResult {
    uint64_t inputByteCount;
    uint64_t outputElementCount;
    uint32_t fullFrameHostToDeviceCopyCount;
    uint32_t kernelLaunchCount;
    uint32_t deviceToHostVerificationCopyCount;
    uint32_t frameSizedDeviceAllocationCount;
    uint32_t explicitFrameSizedDeviceAllocationCountAfterPreparation;
    uint32_t nvrtcCompilationCount;
    int32_t cudaErrorCode;
    int32_t cudaDriverErrorCode;
    int32_t nvrtcErrorCode;
    int32_t cleanupCUDAErrorCode;
    int32_t cleanupCUDADriverErrorCode;
    int32_t cleanupNVRTCErrorCode;
    int32_t cleanupDynamicLoaderErrorCode;
    OVTRTRG10PreprocessingStage failureStage;
    OVTRTRG10PreprocessingStage cleanupFailureStage;
    double gpuMilliseconds;
    uint8_t libraryOpenFailureMask;
    uint8_t sourceReadCompleted;
    uint8_t sourceReadFencePassed;
    uint8_t outputReadyEventPassed;
    uint8_t verificationPassed;
} OVTRTRG10PreprocessingResult;

typedef struct OVTRTDeviceTensorView {
    void const *deviceAddress;
    uint64_t byteCount;
    uint64_t elementCount;
    uint32_t width;
    uint32_t height;
    uint32_t channelCount;
    OVTRTTensorLayout layout;
    OVTRTTensorChannelOrder channelOrder;
} OVTRTDeviceTensorView;

OVTRTStatus ovtrt_probe(OVTRTProbeResult *result);

OVTRTStatus ovtrt_cuda_transfer_probe(
    OVTRTCUDATransferProbeConfiguration const *configuration,
    OVTRTCUDATransferProbeResult *result
);

OVTRTStatus ovtrt_rg10_preprocessor_create(
    OVTRTRG10PreprocessingConfiguration const *configuration,
    OVTRTRG10Preprocessor **preprocessor,
    OVTRTRG10PreprocessingResult *result
);

OVTRTStatus ovtrt_rg10_preprocessor_submit(
    OVTRTRG10Preprocessor *preprocessor,
    void const *source,
    uint64_t sourceByteCount,
    OVTRTRG10Orientation orientation,
    OVTRTRG10PreprocessingResult *result
);

OVTRTStatus ovtrt_rg10_preprocessor_wait(
    OVTRTRG10Preprocessor *preprocessor,
    OVTRTRG10PreprocessingResult *result
);

OVTRTStatus ovtrt_rg10_preprocessor_output(
    OVTRTRG10Preprocessor *preprocessor,
    OVTRTDeviceTensorView *output
);

OVTRTStatus ovtrt_rg10_preprocessor_copy_output(
    OVTRTRG10Preprocessor *preprocessor,
    float *destination,
    uint64_t destinationElementCount,
    OVTRTRG10PreprocessingResult *result
);

OVTRTStatus ovtrt_rg10_preprocessor_destroy(
    OVTRTRG10Preprocessor **preprocessor,
    OVTRTRG10PreprocessingResult *result
);

#if defined(OVTRT_ENABLE_TEST_HOOKS)
OVTRTStatus ovtrt_rg10_preprocessor_test_fail_next_cleanup_synchronization(
    OVTRTRG10Preprocessor *preprocessor
);
#endif

OVTRTStatus ovtrt_runtime_create(OVTRTRuntime **runtime);

void ovtrt_runtime_destroy(OVTRTRuntime *runtime);

OVTRTStatus ovtrt_engine_create(
    char const *path,
    char const *expectedChecksum,
    OVTRTEngine **engine,
    OVTRTEngineLoadResult *result
);

OVTRTStatus ovtrt_engine_tensor_name(
    OVTRTEngine *engine,
    uint32_t index,
    char *destination,
    uint32_t destinationCapacity,
    uint32_t *requiredCapacity
);

OVTRTStatus ovtrt_engine_tensor_info(
    OVTRTEngine *engine,
    uint32_t index,
    OVTRTEngineTensorInfo *info
);

OVTRTStatus ovtrt_engine_tensor_dimension(
    OVTRTEngine *engine,
    uint32_t index,
    uint32_t axis,
    OVTRTShapeSelector selector,
    int64_t *dimension
);

OVTRTStatus ovtrt_engine_prepare_execution(
    OVTRTEngine *engine,
    uint64_t const *outputCapacityByteCounts,
    uint32_t outputCapacityCount,
    OVTRTEngineExecutionResult *result
);

OVTRTStatus ovtrt_engine_execute(
    OVTRTEngine *engine,
    void const *inputDeviceAddress,
    uint64_t inputByteCount,
    int64_t const *inputDimensions,
    uint32_t inputRank,
    OVTRTEngineExecutionResult *result
);

OVTRTStatus ovtrt_engine_output(
    OVTRTEngine *engine,
    uint32_t outputIndex,
    OVTRTEngineOutputView *view
);

OVTRTStatus ovtrt_engine_output_dimension(
    OVTRTEngine *engine,
    uint32_t outputIndex,
    uint32_t axis,
    int64_t *dimension
);

OVTRTStatus ovtrt_engine_release_execution(
    OVTRTEngine *engine,
    OVTRTEngineExecutionResult *result
);

void ovtrt_engine_destroy(OVTRTEngine *engine);

#ifdef __cplusplus
}
#endif

#endif
