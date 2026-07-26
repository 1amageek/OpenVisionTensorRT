#include "CTensorRTShim.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {

constexpr uint64_t MAXIMUM_TRANSFER_BYTE_COUNT =
    512ULL * 1024ULL * 1024ULL;
constexpr uint32_t MAXIMUM_ITERATION_COUNT = 10'000;

}  // namespace

#if defined(__linux__) && __has_include(<NvInfer.h>) && \
    __has_include(<NvInferRuntime.h>)
#define OVTRT_HAS_RUNTIME 1
#include <dlfcn.h>
#include <unistd.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#else
#define OVTRT_HAS_RUNTIME 0
#endif

#if OVTRT_HAS_RUNTIME
namespace {

using CUDAVersionFunction = int (*)(int *);
using CUDADeviceCountFunction = int (*)(int *);
using CUDAStream = void *;
using CUDAEvent = void *;
using CUDAMallocFunction = int (*)(void **, size_t);
using CUDAFreeFunction = int (*)(void *);
using CUDAHostRegisterFunction = int (*)(void *, size_t, unsigned int);
using CUDAHostUnregisterFunction = int (*)(void *);
using CUDAStreamCreateWithFlagsFunction =
    int (*)(CUDAStream *, unsigned int);
using CUDAStreamDestroyFunction = int (*)(CUDAStream);
using CUDAStreamSynchronizeFunction = int (*)(CUDAStream);
using CUDAEventCreateWithFlagsFunction =
    int (*)(CUDAEvent *, unsigned int);
using CUDAEventDestroyFunction = int (*)(CUDAEvent);
using CUDAEventRecordFunction = int (*)(CUDAEvent, CUDAStream);
using CUDAEventSynchronizeFunction = int (*)(CUDAEvent);
using CUDAEventElapsedTimeFunction =
    int (*)(float *, CUDAEvent, CUDAEvent);
using CUDAMemcpyAsyncFunction =
    int (*)(void *, void const *, size_t, int, CUDAStream);
using CUDAMemcpyFunction =
    int (*)(void *, void const *, size_t, int);
using TensorRTVersionFunction = int32_t (*)() noexcept;
using TensorRTRuntimeCreateFunction =
    void *(*)(void *, int32_t) noexcept;

constexpr int CUDA_SUCCESS = 0;
constexpr int CUDA_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int CUDA_MEMCPY_DEVICE_TO_HOST = 2;
constexpr unsigned int CUDA_HOST_REGISTER_DEFAULT = 0;
constexpr unsigned int CUDA_STREAM_NON_BLOCKING = 1;
constexpr unsigned int CUDA_EVENT_DEFAULT = 0;

class OVTRTLogger final : public nvinfer1::ILogger {
public:
    void log(
        Severity severity,
        char const *message
    ) noexcept override {
        (void)severity;
        (void)message;
    }
};

void *openLibrary(
    char const *const *names,
    size_t count
) noexcept {
    for (size_t index = 0; index < count; ++index) {
        void *library = dlopen(
            names[index],
            RTLD_NOW | RTLD_LOCAL
        );
        if (library != nullptr) {
            return library;
        }
    }
    return nullptr;
}

template <typename Function>
Function loadFunction(
    void *library,
    char const *name
) noexcept {
    // POSIX specifies that dlsym returns an address suitable for conversion to
    // the requested function pointer. The pointer is never offset, rebound as
    // data, stored past the dlopen owner, or invoked after dlclose.
    return reinterpret_cast<Function>(dlsym(library, name));
}

OVTRTStatus queryRuntime(
    void *tensorRTLibrary,
    void *cudaRuntimeLibrary,
    OVTRTProbeResult *result
) noexcept {
    auto cudaRuntimeGetVersion = loadFunction<CUDAVersionFunction>(
        cudaRuntimeLibrary,
        "cudaRuntimeGetVersion"
    );
    auto cudaDriverGetVersion = loadFunction<CUDAVersionFunction>(
        cudaRuntimeLibrary,
        "cudaDriverGetVersion"
    );
    auto cudaGetDeviceCount = loadFunction<CUDADeviceCountFunction>(
        cudaRuntimeLibrary,
        "cudaGetDeviceCount"
    );
    auto getInferLibVersion = loadFunction<TensorRTVersionFunction>(
        tensorRTLibrary,
        "getInferLibVersion"
    );
    if (
        cudaRuntimeGetVersion == nullptr ||
        cudaDriverGetVersion == nullptr ||
        cudaGetDeviceCount == nullptr ||
        getInferLibVersion == nullptr
    ) {
        return OVTRTStatusUnavailable;
    }

    if (cudaRuntimeGetVersion(&result->cudaRuntimeVersion) != 0) {
        return OVTRTStatusCUDARuntimeFailure;
    }
    if (cudaDriverGetVersion(&result->cudaDriverVersion) != 0) {
        return OVTRTStatusCUDARuntimeFailure;
    }
    if (cudaGetDeviceCount(&result->cudaDeviceCount) != 0) {
        return OVTRTStatusCUDARuntimeFailure;
    }
    result->tensorRTVersion = getInferLibVersion();
    return OVTRTStatusSuccess;
}

double percentile(
    std::vector<double> const &values,
    double fraction
) {
    double rank = std::ceil(
        fraction * static_cast<double>(values.size())
    );
    size_t index = static_cast<size_t>(rank) - 1;
    return values[index];
}

double gigabytesPerSecond(
    uint64_t byteCount,
    double milliseconds
) noexcept {
    if (milliseconds <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(byteCount) /
        (milliseconds * 1'000'000.0);
}

}  // namespace

struct OVTRTRuntime {
    // The logger is declared before the runtime so it outlives every TensorRT
    // callback. This owner is allocated exactly once by ovtrt_runtime_create
    // and deallocated exactly once by ovtrt_runtime_destroy.
    OVTRTLogger logger;
    nvinfer1::IRuntime *runtime{nullptr};
    void *tensorRTLibrary{nullptr};
    void *cudaRuntimeLibrary{nullptr};
};
#else
struct OVTRTRuntime {
    uint8_t unavailable;
};
#endif

OVTRTStatus ovtrt_probe(OVTRTProbeResult *result) {
    if (result == nullptr) {
        return OVTRTStatusInvalidArgument;
    }

    result->tensorRTVersion = 0;
    result->cudaRuntimeVersion = 0;
    result->cudaDriverVersion = 0;
    result->cudaDeviceCount = 0;

#if OVTRT_HAS_RUNTIME
    char const *tensorRTNames[] = {
        "libnvinfer.so.10",
        "libnvinfer.so"
    };
    char const *cudaRuntimeNames[] = {
        "libcudart.so.13",
        "libcudart.so"
    };
    void *tensorRTLibrary = openLibrary(
        tensorRTNames,
        sizeof(tensorRTNames) / sizeof(tensorRTNames[0])
    );
    void *cudaRuntimeLibrary = openLibrary(
        cudaRuntimeNames,
        sizeof(cudaRuntimeNames) / sizeof(cudaRuntimeNames[0])
    );
    if (
        tensorRTLibrary == nullptr ||
        cudaRuntimeLibrary == nullptr
    ) {
        if (cudaRuntimeLibrary != nullptr) {
            dlclose(cudaRuntimeLibrary);
        }
        if (tensorRTLibrary != nullptr) {
            dlclose(tensorRTLibrary);
        }
        return OVTRTStatusUnavailable;
    }

    OVTRTStatus status = queryRuntime(
        tensorRTLibrary,
        cudaRuntimeLibrary,
        result
    );
    dlclose(cudaRuntimeLibrary);
    dlclose(tensorRTLibrary);
    return status;
#else
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_cuda_transfer_probe(
    OVTRTCUDATransferProbeConfiguration const *configuration,
    OVTRTCUDATransferProbeResult *result
) {
    if (configuration == nullptr || result == nullptr) {
        return OVTRTStatusInvalidArgument;
    }

    *result = OVTRTCUDATransferProbeResult{};
    result->byteCount = configuration->byteCount;
    result->warmupIterationCount =
        configuration->warmupIterationCount;
    result->measuredIterationCount =
        configuration->measuredIterationCount;

    uint64_t totalIterationCount =
        static_cast<uint64_t>(configuration->warmupIterationCount) +
        static_cast<uint64_t>(configuration->measuredIterationCount);
    if (
        configuration->byteCount == 0 ||
        configuration->byteCount > MAXIMUM_TRANSFER_BYTE_COUNT ||
        configuration->byteCount >
            static_cast<uint64_t>(
                std::numeric_limits<size_t>::max()
            ) ||
        configuration->measuredIterationCount == 0 ||
        configuration->warmupIterationCount >
            MAXIMUM_ITERATION_COUNT ||
        configuration->measuredIterationCount >
            MAXIMUM_ITERATION_COUNT ||
        totalIterationCount >
            static_cast<uint64_t>(
                std::numeric_limits<uint32_t>::max()
            )
    ) {
        return OVTRTStatusInvalidArgument;
    }

#if OVTRT_HAS_RUNTIME
    char const *cudaRuntimeNames[] = {
        "libcudart.so.13",
        "libcudart.so"
    };
    void *cudaRuntimeLibrary = openLibrary(
        cudaRuntimeNames,
        sizeof(cudaRuntimeNames) / sizeof(cudaRuntimeNames[0])
    );
    if (cudaRuntimeLibrary == nullptr) {
        result->failureStage =
            OVTRTCUDATransferStageLibraryOpen;
        return OVTRTStatusUnavailable;
    }

    auto cudaMalloc = loadFunction<CUDAMallocFunction>(
        cudaRuntimeLibrary,
        "cudaMalloc"
    );
    auto cudaFree = loadFunction<CUDAFreeFunction>(
        cudaRuntimeLibrary,
        "cudaFree"
    );
    auto cudaHostRegister =
        loadFunction<CUDAHostRegisterFunction>(
            cudaRuntimeLibrary,
            "cudaHostRegister"
        );
    auto cudaHostUnregister =
        loadFunction<CUDAHostUnregisterFunction>(
            cudaRuntimeLibrary,
            "cudaHostUnregister"
        );
    auto cudaStreamCreateWithFlags =
        loadFunction<CUDAStreamCreateWithFlagsFunction>(
            cudaRuntimeLibrary,
            "cudaStreamCreateWithFlags"
        );
    auto cudaStreamDestroy =
        loadFunction<CUDAStreamDestroyFunction>(
            cudaRuntimeLibrary,
            "cudaStreamDestroy"
        );
    auto cudaStreamSynchronize =
        loadFunction<CUDAStreamSynchronizeFunction>(
            cudaRuntimeLibrary,
            "cudaStreamSynchronize"
        );
    auto cudaEventCreateWithFlags =
        loadFunction<CUDAEventCreateWithFlagsFunction>(
            cudaRuntimeLibrary,
            "cudaEventCreateWithFlags"
        );
    auto cudaEventDestroy =
        loadFunction<CUDAEventDestroyFunction>(
            cudaRuntimeLibrary,
            "cudaEventDestroy"
        );
    auto cudaEventRecord =
        loadFunction<CUDAEventRecordFunction>(
            cudaRuntimeLibrary,
            "cudaEventRecord"
        );
    auto cudaEventSynchronize =
        loadFunction<CUDAEventSynchronizeFunction>(
            cudaRuntimeLibrary,
            "cudaEventSynchronize"
        );
    auto cudaEventElapsedTime =
        loadFunction<CUDAEventElapsedTimeFunction>(
            cudaRuntimeLibrary,
            "cudaEventElapsedTime"
        );
    auto cudaMemcpyAsync =
        loadFunction<CUDAMemcpyAsyncFunction>(
            cudaRuntimeLibrary,
            "cudaMemcpyAsync"
        );
    auto cudaMemcpy = loadFunction<CUDAMemcpyFunction>(
        cudaRuntimeLibrary,
        "cudaMemcpy"
    );
    if (
        cudaMalloc == nullptr ||
        cudaFree == nullptr ||
        cudaHostRegister == nullptr ||
        cudaHostUnregister == nullptr ||
        cudaStreamCreateWithFlags == nullptr ||
        cudaStreamDestroy == nullptr ||
        cudaStreamSynchronize == nullptr ||
        cudaEventCreateWithFlags == nullptr ||
        cudaEventDestroy == nullptr ||
        cudaEventRecord == nullptr ||
        cudaEventSynchronize == nullptr ||
        cudaEventElapsedTime == nullptr ||
        cudaMemcpyAsync == nullptr ||
        cudaMemcpy == nullptr
    ) {
        result->failureStage =
            OVTRTCUDATransferStageSymbolLoad;
        dlclose(cudaRuntimeLibrary);
        return OVTRTStatusUnavailable;
    }

    size_t byteCount =
        static_cast<size_t>(configuration->byteCount);
    long systemPageSize = sysconf(_SC_PAGESIZE);
    size_t alignment = systemPageSize > 0
        ? static_cast<size_t>(systemPageSize)
        : static_cast<size_t>(4096);
    void *source = nullptr;
    void *verification = nullptr;
    void *device = nullptr;
    CUDAStream stream = nullptr;
    CUDAEvent startEvent = nullptr;
    CUDAEvent endEvent = nullptr;
    bool sourceRegistered = false;
    OVTRTStatus status = OVTRTStatusSuccess;

    // Memory invariants:
    // - source and verification each have exactly one malloc owner;
    // - source is initialized for [0, byteCount) before CUDA registration;
    // - CUDA reads source only while registration and the owner are alive;
    // - event synchronization is the borrow boundary: source cannot be
    //   released or mutated until the recorded H2D transfer completes;
    // - all raw pointers remain scoped to this function and never cross a
    //   Sendable boundary;
    // - device memory is allocated once and freed exactly once;
    // - byteCount, alignment, and iteration arithmetic are validated above.
    if (
        posix_memalign(&source, alignment, byteCount) != 0 ||
        source == nullptr
    ) {
        result->failureStage =
            OVTRTCUDATransferStageHostAllocation;
        status = OVTRTStatusAllocationFailed;
    } else {
        result->hostFrameAllocationCount += 1;
    }
    if (
        status == OVTRTStatusSuccess &&
        (
            posix_memalign(
                &verification,
                alignment,
                byteCount
            ) != 0 ||
            verification == nullptr
        )
    ) {
        result->failureStage =
            OVTRTCUDATransferStageHostAllocation;
        status = OVTRTStatusAllocationFailed;
    } else if (status == OVTRTStatusSuccess) {
        result->hostFrameAllocationCount += 1;
    }

    if (status == OVTRTStatusSuccess) {
        auto *sourceBytes = static_cast<uint8_t *>(source);
        for (size_t index = 0; index < byteCount; ++index) {
            sourceBytes[index] = static_cast<uint8_t>(
                (index * 131U + 17U) & 0xFFU
            );
        }
        std::memset(verification, 0, byteCount);
        result->sourceAddressBefore =
            static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(source)
            );
        int cudaStatus = cudaHostRegister(
            source,
            byteCount,
            CUDA_HOST_REGISTER_DEFAULT
        );
        if (cudaStatus != CUDA_SUCCESS) {
            result->failureStage =
                OVTRTCUDATransferStageHostRegistration;
            result->cudaErrorCode = cudaStatus;
            status = OVTRTStatusCUDARuntimeFailure;
        } else {
            sourceRegistered = true;
            result->hostRegistrationPassed = 1;
            result->sourceAddressAfter =
                static_cast<uint64_t>(
                    reinterpret_cast<uintptr_t>(source)
                );
            result->sourceAddressPreserved =
                result->sourceAddressBefore ==
                    result->sourceAddressAfter
                ? 1
                : 0;
        }
    }

    if (status == OVTRTStatusSuccess) {
        int cudaStatus = cudaStreamCreateWithFlags(
            &stream,
            CUDA_STREAM_NON_BLOCKING
        );
        if (cudaStatus != CUDA_SUCCESS) {
            result->failureStage =
                OVTRTCUDATransferStageStreamCreation;
            result->cudaErrorCode = cudaStatus;
            status = OVTRTStatusCUDARuntimeFailure;
        }
    }
    if (status == OVTRTStatusSuccess) {
        int cudaStatus = cudaEventCreateWithFlags(
            &startEvent,
            CUDA_EVENT_DEFAULT
        );
        if (cudaStatus != CUDA_SUCCESS) {
            result->failureStage =
                OVTRTCUDATransferStageEventCreation;
            result->cudaErrorCode = cudaStatus;
            status = OVTRTStatusCUDARuntimeFailure;
        }
    }
    if (status == OVTRTStatusSuccess) {
        int cudaStatus = cudaEventCreateWithFlags(
            &endEvent,
            CUDA_EVENT_DEFAULT
        );
        if (cudaStatus != CUDA_SUCCESS) {
            result->failureStage =
                OVTRTCUDATransferStageEventCreation;
            result->cudaErrorCode = cudaStatus;
            status = OVTRTStatusCUDARuntimeFailure;
        }
    }
    if (status == OVTRTStatusSuccess) {
        int cudaStatus = cudaMalloc(&device, byteCount);
        if (cudaStatus != CUDA_SUCCESS) {
            result->failureStage =
                OVTRTCUDATransferStageDeviceAllocation;
            result->cudaErrorCode = cudaStatus;
            status = OVTRTStatusCUDARuntimeFailure;
        } else {
            result->deviceFrameAllocationCount = 1;
        }
    }

    std::vector<double> measuredMilliseconds;
    if (status == OVTRTStatusSuccess) {
        try {
            measuredMilliseconds.reserve(
                configuration->measuredIterationCount
            );
        } catch (std::bad_alloc const &) {
            result->failureStage =
                OVTRTCUDATransferStageHostAllocation;
            status = OVTRTStatusAllocationFailed;
        }
    }

    for (
        uint32_t iteration = 0;
        status == OVTRTStatusSuccess &&
            iteration < totalIterationCount;
        ++iteration
    ) {
        int cudaStatus = cudaEventRecord(startEvent, stream);
        if (cudaStatus != CUDA_SUCCESS) {
            result->failureStage =
                OVTRTCUDATransferStageEventRecord;
            result->cudaErrorCode = cudaStatus;
            status = OVTRTStatusCUDARuntimeFailure;
            break;
        }
        cudaStatus = cudaMemcpyAsync(
            device,
            source,
            byteCount,
            CUDA_MEMCPY_HOST_TO_DEVICE,
            stream
        );
        if (cudaStatus != CUDA_SUCCESS) {
            result->failureStage =
                OVTRTCUDATransferStageHostToDevice;
            result->cudaErrorCode = cudaStatus;
            status = OVTRTStatusCUDARuntimeFailure;
            break;
        }
        result->hostToDeviceCopyCount += 1;
        cudaStatus = cudaEventRecord(endEvent, stream);
        if (cudaStatus != CUDA_SUCCESS) {
            result->failureStage =
                OVTRTCUDATransferStageEventRecord;
            result->cudaErrorCode = cudaStatus;
            status = OVTRTStatusCUDARuntimeFailure;
            break;
        }
        cudaStatus = cudaEventSynchronize(endEvent);
        if (cudaStatus != CUDA_SUCCESS) {
            result->failureStage =
                OVTRTCUDATransferStageEventSynchronization;
            result->cudaErrorCode = cudaStatus;
            status = OVTRTStatusCUDARuntimeFailure;
            break;
        }
        result->inputConsumedEventPassed = 1;
        float milliseconds = 0.0F;
        cudaStatus = cudaEventElapsedTime(
            &milliseconds,
            startEvent,
            endEvent
        );
        if (cudaStatus != CUDA_SUCCESS) {
            result->failureStage =
                OVTRTCUDATransferStageEventTiming;
            result->cudaErrorCode = cudaStatus;
            status = OVTRTStatusCUDARuntimeFailure;
            break;
        }
        if (iteration >= configuration->warmupIterationCount) {
            measuredMilliseconds.push_back(
                static_cast<double>(milliseconds)
            );
        }
    }

    if (status == OVTRTStatusSuccess) {
        result->frameSizedAllocationCountAfterWarmup = 0;
        std::sort(
            measuredMilliseconds.begin(),
            measuredMilliseconds.end()
        );
        result->p50Milliseconds = percentile(
            measuredMilliseconds,
            0.50
        );
        result->p95Milliseconds = percentile(
            measuredMilliseconds,
            0.95
        );
        result->p50GigabytesPerSecond = gigabytesPerSecond(
            configuration->byteCount,
            result->p50Milliseconds
        );
        result->p95GigabytesPerSecond = gigabytesPerSecond(
            configuration->byteCount,
            result->p95Milliseconds
        );

        int cudaStatus = cudaMemcpy(
            verification,
            device,
            byteCount,
            CUDA_MEMCPY_DEVICE_TO_HOST
        );
        if (cudaStatus != CUDA_SUCCESS) {
            result->failureStage =
                OVTRTCUDATransferStageDeviceToHostVerification;
            result->cudaErrorCode = cudaStatus;
            status = OVTRTStatusCUDARuntimeFailure;
        } else {
            result->deviceToHostVerificationCopyCount = 1;
            if (
                std::memcmp(source, verification, byteCount) != 0
            ) {
                result->failureStage =
                    OVTRTCUDATransferStageContentVerification;
                status =
                    OVTRTStatusTransferVerificationFailure;
            } else {
                result->verificationPassed = 1;
            }
        }
    }

    auto recordCleanupFailure = [&](
        int cudaStatus,
        OVTRTCUDATransferStage stage
    ) {
        if (
            cudaStatus != CUDA_SUCCESS &&
            result->cleanupFailureStage ==
                OVTRTCUDATransferStageNone
        ) {
            result->cleanupCUDAErrorCode = cudaStatus;
            result->cleanupFailureStage = stage;
            if (status == OVTRTStatusSuccess) {
                status = OVTRTStatusCUDARuntimeFailure;
            }
        }
    };

    if (stream != nullptr) {
        recordCleanupFailure(
            cudaStreamSynchronize(stream),
            OVTRTCUDATransferStageStreamSynchronization
        );
    }
    if (endEvent != nullptr) {
        recordCleanupFailure(
            cudaEventDestroy(endEvent),
            OVTRTCUDATransferStageEventDestruction
        );
    }
    if (startEvent != nullptr) {
        recordCleanupFailure(
            cudaEventDestroy(startEvent),
            OVTRTCUDATransferStageEventDestruction
        );
    }
    if (device != nullptr) {
        recordCleanupFailure(
            cudaFree(device),
            OVTRTCUDATransferStageDeviceDeallocation
        );
    }
    if (stream != nullptr) {
        recordCleanupFailure(
            cudaStreamDestroy(stream),
            OVTRTCUDATransferStageStreamDestruction
        );
    }
    if (sourceRegistered) {
        recordCleanupFailure(
            cudaHostUnregister(source),
            OVTRTCUDATransferStageHostUnregistration
        );
    }
    std::free(verification);
    std::free(source);
    dlclose(cudaRuntimeLibrary);
    return status;
#else
    result->failureStage =
        OVTRTCUDATransferStageLibraryOpen;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_runtime_create(OVTRTRuntime **runtime) {
    if (runtime == nullptr) {
        return OVTRTStatusInvalidArgument;
    }
    *runtime = nullptr;

#if OVTRT_HAS_RUNTIME
    char const *tensorRTNames[] = {
        "libnvinfer.so.10",
        "libnvinfer.so"
    };
    char const *cudaRuntimeNames[] = {
        "libcudart.so.13",
        "libcudart.so"
    };
    void *tensorRTLibrary = openLibrary(
        tensorRTNames,
        sizeof(tensorRTNames) / sizeof(tensorRTNames[0])
    );
    void *cudaRuntimeLibrary = openLibrary(
        cudaRuntimeNames,
        sizeof(cudaRuntimeNames) / sizeof(cudaRuntimeNames[0])
    );
    if (
        tensorRTLibrary == nullptr ||
        cudaRuntimeLibrary == nullptr
    ) {
        if (cudaRuntimeLibrary != nullptr) {
            dlclose(cudaRuntimeLibrary);
        }
        if (tensorRTLibrary != nullptr) {
            dlclose(tensorRTLibrary);
        }
        return OVTRTStatusUnavailable;
    }

    OVTRTProbeResult probeResult{};
    OVTRTStatus probeStatus = queryRuntime(
        tensorRTLibrary,
        cudaRuntimeLibrary,
        &probeResult
    );
    if (probeStatus != OVTRTStatusSuccess) {
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return probeStatus;
    }
    if (probeResult.cudaDeviceCount <= 0) {
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return OVTRTStatusCUDARuntimeFailure;
    }

    OVTRTRuntime *owner = new (std::nothrow) OVTRTRuntime{};
    if (owner == nullptr) {
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return OVTRTStatusAllocationFailed;
    }
    owner->tensorRTLibrary = tensorRTLibrary;
    owner->cudaRuntimeLibrary = cudaRuntimeLibrary;

    auto createInferRuntime =
        loadFunction<TensorRTRuntimeCreateFunction>(
            tensorRTLibrary,
            "createInferRuntime_INTERNAL"
        );
    if (createInferRuntime == nullptr) {
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        delete owner;
        return OVTRTStatusUnavailable;
    }
    owner->runtime = static_cast<nvinfer1::IRuntime *>(
        createInferRuntime(
            &owner->logger,
            NV_TENSORRT_VERSION
        )
    );
    if (owner->runtime == nullptr) {
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        delete owner;
        return OVTRTStatusTensorRTRuntimeFailure;
    }

    *runtime = owner;
    return OVTRTStatusSuccess;
#else
    return OVTRTStatusUnavailable;
#endif
}

void ovtrt_runtime_destroy(OVTRTRuntime *runtime) {
    if (runtime == nullptr) {
        return;
    }

#if OVTRT_HAS_RUNTIME
#if NV_TENSORRT_MAJOR < 10
    runtime->runtime->destroy();
#else
    delete runtime->runtime;
#endif
    dlclose(runtime->cudaRuntimeLibrary);
    dlclose(runtime->tensorRTLibrary);
#endif
    delete runtime;
}
