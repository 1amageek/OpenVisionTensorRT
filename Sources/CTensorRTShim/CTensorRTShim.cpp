#include "CTensorRTShim.h"

#include <new>

#if defined(__linux__) && __has_include(<NvInfer.h>) && \
    __has_include(<NvInferRuntime.h>)
#define OVTRT_HAS_RUNTIME 1
#include <dlfcn.h>
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
using TensorRTVersionFunction = int32_t (*)() noexcept;
using TensorRTRuntimeCreateFunction =
    void *(*)(void *, int32_t) noexcept;

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
