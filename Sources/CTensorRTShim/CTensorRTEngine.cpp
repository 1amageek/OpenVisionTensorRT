#include "CTensorRTShim.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace {

bool portableValidChecksum(
    char const *checksum
) noexcept {
    if (checksum == nullptr || std::strlen(checksum) != 64) {
        return false;
    }
    for (size_t index = 0; index < 64; ++index) {
        char value = checksum[index];
        if (
            !(
                (value >= '0' && value <= '9') ||
                (value >= 'a' && value <= 'f')
            )
        ) {
            return false;
        }
    }
    return true;
}

}  // namespace

#if defined(__linux__) && __has_include(<NvInfer.h>) && \
    __has_include(<NvInferRuntime.h>)
#define OVTRT_HAS_ENGINE_RUNTIME 1
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#else
#define OVTRT_HAS_ENGINE_RUNTIME 0
#endif

#if OVTRT_HAS_ENGINE_RUNTIME
namespace {

using CUDAVersionFunction = int (*)(int *);
using CUDADeviceCountFunction = int (*)(int *);
using CUDADeviceAttributeFunction = int (*)(int *, int, int);
using CUDAStream = cudaStream_t;
using CUDAEvent = cudaEvent_t;
using CUDAMallocFunction = int (*)(void **, size_t);
using CUDAFreeFunction = int (*)(void *);
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
using TensorRTVersionFunction = int32_t (*)() noexcept;
using TensorRTRuntimeCreateFunction =
    void *(*)(void *, int32_t) noexcept;

constexpr int CUDA_SUCCESS = 0;
constexpr int CUDA_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CUDA_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
constexpr unsigned int CUDA_STREAM_NON_BLOCKING = 1;
constexpr unsigned int CUDA_EVENT_DEFAULT = 0;
constexpr uint64_t MAXIMUM_EXECUTION_DEVICE_BYTE_COUNT =
    1024ULL * 1024ULL * 1024ULL;
constexpr size_t SHA256_BLOCK_BYTE_COUNT = 64;
constexpr size_t SHA256_DIGEST_BYTE_COUNT = 32;

class EngineLogger final : public nvinfer1::ILogger {
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
        void *library = dlopen(names[index], RTLD_NOW | RTLD_LOCAL);
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
    // POSIX defines the dlsym result as suitable for conversion to a function
    // pointer. The function never outlives its owning dlopen handle.
    return reinterpret_cast<Function>(dlsym(library, name));
}

uint32_t rotateRight(uint32_t value, uint32_t count) noexcept {
    return (value >> count) | (value << (32U - count));
}

class SHA256 final {
public:
    SHA256() noexcept
        : state_{
            0x6a09e667U,
            0xbb67ae85U,
            0x3c6ef372U,
            0xa54ff53aU,
            0x510e527fU,
            0x9b05688cU,
            0x1f83d9abU,
            0x5be0cd19U
        } {}

    void update(
        uint8_t const *bytes,
        size_t byteCount
    ) noexcept {
        if (byteCount == 0) {
            return;
        }
        totalByteCount_ += static_cast<uint64_t>(byteCount);
        size_t offset = 0;
        if (bufferedByteCount_ != 0) {
            size_t available =
                SHA256_BLOCK_BYTE_COUNT - bufferedByteCount_;
            size_t copied = byteCount < available
                ? byteCount
                : available;
            std::memcpy(
                buffer_ + bufferedByteCount_,
                bytes,
                copied
            );
            bufferedByteCount_ += copied;
            offset += copied;
            if (bufferedByteCount_ == SHA256_BLOCK_BYTE_COUNT) {
                transform(buffer_);
                bufferedByteCount_ = 0;
            }
        }
        while (
            byteCount - offset >= SHA256_BLOCK_BYTE_COUNT
        ) {
            transform(bytes + offset);
            offset += SHA256_BLOCK_BYTE_COUNT;
        }
        size_t remaining = byteCount - offset;
        if (remaining != 0) {
            std::memcpy(buffer_, bytes + offset, remaining);
            bufferedByteCount_ = remaining;
        }
    }

    void finalize(
        uint8_t digest[SHA256_DIGEST_BYTE_COUNT]
    ) noexcept {
        uint64_t totalBitCount = totalByteCount_ * 8U;
        buffer_[bufferedByteCount_++] = 0x80U;
        if (bufferedByteCount_ > 56) {
            std::memset(
                buffer_ + bufferedByteCount_,
                0,
                SHA256_BLOCK_BYTE_COUNT - bufferedByteCount_
            );
            transform(buffer_);
            bufferedByteCount_ = 0;
        }
        std::memset(
            buffer_ + bufferedByteCount_,
            0,
            56 - bufferedByteCount_
        );
        for (size_t index = 0; index < 8; ++index) {
            buffer_[63 - index] = static_cast<uint8_t>(
                totalBitCount >> (index * 8)
            );
        }
        transform(buffer_);
        for (size_t word = 0; word < 8; ++word) {
            for (size_t byte = 0; byte < 4; ++byte) {
                digest[word * 4 + byte] = static_cast<uint8_t>(
                    state_[word] >> (24 - byte * 8)
                );
            }
        }
    }

private:
    void transform(
        uint8_t const block[SHA256_BLOCK_BYTE_COUNT]
    ) noexcept {
        static constexpr uint32_t constants[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
        };
        uint32_t schedule[64]{};
        for (size_t index = 0; index < 16; ++index) {
            size_t offset = index * 4;
            schedule[index] =
                (static_cast<uint32_t>(block[offset]) << 24) |
                (static_cast<uint32_t>(block[offset + 1]) << 16) |
                (static_cast<uint32_t>(block[offset + 2]) << 8) |
                static_cast<uint32_t>(block[offset + 3]);
        }
        for (size_t index = 16; index < 64; ++index) {
            uint32_t first =
                rotateRight(schedule[index - 15], 7) ^
                rotateRight(schedule[index - 15], 18) ^
                (schedule[index - 15] >> 3);
            uint32_t second =
                rotateRight(schedule[index - 2], 17) ^
                rotateRight(schedule[index - 2], 19) ^
                (schedule[index - 2] >> 10);
            schedule[index] =
                schedule[index - 16] +
                first +
                schedule[index - 7] +
                second;
        }

        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];
        for (size_t index = 0; index < 64; ++index) {
            uint32_t sum1 =
                rotateRight(e, 6) ^
                rotateRight(e, 11) ^
                rotateRight(e, 25);
            uint32_t choose = (e & f) ^ ((~e) & g);
            uint32_t temporary1 =
                h + sum1 + choose + constants[index] +
                schedule[index];
            uint32_t sum0 =
                rotateRight(a, 2) ^
                rotateRight(a, 13) ^
                rotateRight(a, 22);
            uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    uint32_t state_[8];
    uint8_t buffer_[SHA256_BLOCK_BYTE_COUNT]{};
    size_t bufferedByteCount_{0};
    uint64_t totalByteCount_{0};
};

bool checksumMatches(
    uint8_t const *bytes,
    size_t byteCount,
    char const *expected
) noexcept {
    SHA256 hash;
    hash.update(bytes, byteCount);
    uint8_t digest[SHA256_DIGEST_BYTE_COUNT]{};
    hash.finalize(digest);
    static constexpr char hexadecimal[] = "0123456789abcdef";
    uint8_t difference = 0;
    for (
        size_t index = 0;
        index < SHA256_DIGEST_BYTE_COUNT;
        ++index
    ) {
        difference |= static_cast<uint8_t>(
            expected[index * 2] ^
            hexadecimal[digest[index] >> 4]
        );
        difference |= static_cast<uint8_t>(
            expected[index * 2 + 1] ^
            hexadecimal[digest[index] & 0x0f]
        );
    }
    return difference == 0;
}

OVTRTTensorElementType elementType(
    nvinfer1::DataType value
) noexcept {
    switch (value) {
    case nvinfer1::DataType::kFLOAT:
        return OVTRTTensorElementTypeFloat32;
    case nvinfer1::DataType::kHALF:
        return OVTRTTensorElementTypeFloat16;
    case nvinfer1::DataType::kINT8:
        return OVTRTTensorElementTypeInt8;
    case nvinfer1::DataType::kINT32:
        return OVTRTTensorElementTypeInt32;
    case nvinfer1::DataType::kBOOL:
        return OVTRTTensorElementTypeBool;
    case nvinfer1::DataType::kUINT8:
        return OVTRTTensorElementTypeUInt8;
    case nvinfer1::DataType::kFP8:
        return OVTRTTensorElementTypeFP8;
    case nvinfer1::DataType::kBF16:
        return OVTRTTensorElementTypeBF16;
    case nvinfer1::DataType::kINT64:
        return OVTRTTensorElementTypeInt64;
    case nvinfer1::DataType::kINT4:
        return OVTRTTensorElementTypeInt4;
    case nvinfer1::DataType::kFP4:
        return OVTRTTensorElementTypeFP4;
    case nvinfer1::DataType::kE8M0:
        return OVTRTTensorElementTypeUnknown;
    }
    return OVTRTTensorElementTypeUnknown;
}

uint64_t elementByteCount(
    nvinfer1::DataType value
) noexcept {
    switch (value) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
        return 2;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kFP8:
    case nvinfer1::DataType::kE8M0:
        return 1;
    case nvinfer1::DataType::kINT64:
        return 8;
    case nvinfer1::DataType::kINT4:
    case nvinfer1::DataType::kFP4:
        return 0;
    }
    return 0;
}

bool dimensionsElementCount(
    nvinfer1::Dims const &dimensions,
    uint64_t *elementCount
) noexcept {
    if (
        elementCount == nullptr ||
        dimensions.nbDims <= 0 ||
        dimensions.nbDims > nvinfer1::Dims::MAX_DIMS
    ) {
        return false;
    }
    uint64_t count = 1;
    for (int32_t axis = 0; axis < dimensions.nbDims; ++axis) {
        int64_t dimension = dimensions.d[axis];
        if (
            dimension < 0 ||
            static_cast<uint64_t>(dimension) >
                std::numeric_limits<uint64_t>::max() / count
        ) {
            return false;
        }
        count *= static_cast<uint64_t>(dimension);
    }
    *elementCount = count;
    return true;
}

class FixedOutputAllocator final :
    public nvinfer1::IOutputAllocator
{
public:
    FixedOutputAllocator(
        void *address,
        uint64_t capacity
    ) noexcept
        : address_(address),
          capacity_(capacity) {}

    void *reallocateOutput(
        char const *tensorName,
        void *currentMemory,
        uint64_t size,
        uint64_t alignment
    ) noexcept override {
        (void)tensorName;
        (void)alignment;
        return allocation(currentMemory, size);
    }

    void *reallocateOutputAsync(
        char const *tensorName,
        void *currentMemory,
        uint64_t size,
        uint64_t alignment,
        cudaStream_t stream
    ) noexcept override {
        (void)tensorName;
        (void)alignment;
        (void)stream;
        return allocation(currentMemory, size);
    }

    void notifyShape(
        char const *tensorName,
        nvinfer1::Dims const &dimensions
    ) noexcept override {
        (void)tensorName;
        finalDimensions_ = dimensions;
        shapeWasNotified_ = true;
    }

    void reset() noexcept {
        requestedByteCount_ = 0;
        capacityWasExceeded_ = false;
        shapeWasNotified_ = false;
        finalDimensions_ = nvinfer1::Dims{};
    }

    uint64_t requestedByteCount() const noexcept {
        return requestedByteCount_;
    }

    bool capacityWasExceeded() const noexcept {
        return capacityWasExceeded_;
    }

    bool shapeWasNotified() const noexcept {
        return shapeWasNotified_;
    }

    nvinfer1::Dims finalDimensions() const noexcept {
        return finalDimensions_;
    }

private:
    void *allocation(
        void *currentMemory,
        uint64_t size
    ) noexcept {
        requestedByteCount_ = size;
        if (
            size > capacity_ ||
            (
                currentMemory != nullptr &&
                currentMemory != address_
            )
        ) {
            capacityWasExceeded_ = true;
            return nullptr;
        }
        return address_;
    }

    void *address_;
    uint64_t capacity_;
    uint64_t requestedByteCount_{0};
    bool capacityWasExceeded_{false};
    bool shapeWasNotified_{false};
    nvinfer1::Dims finalDimensions_{};
};

struct ExecutionOutput {
    char const *name{nullptr};
    void *deviceAddress{nullptr};
    uint64_t capacityByteCount{0};
    uint64_t byteCount{0};
    uint64_t elementCount{0};
    nvinfer1::DataType dataType{nvinfer1::DataType::kFLOAT};
    nvinfer1::Dims dimensions{};
    FixedOutputAllocator *allocator{nullptr};
};

}  // namespace

struct OVTRTEngine {
    // Ownership and unsafe invariants:
    // - logger outlives both TensorRT owners;
    // - engine is destroyed before runtime and both precede dlclose;
    // - mapped plan bytes are read-only, initialized for artifactByteCount,
    //   never rebound, and do not escape deserializeCudaEngine;
    // - TensorRT tensor names remain borrowed only during each C call.
    EngineLogger logger;
    nvinfer1::IRuntime *runtime{nullptr};
    nvinfer1::ICudaEngine *engine{nullptr};
    void *tensorRTLibrary{nullptr};
    void *cudaRuntimeLibrary{nullptr};
    CUDAMallocFunction cudaMalloc{nullptr};
    CUDAFreeFunction cudaFree{nullptr};
    CUDAStreamCreateWithFlagsFunction cudaStreamCreateWithFlags{nullptr};
    CUDAStreamDestroyFunction cudaStreamDestroy{nullptr};
    CUDAStreamSynchronizeFunction cudaStreamSynchronize{nullptr};
    CUDAEventCreateWithFlagsFunction cudaEventCreateWithFlags{nullptr};
    CUDAEventDestroyFunction cudaEventDestroy{nullptr};
    CUDAEventRecordFunction cudaEventRecord{nullptr};
    CUDAEventSynchronizeFunction cudaEventSynchronize{nullptr};
    CUDAEventElapsedTimeFunction cudaEventElapsedTime{nullptr};
    nvinfer1::IExecutionContext *executionContext{nullptr};
    CUDAStream executionStream{nullptr};
    CUDAEvent executionStart{nullptr};
    CUDAEvent executionStop{nullptr};
    ExecutionOutput *outputs{nullptr};
    uint32_t outputCount{0};
    char const *inputName{nullptr};
    nvinfer1::Dims maximumInputDimensions{};
    uint64_t persistentDeviceAllocationByteCount{0};
    uint64_t submissionCount{0};
};
#else
struct OVTRTEngine {
    uint8_t unavailable;
};
#endif

#if OVTRT_HAS_ENGINE_RUNTIME
namespace {

void populateExecutionResources(
    OVTRTEngine const *engine,
    OVTRTEngineExecutionResult *result
) noexcept {
    result->outputTensorCount = engine->outputCount;
    result->persistentDeviceAllocationCount =
        engine->outputCount;
    result->persistentDeviceAllocationByteCount =
        engine->persistentDeviceAllocationByteCount;
    result->submissionCount = engine->submissionCount;
}

OVTRTStatus releaseExecutionResources(
    OVTRTEngine *engine,
    OVTRTEngineExecutionResult *result
) noexcept {
    if (
        engine->executionStream != nullptr &&
        engine->cudaStreamSynchronize(
            engine->executionStream
        ) != CUDA_SUCCESS
    ) {
        result->failureStage =
            OVTRTEngineExecutionStageCleanup;
        populateExecutionResources(engine, result);
        return OVTRTStatusCUDARuntimeFailure;
    }

    delete engine->executionContext;
    engine->executionContext = nullptr;

    bool cleanupFailed = false;
    if (engine->outputs != nullptr) {
        for (
            uint32_t index = 0;
            index < engine->outputCount;
            ++index
        ) {
            ExecutionOutput &output = engine->outputs[index];
            delete output.allocator;
            output.allocator = nullptr;
            if (
                output.deviceAddress != nullptr &&
                engine->cudaFree(output.deviceAddress) !=
                    CUDA_SUCCESS
            ) {
                cleanupFailed = true;
            } else {
                output.deviceAddress = nullptr;
            }
        }
    }
    if (
        engine->executionStart != nullptr &&
        engine->cudaEventDestroy(engine->executionStart) !=
            CUDA_SUCCESS
    ) {
        cleanupFailed = true;
    } else {
        engine->executionStart = nullptr;
    }
    if (
        engine->executionStop != nullptr &&
        engine->cudaEventDestroy(engine->executionStop) !=
            CUDA_SUCCESS
    ) {
        cleanupFailed = true;
    } else {
        engine->executionStop = nullptr;
    }
    if (
        engine->executionStream != nullptr &&
        engine->cudaStreamDestroy(engine->executionStream) !=
            CUDA_SUCCESS
    ) {
        cleanupFailed = true;
    } else {
        engine->executionStream = nullptr;
    }
    if (cleanupFailed) {
        result->failureStage =
            OVTRTEngineExecutionStageCleanup;
        populateExecutionResources(engine, result);
        return OVTRTStatusCUDARuntimeFailure;
    }

    delete[] engine->outputs;
    engine->outputs = nullptr;
    engine->outputCount = 0;
    engine->inputName = nullptr;
    engine->maximumInputDimensions = nvinfer1::Dims{};
    engine->persistentDeviceAllocationByteCount = 0;
    populateExecutionResources(engine, result);
    return OVTRTStatusSuccess;
}

OVTRTStatus statusAfterPreparationCleanup(
    OVTRTEngine *engine,
    OVTRTEngineExecutionResult *result,
    OVTRTStatus preparationStatus
) noexcept {
    OVTRTEngineExecutionResult cleanup{};
    OVTRTStatus cleanupStatus =
        releaseExecutionResources(engine, &cleanup);
    if (cleanupStatus != OVTRTStatusSuccess) {
        *result = cleanup;
        return cleanupStatus;
    }
    return preparationStatus;
}

}  // namespace
#endif

OVTRTStatus ovtrt_engine_create(
    char const *path,
    char const *expectedChecksum,
    OVTRTEngine **engine,
    OVTRTEngineLoadResult *result
) {
    if (
        path == nullptr ||
        path[0] == '\0' ||
        engine == nullptr ||
        result == nullptr
    ) {
        return OVTRTStatusInvalidArgument;
    }
    *engine = nullptr;
    *result = OVTRTEngineLoadResult{};
    if (!portableValidChecksum(expectedChecksum)) {
        result->failureStage = OVTRTEngineLoadStageConfiguration;
        return OVTRTStatusInvalidArgument;
    }

#if OVTRT_HAS_ENGINE_RUNTIME
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
        result->failureStage = OVTRTEngineLoadStageLibraryOpen;
        if (cudaRuntimeLibrary != nullptr) {
            dlclose(cudaRuntimeLibrary);
        }
        if (tensorRTLibrary != nullptr) {
            dlclose(tensorRTLibrary);
        }
        return OVTRTStatusUnavailable;
    }

    auto getInferLibVersion = loadFunction<TensorRTVersionFunction>(
        tensorRTLibrary,
        "getInferLibVersion"
    );
    auto createInferRuntime =
        loadFunction<TensorRTRuntimeCreateFunction>(
            tensorRTLibrary,
            "createInferRuntime_INTERNAL"
        );
    auto cudaRuntimeGetVersion = loadFunction<CUDAVersionFunction>(
        cudaRuntimeLibrary,
        "cudaRuntimeGetVersion"
    );
    auto cudaGetDeviceCount = loadFunction<CUDADeviceCountFunction>(
        cudaRuntimeLibrary,
        "cudaGetDeviceCount"
    );
    auto cudaDeviceGetAttribute =
        loadFunction<CUDADeviceAttributeFunction>(
            cudaRuntimeLibrary,
            "cudaDeviceGetAttribute"
        );
    auto cudaMalloc = loadFunction<CUDAMallocFunction>(
        cudaRuntimeLibrary,
        "cudaMalloc"
    );
    auto cudaFree = loadFunction<CUDAFreeFunction>(
        cudaRuntimeLibrary,
        "cudaFree"
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
    if (
        getInferLibVersion == nullptr ||
        createInferRuntime == nullptr ||
        cudaRuntimeGetVersion == nullptr ||
        cudaGetDeviceCount == nullptr ||
        cudaDeviceGetAttribute == nullptr ||
        cudaMalloc == nullptr ||
        cudaFree == nullptr ||
        cudaStreamCreateWithFlags == nullptr ||
        cudaStreamDestroy == nullptr ||
        cudaStreamSynchronize == nullptr ||
        cudaEventCreateWithFlags == nullptr ||
        cudaEventDestroy == nullptr ||
        cudaEventRecord == nullptr ||
        cudaEventSynchronize == nullptr ||
        cudaEventElapsedTime == nullptr
    ) {
        result->failureStage = OVTRTEngineLoadStageSymbolLoad;
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return OVTRTStatusUnavailable;
    }

    result->tensorRTVersion = getInferLibVersion();
    int deviceCount = 0;
    if (
        cudaRuntimeGetVersion(&result->cudaRuntimeVersion) !=
            CUDA_SUCCESS ||
        cudaGetDeviceCount(&deviceCount) != CUDA_SUCCESS ||
        deviceCount <= 0 ||
        cudaDeviceGetAttribute(
            &result->computeCapabilityMajor,
            CUDA_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            0
        ) != CUDA_SUCCESS ||
        cudaDeviceGetAttribute(
            &result->computeCapabilityMinor,
            CUDA_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            0
        ) != CUDA_SUCCESS
    ) {
        result->failureStage = OVTRTEngineLoadStageConfiguration;
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return OVTRTStatusCUDARuntimeFailure;
    }

    int fileDescriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (fileDescriptor < 0) {
        result->failureStage = OVTRTEngineLoadStageFileOpen;
        result->systemErrorCode = errno;
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return OVTRTStatusEngineArtifactFailure;
    }
    struct stat fileStatus {};
    if (
        fstat(fileDescriptor, &fileStatus) != 0 ||
        fileStatus.st_size <= 0 ||
        static_cast<uint64_t>(fileStatus.st_size) >
            static_cast<uint64_t>(
                std::numeric_limits<size_t>::max()
            ) ||
        static_cast<uint64_t>(fileStatus.st_size) >
            std::numeric_limits<uint64_t>::max() / 8U
    ) {
        result->failureStage = OVTRTEngineLoadStageFileStat;
        result->systemErrorCode = errno;
        close(fileDescriptor);
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return OVTRTStatusEngineArtifactFailure;
    }
    size_t artifactByteCount =
        static_cast<size_t>(fileStatus.st_size);
    result->artifactByteCount =
        static_cast<uint64_t>(artifactByteCount);
    void *mapped = mmap(
        nullptr,
        artifactByteCount,
        PROT_READ,
        MAP_PRIVATE,
        fileDescriptor,
        0
    );
    if (mapped == MAP_FAILED) {
        result->failureStage = OVTRTEngineLoadStageFileMapping;
        result->systemErrorCode = errno;
        close(fileDescriptor);
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return OVTRTStatusEngineArtifactFailure;
    }

    auto const *artifactBytes =
        static_cast<uint8_t const *>(mapped);
    if (
        !checksumMatches(
            artifactBytes,
            artifactByteCount,
            expectedChecksum
        )
    ) {
        result->failureStage = OVTRTEngineLoadStageChecksum;
        munmap(mapped, artifactByteCount);
        close(fileDescriptor);
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return OVTRTStatusEngineChecksumMismatch;
    }
    result->checksumVerified = 1;

    OVTRTEngine *owner = new (std::nothrow) OVTRTEngine{};
    if (owner == nullptr) {
        result->failureStage = OVTRTEngineLoadStageRuntimeCreation;
        munmap(mapped, artifactByteCount);
        close(fileDescriptor);
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return OVTRTStatusAllocationFailed;
    }
    owner->tensorRTLibrary = tensorRTLibrary;
    owner->cudaRuntimeLibrary = cudaRuntimeLibrary;
    owner->cudaMalloc = cudaMalloc;
    owner->cudaFree = cudaFree;
    owner->cudaStreamCreateWithFlags =
        cudaStreamCreateWithFlags;
    owner->cudaStreamDestroy = cudaStreamDestroy;
    owner->cudaStreamSynchronize = cudaStreamSynchronize;
    owner->cudaEventCreateWithFlags =
        cudaEventCreateWithFlags;
    owner->cudaEventDestroy = cudaEventDestroy;
    owner->cudaEventRecord = cudaEventRecord;
    owner->cudaEventSynchronize = cudaEventSynchronize;
    owner->cudaEventElapsedTime = cudaEventElapsedTime;
    owner->runtime = static_cast<nvinfer1::IRuntime *>(
        createInferRuntime(&owner->logger, NV_TENSORRT_VERSION)
    );
    if (owner->runtime == nullptr) {
        result->failureStage = OVTRTEngineLoadStageRuntimeCreation;
        delete owner;
        munmap(mapped, artifactByteCount);
        close(fileDescriptor);
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return OVTRTStatusTensorRTRuntimeFailure;
    }
    owner->engine = owner->runtime->deserializeCudaEngine(
        artifactBytes,
        artifactByteCount
    );
    munmap(mapped, artifactByteCount);
    close(fileDescriptor);
    if (owner->engine == nullptr) {
        result->failureStage = OVTRTEngineLoadStageDeserialization;
        delete owner->runtime;
        delete owner;
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return OVTRTStatusEngineDeserializationFailure;
    }
    int32_t tensorCount = owner->engine->getNbIOTensors();
    if (tensorCount <= 0) {
        result->failureStage = OVTRTEngineLoadStageTensorInspection;
        delete owner->engine;
        delete owner->runtime;
        delete owner;
        dlclose(cudaRuntimeLibrary);
        dlclose(tensorRTLibrary);
        return OVTRTStatusEngineArtifactFailure;
    }
    result->ioTensorCount = static_cast<uint32_t>(tensorCount);
    *engine = owner;
    return OVTRTStatusSuccess;
#else
    result->failureStage = OVTRTEngineLoadStageLibraryOpen;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_engine_tensor_name(
    OVTRTEngine *engine,
    uint32_t index,
    char *destination,
    uint32_t destinationCapacity,
    uint32_t *requiredCapacity
) {
    if (
        engine == nullptr ||
        requiredCapacity == nullptr
    ) {
        return OVTRTStatusInvalidArgument;
    }
#if OVTRT_HAS_ENGINE_RUNTIME
    int32_t count = engine->engine->getNbIOTensors();
    if (index >= static_cast<uint32_t>(count)) {
        return OVTRTStatusInvalidArgument;
    }
    char const *name = engine->engine->getIOTensorName(
        static_cast<int32_t>(index)
    );
    if (name == nullptr) {
        return OVTRTStatusEngineArtifactFailure;
    }
    size_t length = std::strlen(name);
    if (
        length >= static_cast<size_t>(
            std::numeric_limits<uint32_t>::max()
        )
    ) {
        return OVTRTStatusEngineArtifactFailure;
    }
    *requiredCapacity = static_cast<uint32_t>(length + 1);
    if (destination == nullptr) {
        return OVTRTStatusSuccess;
    }
    if (
        destinationCapacity < *requiredCapacity
    ) {
        return OVTRTStatusAllocationFailed;
    }
    std::memcpy(destination, name, length + 1);
    return OVTRTStatusSuccess;
#else
    (void)index;
    (void)destination;
    (void)destinationCapacity;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_engine_tensor_info(
    OVTRTEngine *engine,
    uint32_t index,
    OVTRTEngineTensorInfo *info
) {
    if (engine == nullptr || info == nullptr) {
        return OVTRTStatusInvalidArgument;
    }
    *info = OVTRTEngineTensorInfo{};
#if OVTRT_HAS_ENGINE_RUNTIME
    int32_t count = engine->engine->getNbIOTensors();
    if (index >= static_cast<uint32_t>(count)) {
        return OVTRTStatusInvalidArgument;
    }
    char const *name = engine->engine->getIOTensorName(
        static_cast<int32_t>(index)
    );
    if (name == nullptr) {
        return OVTRTStatusEngineArtifactFailure;
    }
    nvinfer1::TensorIOMode mode =
        engine->engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
        info->ioMode = OVTRTTensorIOModeInput;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
        info->ioMode = OVTRTTensorIOModeOutput;
    } else {
        info->ioMode = OVTRTTensorIOModeUnknown;
    }
    info->elementType = elementType(
        engine->engine->getTensorDataType(name)
    );
    nvinfer1::Dims dimensions =
        engine->engine->getTensorShape(name);
    info->rank = dimensions.nbDims;
    if (info->rank < 0) {
        return OVTRTStatusEngineArtifactFailure;
    }
    return OVTRTStatusSuccess;
#else
    (void)index;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_engine_tensor_dimension(
    OVTRTEngine *engine,
    uint32_t index,
    uint32_t axis,
    OVTRTShapeSelector selector,
    int64_t *dimension
) {
    if (engine == nullptr || dimension == nullptr) {
        return OVTRTStatusInvalidArgument;
    }
#if OVTRT_HAS_ENGINE_RUNTIME
    int32_t count = engine->engine->getNbIOTensors();
    if (index >= static_cast<uint32_t>(count)) {
        return OVTRTStatusInvalidArgument;
    }
    char const *name = engine->engine->getIOTensorName(
        static_cast<int32_t>(index)
    );
    if (name == nullptr) {
        return OVTRTStatusEngineArtifactFailure;
    }
    nvinfer1::Dims dimensions =
        engine->engine->getTensorShape(name);
    if (selector != OVTRTShapeSelectorDeclared) {
        if (
            engine->engine->getTensorIOMode(name) !=
                nvinfer1::TensorIOMode::kINPUT
        ) {
            return OVTRTStatusInvalidArgument;
        }
        nvinfer1::OptProfileSelector profileSelector;
        switch (selector) {
        case OVTRTShapeSelectorMinimum:
            profileSelector = nvinfer1::OptProfileSelector::kMIN;
            break;
        case OVTRTShapeSelectorOptimum:
            profileSelector = nvinfer1::OptProfileSelector::kOPT;
            break;
        case OVTRTShapeSelectorMaximum:
            profileSelector = nvinfer1::OptProfileSelector::kMAX;
            break;
        default:
            return OVTRTStatusInvalidArgument;
        }
        dimensions = engine->engine->getProfileShape(
            name,
            0,
            profileSelector
        );
    }
    if (
        dimensions.nbDims < 0 ||
        axis >= static_cast<uint32_t>(dimensions.nbDims)
    ) {
        return OVTRTStatusInvalidArgument;
    }
    *dimension = static_cast<int64_t>(dimensions.d[axis]);
    return OVTRTStatusSuccess;
#else
    (void)index;
    (void)axis;
    (void)selector;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_engine_prepare_execution(
    OVTRTEngine *engine,
    uint64_t const *outputCapacityByteCounts,
    uint32_t outputCapacityCount,
    OVTRTEngineExecutionResult *result
) {
    if (
        engine == nullptr ||
        outputCapacityByteCounts == nullptr ||
        outputCapacityCount == 0 ||
        result == nullptr
    ) {
        return OVTRTStatusInvalidArgument;
    }
    *result = OVTRTEngineExecutionResult{};
#if OVTRT_HAS_ENGINE_RUNTIME
    if (
        engine->executionContext != nullptr ||
        engine->outputs != nullptr ||
        engine->executionStream != nullptr
    ) {
        result->failureStage =
            OVTRTEngineExecutionStageConfiguration;
        populateExecutionResources(engine, result);
        return OVTRTStatusResourceBusy;
    }

    int32_t tensorCount = engine->engine->getNbIOTensors();
    uint32_t inputCount = 0;
    uint32_t outputCount = 0;
    for (int32_t index = 0; index < tensorCount; ++index) {
        char const *name = engine->engine->getIOTensorName(index);
        if (name == nullptr) {
            result->failureStage =
                OVTRTEngineExecutionStageConfiguration;
            return OVTRTStatusEngineExecutionSetupFailure;
        }
        nvinfer1::TensorIOMode mode =
            engine->engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            ++inputCount;
            engine->inputName = name;
        } else if (
            mode == nvinfer1::TensorIOMode::kOUTPUT
        ) {
            ++outputCount;
        }
    }
    if (
        inputCount != 1 ||
        outputCount == 0 ||
        outputCount > 64 ||
        outputCapacityCount != outputCount
    ) {
        engine->inputName = nullptr;
        result->failureStage =
            OVTRTEngineExecutionStageConfiguration;
        return OVTRTStatusEngineExecutionSetupFailure;
    }

    engine->executionContext =
        engine->engine->createExecutionContext(
            nvinfer1::ExecutionContextAllocationStrategy::kSTATIC
        );
    if (engine->executionContext == nullptr) {
        engine->inputName = nullptr;
        result->failureStage =
            OVTRTEngineExecutionStageContextCreation;
        return OVTRTStatusEngineExecutionSetupFailure;
    }
    if (
        engine->cudaStreamCreateWithFlags(
            &engine->executionStream,
            CUDA_STREAM_NON_BLOCKING
        ) != CUDA_SUCCESS
    ) {
        result->failureStage =
            OVTRTEngineExecutionStageStreamCreation;
        return statusAfterPreparationCleanup(
            engine,
            result,
            OVTRTStatusCUDARuntimeFailure
        );
    }
    if (
        engine->cudaEventCreateWithFlags(
            &engine->executionStart,
            CUDA_EVENT_DEFAULT
        ) != CUDA_SUCCESS ||
        engine->cudaEventCreateWithFlags(
            &engine->executionStop,
            CUDA_EVENT_DEFAULT
        ) != CUDA_SUCCESS
    ) {
        result->failureStage =
            OVTRTEngineExecutionStageEventCreation;
        return statusAfterPreparationCleanup(
            engine,
            result,
            OVTRTStatusCUDARuntimeFailure
        );
    }

    nvinfer1::Dims declaredInput =
        engine->engine->getTensorShape(engine->inputName);
    if (declaredInput.nbDims <= 0) {
        result->failureStage =
            OVTRTEngineExecutionStageShapeConfiguration;
        return statusAfterPreparationCleanup(
            engine,
            result,
            OVTRTStatusEngineExecutionSetupFailure
        );
    }
    nvinfer1::Dims maximumInput = declaredInput;
    bool hasDynamicDimension = false;
    for (
        int32_t axis = 0;
        axis < declaredInput.nbDims;
        ++axis
    ) {
        hasDynamicDimension =
            hasDynamicDimension ||
            declaredInput.d[axis] < 0;
    }
    if (hasDynamicDimension) {
        maximumInput = engine->engine->getProfileShape(
            engine->inputName,
            0,
            nvinfer1::OptProfileSelector::kMAX
        );
    }
    if (maximumInput.nbDims != declaredInput.nbDims) {
        result->failureStage =
            OVTRTEngineExecutionStageShapeConfiguration;
        return statusAfterPreparationCleanup(
            engine,
            result,
            OVTRTStatusEngineExecutionSetupFailure
        );
    }
    for (
        int32_t axis = 0;
        axis < declaredInput.nbDims;
        ++axis
    ) {
        if (declaredInput.d[axis] >= 0) {
            maximumInput.d[axis] = declaredInput.d[axis];
        } else if (maximumInput.d[axis] <= 0) {
            result->failureStage =
                OVTRTEngineExecutionStageShapeConfiguration;
            return statusAfterPreparationCleanup(
                engine,
                result,
                OVTRTStatusEngineExecutionSetupFailure
            );
        }
    }
    if (
        !engine->executionContext->setInputShape(
            engine->inputName,
            maximumInput
        )
    ) {
        result->failureStage =
            OVTRTEngineExecutionStageShapeConfiguration;
        return statusAfterPreparationCleanup(
            engine,
            result,
            OVTRTStatusEngineExecutionSetupFailure
        );
    }
    engine->maximumInputDimensions = maximumInput;

    engine->outputs =
        new (std::nothrow) ExecutionOutput[outputCount];
    if (engine->outputs == nullptr) {
        result->failureStage =
            OVTRTEngineExecutionStageOutputAllocation;
        return statusAfterPreparationCleanup(
            engine,
            result,
            OVTRTStatusAllocationFailed
        );
    }
    engine->outputCount = outputCount;
    uint32_t outputIndex = 0;
    for (int32_t index = 0; index < tensorCount; ++index) {
        char const *name = engine->engine->getIOTensorName(index);
        if (
            engine->engine->getTensorIOMode(name) !=
                nvinfer1::TensorIOMode::kOUTPUT
        ) {
            continue;
        }
        uint64_t maximumByteCount =
            outputCapacityByteCounts[outputIndex];
        if (
            maximumByteCount == 0 ||
            maximumByteCount >
                MAXIMUM_EXECUTION_DEVICE_BYTE_COUNT ||
            maximumByteCount >
                MAXIMUM_EXECUTION_DEVICE_BYTE_COUNT -
                engine->persistentDeviceAllocationByteCount
        ) {
            result->failureStage =
                OVTRTEngineExecutionStageOutputAllocation;
            return statusAfterPreparationCleanup(
                engine,
                result,
                OVTRTStatusEngineExecutionSetupFailure
            );
        }
        ExecutionOutput &output =
            engine->outputs[outputIndex];
        output.name = name;
        output.dataType =
            engine->engine->getTensorDataType(name);
        output.capacityByteCount = maximumByteCount;
        if (
            output.capacityByteCount >
                static_cast<uint64_t>(
                    std::numeric_limits<size_t>::max()
                ) ||
            engine->cudaMalloc(
                &output.deviceAddress,
                static_cast<size_t>(output.capacityByteCount)
            ) != CUDA_SUCCESS
        ) {
            result->failureStage =
                OVTRTEngineExecutionStageOutputAllocation;
            return statusAfterPreparationCleanup(
                engine,
                result,
                OVTRTStatusAllocationFailed
            );
        }
        output.allocator = new (std::nothrow)
            FixedOutputAllocator(
                output.deviceAddress,
                output.capacityByteCount
            );
        if (output.allocator == nullptr) {
            result->failureStage =
                OVTRTEngineExecutionStageOutputAllocation;
            return statusAfterPreparationCleanup(
                engine,
                result,
                OVTRTStatusAllocationFailed
            );
        }
        if (
            !engine->executionContext->setTensorAddress(
                name,
                output.deviceAddress
            ) ||
            !engine->executionContext->setOutputAllocator(
                name,
                output.allocator
            )
        ) {
            result->failureStage =
                OVTRTEngineExecutionStageTensorBinding;
            return statusAfterPreparationCleanup(
                engine,
                result,
                OVTRTStatusEngineExecutionSetupFailure
            );
        }
        engine->persistentDeviceAllocationByteCount +=
            output.capacityByteCount;
        ++outputIndex;
    }
    populateExecutionResources(engine, result);
    result->batchSize = static_cast<uint32_t>(
        maximumInput.d[0]
    );
    return OVTRTStatusSuccess;
#else
    result->failureStage =
        OVTRTEngineExecutionStageContextCreation;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_engine_execute(
    OVTRTEngine *engine,
    void const *inputDeviceAddress,
    uint64_t inputByteCount,
    int64_t const *inputDimensions,
    uint32_t inputRank,
    OVTRTEngineExecutionResult *result
) {
    if (
        engine == nullptr ||
        inputDeviceAddress == nullptr ||
        inputDimensions == nullptr ||
        inputRank == 0 ||
        result == nullptr
    ) {
        return OVTRTStatusInvalidArgument;
    }
    *result = OVTRTEngineExecutionResult{};
#if OVTRT_HAS_ENGINE_RUNTIME
    if (
        engine->executionContext == nullptr ||
        engine->executionStream == nullptr ||
        engine->outputs == nullptr ||
        engine->inputName == nullptr ||
        inputRank != static_cast<uint32_t>(
            engine->maximumInputDimensions.nbDims
        )
    ) {
        result->failureStage =
            OVTRTEngineExecutionStageConfiguration;
        populateExecutionResources(engine, result);
        return OVTRTStatusEngineExecutionFailure;
    }

    nvinfer1::Dims dimensions{};
    dimensions.nbDims = static_cast<int32_t>(inputRank);
    for (uint32_t axis = 0; axis < inputRank; ++axis) {
        if (
            inputDimensions[axis] <= 0 ||
            inputDimensions[axis] >
                engine->maximumInputDimensions.d[axis]
        ) {
            result->failureStage =
                OVTRTEngineExecutionStageShapeConfiguration;
            populateExecutionResources(engine, result);
            return OVTRTStatusInvalidArgument;
        }
        dimensions.d[axis] = inputDimensions[axis];
    }
    uint64_t inputElementCount = 0;
    uint64_t inputElementByteCount = elementByteCount(
        engine->engine->getTensorDataType(engine->inputName)
    );
    if (
        inputElementByteCount == 0 ||
        !dimensionsElementCount(
            dimensions,
            &inputElementCount
        ) ||
        inputElementCount >
            std::numeric_limits<uint64_t>::max() /
                inputElementByteCount ||
        inputByteCount !=
            inputElementCount * inputElementByteCount
    ) {
        result->failureStage =
            OVTRTEngineExecutionStageConfiguration;
        populateExecutionResources(engine, result);
        return OVTRTStatusInvalidArgument;
    }
    if (
        !engine->executionContext->setInputShape(
            engine->inputName,
            dimensions
        ) ||
        !engine->executionContext->setInputTensorAddress(
            engine->inputName,
            inputDeviceAddress
        )
    ) {
        result->failureStage =
            OVTRTEngineExecutionStageTensorBinding;
        populateExecutionResources(engine, result);
        return OVTRTStatusEngineExecutionFailure;
    }
    for (
        uint32_t index = 0;
        index < engine->outputCount;
        ++index
    ) {
        engine->outputs[index].allocator->reset();
        engine->outputs[index].byteCount = 0;
        engine->outputs[index].elementCount = 0;
        engine->outputs[index].dimensions = nvinfer1::Dims{};
    }

    if (
        engine->cudaEventRecord(
            engine->executionStart,
            engine->executionStream
        ) != CUDA_SUCCESS
    ) {
        result->failureStage =
            OVTRTEngineExecutionStageEnqueue;
        populateExecutionResources(engine, result);
        return OVTRTStatusCUDARuntimeFailure;
    }
    if (
        !engine->executionContext->enqueueV3(
            engine->executionStream
        )
    ) {
        bool capacityExceeded = false;
        for (
            uint32_t index = 0;
            index < engine->outputCount;
            ++index
        ) {
            capacityExceeded =
                capacityExceeded ||
                engine->outputs[index]
                    .allocator
                    ->capacityWasExceeded();
        }
        result->failureStage =
            OVTRTEngineExecutionStageEnqueue;
        populateExecutionResources(engine, result);
        return capacityExceeded
            ? OVTRTStatusOutputCapacityExceeded
            : OVTRTStatusEngineExecutionFailure;
    }
    if (
        engine->cudaEventRecord(
            engine->executionStop,
            engine->executionStream
        ) != CUDA_SUCCESS ||
        engine->cudaEventSynchronize(
            engine->executionStop
        ) != CUDA_SUCCESS ||
        engine->cudaEventElapsedTime(
            &result->inferenceMilliseconds,
            engine->executionStart,
            engine->executionStop
        ) != CUDA_SUCCESS
    ) {
        result->failureStage =
            OVTRTEngineExecutionStageSynchronization;
        populateExecutionResources(engine, result);
        return OVTRTStatusCUDARuntimeFailure;
    }

    uint64_t totalOutputByteCount = 0;
    for (
        uint32_t index = 0;
        index < engine->outputCount;
        ++index
    ) {
        ExecutionOutput &output = engine->outputs[index];
        if (output.allocator->capacityWasExceeded()) {
            result->failureStage =
                OVTRTEngineExecutionStageOutputInspection;
            populateExecutionResources(engine, result);
            return OVTRTStatusOutputCapacityExceeded;
        }
        nvinfer1::Dims outputDimensions =
            engine->executionContext->getTensorShape(
                output.name
            );
        if (
            output.allocator->shapeWasNotified()
        ) {
            outputDimensions =
                output.allocator->finalDimensions();
        }
        uint64_t outputElementCount = 0;
        uint64_t outputElementByteCount =
            elementByteCount(output.dataType);
        if (
            outputElementByteCount == 0 ||
            !dimensionsElementCount(
                outputDimensions,
                &outputElementCount
            ) ||
            outputElementCount >
                std::numeric_limits<uint64_t>::max() /
                    outputElementByteCount
        ) {
            result->failureStage =
                OVTRTEngineExecutionStageOutputInspection;
            populateExecutionResources(engine, result);
            return OVTRTStatusEngineExecutionFailure;
        }
        uint64_t outputByteCount =
            outputElementCount * outputElementByteCount;
        if (
            outputByteCount > output.capacityByteCount ||
            outputByteCount >
                std::numeric_limits<uint64_t>::max() -
                    totalOutputByteCount
        ) {
            result->failureStage =
                OVTRTEngineExecutionStageOutputInspection;
            populateExecutionResources(engine, result);
            return OVTRTStatusOutputCapacityExceeded;
        }
        output.dimensions = outputDimensions;
        output.elementCount = outputElementCount;
        output.byteCount = outputByteCount;
        totalOutputByteCount += outputByteCount;
    }
    ++engine->submissionCount;
    result->batchSize = static_cast<uint32_t>(
        dimensions.d[0]
    );
    result->inputByteCount = inputByteCount;
    result->outputByteCount = totalOutputByteCount;
    result->frameDeviceAllocationCount = 0;
    populateExecutionResources(engine, result);
    return OVTRTStatusSuccess;
#else
    (void)inputByteCount;
    (void)inputRank;
    result->failureStage =
        OVTRTEngineExecutionStageConfiguration;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_engine_output(
    OVTRTEngine *engine,
    uint32_t outputIndex,
    OVTRTEngineOutputView *view
) {
    if (engine == nullptr || view == nullptr) {
        return OVTRTStatusInvalidArgument;
    }
    *view = OVTRTEngineOutputView{};
#if OVTRT_HAS_ENGINE_RUNTIME
    if (
        engine->submissionCount == 0 ||
        engine->outputs == nullptr ||
        outputIndex >= engine->outputCount
    ) {
        return OVTRTStatusResourceBusy;
    }
    ExecutionOutput const &output =
        engine->outputs[outputIndex];
    if (
        output.deviceAddress == nullptr ||
        output.dimensions.nbDims <= 0
    ) {
        return OVTRTStatusEngineExecutionFailure;
    }
    view->deviceAddress = output.deviceAddress;
    view->byteCount = output.byteCount;
    view->elementCount = output.elementCount;
    view->elementType = elementType(output.dataType);
    view->rank = output.dimensions.nbDims;
    return OVTRTStatusSuccess;
#else
    (void)outputIndex;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_engine_output_dimension(
    OVTRTEngine *engine,
    uint32_t outputIndex,
    uint32_t axis,
    int64_t *dimension
) {
    if (engine == nullptr || dimension == nullptr) {
        return OVTRTStatusInvalidArgument;
    }
#if OVTRT_HAS_ENGINE_RUNTIME
    if (
        engine->submissionCount == 0 ||
        engine->outputs == nullptr ||
        outputIndex >= engine->outputCount ||
        axis >= static_cast<uint32_t>(
            engine->outputs[outputIndex].dimensions.nbDims
        )
    ) {
        return OVTRTStatusInvalidArgument;
    }
    *dimension =
        engine->outputs[outputIndex].dimensions.d[axis];
    return OVTRTStatusSuccess;
#else
    (void)outputIndex;
    (void)axis;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_engine_release_execution(
    OVTRTEngine *engine,
    OVTRTEngineExecutionResult *result
) {
    if (engine == nullptr || result == nullptr) {
        return OVTRTStatusInvalidArgument;
    }
    *result = OVTRTEngineExecutionResult{};
#if OVTRT_HAS_ENGINE_RUNTIME
    if (
        engine->executionContext == nullptr &&
        engine->outputs == nullptr &&
        engine->executionStream == nullptr &&
        engine->executionStart == nullptr &&
        engine->executionStop == nullptr
    ) {
        result->failureStage =
            OVTRTEngineExecutionStageConfiguration;
        return OVTRTStatusResourceBusy;
    }
    return releaseExecutionResources(engine, result);
#else
    result->failureStage =
        OVTRTEngineExecutionStageCleanup;
    return OVTRTStatusUnavailable;
#endif
}

void ovtrt_engine_destroy(OVTRTEngine *engine) {
    if (engine == nullptr) {
        return;
    }
#if OVTRT_HAS_ENGINE_RUNTIME
    if (
        engine->executionContext != nullptr ||
        engine->outputs != nullptr ||
        engine->executionStream != nullptr ||
        engine->executionStart != nullptr ||
        engine->executionStop != nullptr
    ) {
        OVTRTEngineExecutionResult cleanup{};
        releaseExecutionResources(engine, &cleanup);
    }
    delete engine->engine;
    delete engine->runtime;
    dlclose(engine->cudaRuntimeLibrary);
    dlclose(engine->tensorRTLibrary);
#endif
    delete engine;
}
