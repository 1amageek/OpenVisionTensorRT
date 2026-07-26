#include "CTensorRTShim.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

#if defined(__linux__) && __has_include(<cuda.h>) && \
    __has_include(<cuda_runtime_api.h>) && __has_include(<nvrtc.h>)
#define OVTRT_HAS_RG10_PREPROCESSING 1
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#else
#define OVTRT_HAS_RG10_PREPROCESSING 0
#endif

namespace {

constexpr uint64_t MAXIMUM_INPUT_BYTE_COUNT =
    512ULL * 1024ULL * 1024ULL;
constexpr uint64_t MAXIMUM_OUTPUT_ELEMENT_COUNT =
    3ULL * 4096ULL * 4096ULL;

bool isFinite(float value) noexcept {
    return std::isfinite(static_cast<double>(value));
}

bool isUnitValue(float value) noexcept {
    return isFinite(value) && value >= 0.0F && value <= 1.0F;
}

bool isValidConfiguration(
    OVTRTRG10PreprocessingConfiguration const &configuration
) noexcept {
    if (
        configuration.sourceWidth < 2 ||
        configuration.sourceHeight < 2 ||
        configuration.outputWidth == 0 ||
        configuration.outputHeight == 0
    ) {
        return false;
    }
    uint64_t minimumRowByteCount =
        static_cast<uint64_t>(configuration.sourceWidth) * 2ULL;
    if (
        configuration.sourceBytesPerRow < minimumRowByteCount ||
        configuration.sourceByteCount == 0 ||
        configuration.sourceByteCount > MAXIMUM_INPUT_BYTE_COUNT
    ) {
        return false;
    }
    uint64_t minimumInputByteCount =
        static_cast<uint64_t>(configuration.sourceBytesPerRow) *
        static_cast<uint64_t>(configuration.sourceHeight);
    if (configuration.sourceByteCount < minimumInputByteCount) {
        return false;
    }
    uint64_t outputElementCount =
        static_cast<uint64_t>(configuration.outputWidth) *
        static_cast<uint64_t>(configuration.outputHeight) *
        3ULL;
    if (
        outputElementCount == 0 ||
        outputElementCount > MAXIMUM_OUTPUT_ELEMENT_COUNT
    ) {
        return false;
    }
    if (
        configuration.resizePolicy <
            OVTRTRG10ResizePolicyScaleFill ||
        configuration.resizePolicy >
            OVTRTRG10ResizePolicyCenterCrop ||
        configuration.tensorLayout < OVTRTTensorLayoutNCHW ||
        configuration.tensorLayout > OVTRTTensorLayoutNHWC ||
        configuration.channelOrder <
            OVTRTTensorChannelOrderRGB ||
        configuration.channelOrder >
            OVTRTTensorChannelOrderBGR ||
        configuration.applySRGBTransfer > 1
    ) {
        return false;
    }

    float blackLevels[] = {
        configuration.blackLevelR,
        configuration.blackLevelGreenR,
        configuration.blackLevelGreenB,
        configuration.blackLevelB
    };
    float gains[] = {
        configuration.gainR,
        configuration.gainGreenR,
        configuration.gainGreenB,
        configuration.gainB
    };
    for (size_t index = 0; index < 4; ++index) {
        if (
            !isFinite(blackLevels[index]) ||
            blackLevels[index] < 0.0F ||
            !isFinite(gains[index]) ||
            gains[index] <= 0.0F
        ) {
            return false;
        }
    }
    if (
        !isFinite(configuration.whiteLevel) ||
        configuration.whiteLevel <= 0.0F ||
        configuration.whiteLevel > 1023.0F
    ) {
        return false;
    }
    for (float blackLevel : blackLevels) {
        if (blackLevel >= configuration.whiteLevel) {
            return false;
        }
    }
    float matrix[] = {
        configuration.colorMatrix00,
        configuration.colorMatrix01,
        configuration.colorMatrix02,
        configuration.colorMatrix10,
        configuration.colorMatrix11,
        configuration.colorMatrix12,
        configuration.colorMatrix20,
        configuration.colorMatrix21,
        configuration.colorMatrix22
    };
    for (float value : matrix) {
        if (!isFinite(value)) {
            return false;
        }
    }
    if (
        !isUnitValue(configuration.letterboxR) ||
        !isUnitValue(configuration.letterboxG) ||
        !isUnitValue(configuration.letterboxB) ||
        !isFinite(configuration.normalizationScale) ||
        !isFinite(configuration.normalizationBias)
    ) {
        return false;
    }
    return true;
}

void resetResult(
    OVTRTRG10PreprocessingResult *result,
    OVTRTRG10PreprocessingConfiguration const *configuration
) noexcept {
    *result = OVTRTRG10PreprocessingResult{};
    if (configuration == nullptr) {
        return;
    }
    result->inputByteCount = configuration->sourceByteCount;
    result->outputElementCount =
        static_cast<uint64_t>(configuration->outputWidth) *
        static_cast<uint64_t>(configuration->outputHeight) *
        3ULL;
}

}  // namespace

#if OVTRT_HAS_RG10_PREPROCESSING
namespace {

template <typename Function>
Function loadFunction(
    void *library,
    char const *name
) noexcept {
    // POSIX guarantees that the dlsym address can be converted to the
    // requested function pointer. Every function remains scoped to its
    // corresponding dlopen owner and is never invoked after dlclose.
    return reinterpret_cast<Function>(dlsym(library, name));
}

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

using CUDAMallocFunction = cudaError_t (*)(void **, size_t);
using CUDAFreeFunction = cudaError_t (*)(void *);
using CUDAStreamCreateWithFlagsFunction =
    cudaError_t (*)(cudaStream_t *, unsigned int);
using CUDAStreamDestroyFunction = cudaError_t (*)(cudaStream_t);
using CUDAStreamSynchronizeFunction = cudaError_t (*)(cudaStream_t);
using CUDAEventCreateWithFlagsFunction =
    cudaError_t (*)(cudaEvent_t *, unsigned int);
using CUDAEventDestroyFunction = cudaError_t (*)(cudaEvent_t);
using CUDAEventRecordFunction =
    cudaError_t (*)(cudaEvent_t, cudaStream_t);
using CUDAEventSynchronizeFunction = cudaError_t (*)(cudaEvent_t);
using CUDAEventElapsedTimeFunction =
    cudaError_t (*)(float *, cudaEvent_t, cudaEvent_t);
using CUDAMemcpyFunction =
    cudaError_t (*)(void *, void const *, size_t, cudaMemcpyKind);

using CUInitFunction = CUresult (*)(unsigned int);
using CUDeviceGetFunction = CUresult (*)(CUdevice *, int);
using CUDeviceGetAttributeFunction =
    CUresult (*)(int *, CUdevice_attribute, CUdevice);
using CUModuleLoadDataFunction =
    CUresult (*)(CUmodule *, void const *);
using CUModuleUnloadFunction = CUresult (*)(CUmodule);
using CUModuleGetFunctionFunction =
    CUresult (*)(CUfunction *, CUmodule, char const *);
using CULaunchKernelFunction = CUresult (*)(
    CUfunction,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    CUstream,
    void **,
    void **
);

using NVRTCCreateProgramFunction = nvrtcResult (*)(
    nvrtcProgram *,
    char const *,
    char const *,
    int,
    char const *const *,
    char const *const *
);
using NVRTCCompileProgramFunction =
    nvrtcResult (*)(nvrtcProgram, int, char const *const *);
using NVRTCGetPTXSizeFunction =
    nvrtcResult (*)(nvrtcProgram, size_t *);
using NVRTCGetPTXFunction =
    nvrtcResult (*)(nvrtcProgram, char *);
using NVRTCGetProgramLogSizeFunction =
    nvrtcResult (*)(nvrtcProgram, size_t *);
using NVRTCGetProgramLogFunction =
    nvrtcResult (*)(nvrtcProgram, char *);
using NVRTCDestroyProgramFunction =
    nvrtcResult (*)(nvrtcProgram *);

struct RG10RuntimeAPI {
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
    CUDAMemcpyFunction cudaMemcpy{nullptr};

    CUInitFunction cuInit{nullptr};
    CUDeviceGetFunction cuDeviceGet{nullptr};
    CUDeviceGetAttributeFunction cuDeviceGetAttribute{nullptr};
    CUModuleLoadDataFunction cuModuleLoadData{nullptr};
    CUModuleUnloadFunction cuModuleUnload{nullptr};
    CUModuleGetFunctionFunction cuModuleGetFunction{nullptr};
    CULaunchKernelFunction cuLaunchKernel{nullptr};

    NVRTCCreateProgramFunction nvrtcCreateProgram{nullptr};
    NVRTCCompileProgramFunction nvrtcCompileProgram{nullptr};
    NVRTCGetPTXSizeFunction nvrtcGetPTXSize{nullptr};
    NVRTCGetPTXFunction nvrtcGetPTX{nullptr};
    NVRTCGetProgramLogSizeFunction nvrtcGetProgramLogSize{nullptr};
    NVRTCGetProgramLogFunction nvrtcGetProgramLog{nullptr};
    NVRTCDestroyProgramFunction nvrtcDestroyProgram{nullptr};

    bool isComplete() const noexcept {
        return
            cudaMalloc != nullptr &&
            cudaFree != nullptr &&
            cudaStreamCreateWithFlags != nullptr &&
            cudaStreamDestroy != nullptr &&
            cudaStreamSynchronize != nullptr &&
            cudaEventCreateWithFlags != nullptr &&
            cudaEventDestroy != nullptr &&
            cudaEventRecord != nullptr &&
            cudaEventSynchronize != nullptr &&
            cudaEventElapsedTime != nullptr &&
            cudaMemcpy != nullptr &&
            cuInit != nullptr &&
            cuDeviceGet != nullptr &&
            cuDeviceGetAttribute != nullptr &&
            cuModuleLoadData != nullptr &&
            cuModuleUnload != nullptr &&
            cuModuleGetFunction != nullptr &&
            cuLaunchKernel != nullptr &&
            nvrtcCreateProgram != nullptr &&
            nvrtcCompileProgram != nullptr &&
            nvrtcGetPTXSize != nullptr &&
            nvrtcGetPTX != nullptr &&
            nvrtcGetProgramLogSize != nullptr &&
            nvrtcGetProgramLog != nullptr &&
            nvrtcDestroyProgram != nullptr;
    }
};

constexpr size_t KERNEL_CONFIGURATION_FLOAT_COUNT = 23;

char const *RG10_KERNEL_SOURCE = R"cuda(
extern "C" {

__device__ float ovtrt_clamp(float value, float minimum, float maximum) {
    return fminf(fmaxf(value, minimum), maximum);
}

__device__ int ovtrt_reflect(int coordinate, int limit) {
    while (coordinate < 0 || coordinate >= limit) {
        coordinate = coordinate < 0
            ? -coordinate
            : 2 * limit - 2 - coordinate;
    }
    return coordinate;
}

__device__ float ovtrt_raw(
    unsigned char const *source,
    unsigned int width,
    unsigned int height,
    unsigned int bytesPerRow,
    int x,
    int y,
    float const *configuration
) {
    x = ovtrt_reflect(x, static_cast<int>(width));
    y = ovtrt_reflect(y, static_cast<int>(height));
    unsigned long long offset =
        static_cast<unsigned long long>(y) * bytesPerRow +
        static_cast<unsigned long long>(x) * 2ULL;
    unsigned int value =
        static_cast<unsigned int>(source[offset]) |
        (static_cast<unsigned int>(source[offset + 1]) << 8U);
    value &= 1023U;
    unsigned int site =
        ((static_cast<unsigned int>(y) & 1U) << 1U) |
        (static_cast<unsigned int>(x) & 1U);
    float blackLevel = configuration[site];
    float whiteLevel = configuration[4];
    float gain = configuration[5 + site];
    return ovtrt_clamp(
        (static_cast<float>(value) - blackLevel) /
            (whiteLevel - blackLevel) * gain,
        0.0F,
        1.0F
    );
}

__device__ void ovtrt_rgb_at(
    unsigned char const *source,
    unsigned int width,
    unsigned int height,
    unsigned int bytesPerRow,
    int x,
    int y,
    float const *configuration,
    float &red,
    float &green,
    float &blue
) {
    bool evenX = (x & 1) == 0;
    bool evenY = (y & 1) == 0;
    float center = ovtrt_raw(
        source,
        width,
        height,
        bytesPerRow,
        x,
        y,
        configuration
    );
    if (evenX && evenY) {
        red = center;
        green = 0.25F * (
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x, y - 1, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x, y + 1, configuration)
        );
        blue = 0.25F * (
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y - 1, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y - 1, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y + 1, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y + 1, configuration)
        );
        return;
    }
    if (!evenX && !evenY) {
        blue = center;
        green = 0.25F * (
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x, y - 1, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x, y + 1, configuration)
        );
        red = 0.25F * (
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y - 1, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y - 1, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y + 1, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y + 1, configuration)
        );
        return;
    }
    green = center;
    if (!evenX && evenY) {
        red = 0.5F * (
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y, configuration)
        );
        blue = 0.5F * (
            ovtrt_raw(source, width, height, bytesPerRow, x, y - 1, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x, y + 1, configuration)
        );
        return;
    }
    red = 0.5F * (
        ovtrt_raw(source, width, height, bytesPerRow, x, y - 1, configuration) +
        ovtrt_raw(source, width, height, bytesPerRow, x, y + 1, configuration)
    );
    blue = 0.5F * (
        ovtrt_raw(source, width, height, bytesPerRow, x - 1, y, configuration) +
        ovtrt_raw(source, width, height, bytesPerRow, x + 1, y, configuration)
    );
}

__device__ void ovtrt_oriented_source(
    float orientedX,
    float orientedY,
    unsigned int sourceWidth,
    unsigned int sourceHeight,
    unsigned int orientation,
    float &sourceX,
    float &sourceY
) {
    float maximumX = static_cast<float>(sourceWidth - 1U);
    float maximumY = static_cast<float>(sourceHeight - 1U);
    if (orientation == 1U) {
        sourceX = orientedX;
        sourceY = orientedY;
    } else if (orientation == 2U) {
        sourceX = maximumX - orientedX;
        sourceY = orientedY;
    } else if (orientation == 3U) {
        sourceX = maximumX - orientedX;
        sourceY = maximumY - orientedY;
    } else if (orientation == 4U) {
        sourceX = orientedX;
        sourceY = maximumY - orientedY;
    } else if (orientation == 5U) {
        sourceX = orientedY;
        sourceY = orientedX;
    } else if (orientation == 6U) {
        sourceX = orientedY;
        sourceY = maximumY - orientedX;
    } else if (orientation == 7U) {
        sourceX = maximumX - orientedY;
        sourceY = maximumY - orientedX;
    } else {
        sourceX = maximumX - orientedY;
        sourceY = orientedX;
    }
}

__device__ float ovtrt_srgb(float linear) {
    linear = ovtrt_clamp(linear, 0.0F, 1.0F);
    if (linear <= 0.0031308F) {
        return linear * 12.92F;
    }
    return 1.055F * powf(linear, 1.0F / 2.4F) - 0.055F;
}

__global__ void ovtrt_rg10_preprocess(
    unsigned char const *source,
    float *destination,
    float const *configuration,
    unsigned int sourceWidth,
    unsigned int sourceHeight,
    unsigned int sourceBytesPerRow,
    unsigned int outputWidth,
    unsigned int outputHeight,
    unsigned int orientation,
    unsigned int resizePolicy,
    unsigned int tensorLayout,
    unsigned int channelOrder,
    unsigned int applySRGBTransfer
) {
    unsigned int outputX =
        blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int outputY =
        blockIdx.y * blockDim.y + threadIdx.y;
    if (outputX >= outputWidth || outputY >= outputHeight) {
        return;
    }

    bool swapsDimensions = orientation >= 5U;
    float orientedWidth = static_cast<float>(
        swapsDimensions ? sourceHeight : sourceWidth
    );
    float orientedHeight = static_cast<float>(
        swapsDimensions ? sourceWidth : sourceHeight
    );
    float scaleX =
        static_cast<float>(outputWidth) / orientedWidth;
    float scaleY =
        static_cast<float>(outputHeight) / orientedHeight;
    float sampleX = 0.0F;
    float sampleY = 0.0F;
    bool isLetterbox = false;

    if (resizePolicy == 0U) {
        sampleX =
            (static_cast<float>(outputX) + 0.5F) / scaleX -
            0.5F;
        sampleY =
            (static_cast<float>(outputY) + 0.5F) / scaleY -
            0.5F;
    } else {
        float scale = resizePolicy == 1U
            ? fminf(scaleX, scaleY)
            : fmaxf(scaleX, scaleY);
        float offsetX =
            (static_cast<float>(outputWidth) -
                orientedWidth * scale) * 0.5F;
        float offsetY =
            (static_cast<float>(outputHeight) -
                orientedHeight * scale) * 0.5F;
        float centerX = static_cast<float>(outputX) + 0.5F;
        float centerY = static_cast<float>(outputY) + 0.5F;
        isLetterbox =
            resizePolicy == 1U &&
            (
                centerX < offsetX ||
                centerX >=
                    static_cast<float>(outputWidth) - offsetX ||
                centerY < offsetY ||
                centerY >=
                    static_cast<float>(outputHeight) - offsetY
            );
        sampleX = (centerX - offsetX) / scale - 0.5F;
        sampleY = (centerY - offsetY) / scale - 0.5F;
    }

    float red = configuration[18];
    float green = configuration[19];
    float blue = configuration[20];
    if (!isLetterbox) {
        float sourceX = 0.0F;
        float sourceY = 0.0F;
        ovtrt_oriented_source(
            sampleX,
            sampleY,
            sourceWidth,
            sourceHeight,
            orientation,
            sourceX,
            sourceY
        );
        sourceX = ovtrt_clamp(
            sourceX,
            0.0F,
            static_cast<float>(sourceWidth - 1U)
        );
        sourceY = ovtrt_clamp(
            sourceY,
            0.0F,
            static_cast<float>(sourceHeight - 1U)
        );
        int x0 = static_cast<int>(floorf(sourceX));
        int y0 = static_cast<int>(floorf(sourceY));
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fractionX = sourceX - static_cast<float>(x0);
        float fractionY = sourceY - static_cast<float>(y0);
        float samples[12];
        ovtrt_rgb_at(source, sourceWidth, sourceHeight, sourceBytesPerRow, x0, y0, configuration, samples[0], samples[1], samples[2]);
        ovtrt_rgb_at(source, sourceWidth, sourceHeight, sourceBytesPerRow, x1, y0, configuration, samples[3], samples[4], samples[5]);
        ovtrt_rgb_at(source, sourceWidth, sourceHeight, sourceBytesPerRow, x0, y1, configuration, samples[6], samples[7], samples[8]);
        ovtrt_rgb_at(source, sourceWidth, sourceHeight, sourceBytesPerRow, x1, y1, configuration, samples[9], samples[10], samples[11]);
        float topRed = samples[0] + (samples[3] - samples[0]) * fractionX;
        float topGreen = samples[1] + (samples[4] - samples[1]) * fractionX;
        float topBlue = samples[2] + (samples[5] - samples[2]) * fractionX;
        float bottomRed = samples[6] + (samples[9] - samples[6]) * fractionX;
        float bottomGreen = samples[7] + (samples[10] - samples[7]) * fractionX;
        float bottomBlue = samples[8] + (samples[11] - samples[8]) * fractionX;
        float cameraRed =
            topRed + (bottomRed - topRed) * fractionY;
        float cameraGreen =
            topGreen + (bottomGreen - topGreen) * fractionY;
        float cameraBlue =
            topBlue + (bottomBlue - topBlue) * fractionY;
        red =
            configuration[9] * cameraRed +
            configuration[10] * cameraGreen +
            configuration[11] * cameraBlue;
        green =
            configuration[12] * cameraRed +
            configuration[13] * cameraGreen +
            configuration[14] * cameraBlue;
        blue =
            configuration[15] * cameraRed +
            configuration[16] * cameraGreen +
            configuration[17] * cameraBlue;
        red = ovtrt_clamp(red, 0.0F, 1.0F);
        green = ovtrt_clamp(green, 0.0F, 1.0F);
        blue = ovtrt_clamp(blue, 0.0F, 1.0F);
        if (applySRGBTransfer != 0U) {
            red = ovtrt_srgb(red);
            green = ovtrt_srgb(green);
            blue = ovtrt_srgb(blue);
        }
    }

    red = red * configuration[21] + configuration[22];
    green = green * configuration[21] + configuration[22];
    blue = blue * configuration[21] + configuration[22];
    float channels[3];
    if (channelOrder == 0U) {
        channels[0] = red;
        channels[1] = green;
        channels[2] = blue;
    } else {
        channels[0] = blue;
        channels[1] = green;
        channels[2] = red;
    }
    unsigned long long pixelIndex =
        static_cast<unsigned long long>(outputY) * outputWidth +
        outputX;
    unsigned long long pixelCount =
        static_cast<unsigned long long>(outputWidth) * outputHeight;
    for (unsigned int channel = 0; channel < 3U; ++channel) {
        unsigned long long outputIndex = tensorLayout == 0U
            ? static_cast<unsigned long long>(channel) *
                pixelCount + pixelIndex
            : pixelIndex * 3ULL + channel;
        destination[outputIndex] = channels[channel];
    }
}

}  // extern "C"
)cuda";

void recordCleanupCUDAFailure(
    OVTRTRG10PreprocessingResult *result,
    cudaError_t error,
    OVTRTRG10PreprocessingStage stage
) noexcept {
    if (
        error != cudaSuccess &&
        result->cleanupFailureStage ==
            OVTRTRG10PreprocessingStageNone
    ) {
        result->cleanupCUDAErrorCode = static_cast<int32_t>(error);
        result->cleanupFailureStage = stage;
    }
}

void recordCleanupDriverFailure(
    OVTRTRG10PreprocessingResult *result,
    CUresult error,
    OVTRTRG10PreprocessingStage stage
) noexcept {
    if (
        error != CUDA_SUCCESS &&
        result->cleanupFailureStage ==
            OVTRTRG10PreprocessingStageNone
    ) {
        result->cleanupCUDADriverErrorCode =
            static_cast<int32_t>(error);
        result->cleanupFailureStage = stage;
    }
}

void recordCleanupNVRTCFailure(
    OVTRTRG10PreprocessingResult *result,
    nvrtcResult error,
    OVTRTRG10PreprocessingStage stage
) noexcept {
    if (
        error != NVRTC_SUCCESS &&
        result->cleanupFailureStage ==
            OVTRTRG10PreprocessingStageNone
    ) {
        result->cleanupNVRTCErrorCode =
            static_cast<int32_t>(error);
        result->cleanupFailureStage = stage;
    }
}

void recordCleanupDynamicLoaderFailure(
    OVTRTRG10PreprocessingResult *result,
    int error,
    OVTRTRG10PreprocessingStage stage
) noexcept {
    if (
        error != 0 &&
        result->cleanupFailureStage ==
            OVTRTRG10PreprocessingStageNone
    ) {
        result->cleanupDynamicLoaderErrorCode =
            static_cast<int32_t>(error);
        result->cleanupFailureStage = stage;
    }
}

}  // namespace

struct OVTRTRG10Preprocessor {
    // This allocation is the exactly-once owner of every CUDA/NVRTC resource
    // below. cleanupOwner clears a field only after the corresponding release
    // succeeds, so a failed destruction can retry from the remaining fields.
    // Frame source pointers are never stored: the one H2D copy completes
    // synchronously inside submit before the scoped host borrow can end.
    // deviceOutput is exposed only as a non-owning view whose Swift lease
    // retains this owner and excludes overwrite or destruction.
    OVTRTRG10PreprocessingConfiguration configuration{};
    RG10RuntimeAPI api{};
    void *cudaRuntimeLibrary{nullptr};
    void *cudaDriverLibrary{nullptr};
    void *nvrtcLibrary{nullptr};
    cudaStream_t stream{nullptr};
    cudaEvent_t startEvent{nullptr};
    cudaEvent_t outputReadyEvent{nullptr};
    void *deviceInput{nullptr};
    void *deviceOutput{nullptr};
    void *deviceConfiguration{nullptr};
    CUmodule module{nullptr};
    CUfunction kernel{nullptr};
    nvrtcProgram program{nullptr};
    bool inFlight{false};
#if defined(OVTRT_ENABLE_TEST_HOOKS)
    bool failNextCleanupSynchronization{false};
#endif
};

namespace {

bool loadRuntimeAPI(OVTRTRG10Preprocessor *owner) noexcept {
    auto &api = owner->api;
    api.cudaMalloc = loadFunction<CUDAMallocFunction>(
        owner->cudaRuntimeLibrary,
        "cudaMalloc"
    );
    api.cudaFree = loadFunction<CUDAFreeFunction>(
        owner->cudaRuntimeLibrary,
        "cudaFree"
    );
    api.cudaStreamCreateWithFlags =
        loadFunction<CUDAStreamCreateWithFlagsFunction>(
            owner->cudaRuntimeLibrary,
            "cudaStreamCreateWithFlags"
        );
    api.cudaStreamDestroy =
        loadFunction<CUDAStreamDestroyFunction>(
            owner->cudaRuntimeLibrary,
            "cudaStreamDestroy"
        );
    api.cudaStreamSynchronize =
        loadFunction<CUDAStreamSynchronizeFunction>(
            owner->cudaRuntimeLibrary,
            "cudaStreamSynchronize"
        );
    api.cudaEventCreateWithFlags =
        loadFunction<CUDAEventCreateWithFlagsFunction>(
            owner->cudaRuntimeLibrary,
            "cudaEventCreateWithFlags"
        );
    api.cudaEventDestroy =
        loadFunction<CUDAEventDestroyFunction>(
            owner->cudaRuntimeLibrary,
            "cudaEventDestroy"
        );
    api.cudaEventRecord = loadFunction<CUDAEventRecordFunction>(
        owner->cudaRuntimeLibrary,
        "cudaEventRecord"
    );
    api.cudaEventSynchronize =
        loadFunction<CUDAEventSynchronizeFunction>(
            owner->cudaRuntimeLibrary,
            "cudaEventSynchronize"
        );
    api.cudaEventElapsedTime =
        loadFunction<CUDAEventElapsedTimeFunction>(
            owner->cudaRuntimeLibrary,
            "cudaEventElapsedTime"
        );
    api.cudaMemcpy = loadFunction<CUDAMemcpyFunction>(
        owner->cudaRuntimeLibrary,
        "cudaMemcpy"
    );

    api.cuInit = loadFunction<CUInitFunction>(
        owner->cudaDriverLibrary,
        "cuInit"
    );
    api.cuDeviceGet = loadFunction<CUDeviceGetFunction>(
        owner->cudaDriverLibrary,
        "cuDeviceGet"
    );
    api.cuDeviceGetAttribute =
        loadFunction<CUDeviceGetAttributeFunction>(
            owner->cudaDriverLibrary,
            "cuDeviceGetAttribute"
        );
    api.cuModuleLoadData = loadFunction<CUModuleLoadDataFunction>(
        owner->cudaDriverLibrary,
        "cuModuleLoadData"
    );
    api.cuModuleUnload = loadFunction<CUModuleUnloadFunction>(
        owner->cudaDriverLibrary,
        "cuModuleUnload"
    );
    api.cuModuleGetFunction =
        loadFunction<CUModuleGetFunctionFunction>(
            owner->cudaDriverLibrary,
            "cuModuleGetFunction"
        );
    api.cuLaunchKernel = loadFunction<CULaunchKernelFunction>(
        owner->cudaDriverLibrary,
        "cuLaunchKernel"
    );

    api.nvrtcCreateProgram =
        loadFunction<NVRTCCreateProgramFunction>(
            owner->nvrtcLibrary,
            "nvrtcCreateProgram"
        );
    api.nvrtcCompileProgram =
        loadFunction<NVRTCCompileProgramFunction>(
            owner->nvrtcLibrary,
            "nvrtcCompileProgram"
        );
    api.nvrtcGetPTXSize =
        loadFunction<NVRTCGetPTXSizeFunction>(
            owner->nvrtcLibrary,
            "nvrtcGetPTXSize"
        );
    api.nvrtcGetPTX = loadFunction<NVRTCGetPTXFunction>(
        owner->nvrtcLibrary,
        "nvrtcGetPTX"
    );
    api.nvrtcGetProgramLogSize =
        loadFunction<NVRTCGetProgramLogSizeFunction>(
            owner->nvrtcLibrary,
            "nvrtcGetProgramLogSize"
        );
    api.nvrtcGetProgramLog =
        loadFunction<NVRTCGetProgramLogFunction>(
            owner->nvrtcLibrary,
            "nvrtcGetProgramLog"
        );
    api.nvrtcDestroyProgram =
        loadFunction<NVRTCDestroyProgramFunction>(
            owner->nvrtcLibrary,
            "nvrtcDestroyProgram"
        );
    return api.isComplete();
}

OVTRTStatus cleanupOwner(
    OVTRTRG10Preprocessor *owner,
    OVTRTRG10PreprocessingResult *result
) noexcept {
    if (owner->stream != nullptr) {
#if defined(OVTRT_ENABLE_TEST_HOOKS)
        if (owner->failNextCleanupSynchronization) {
            owner->failNextCleanupSynchronization = false;
            recordCleanupCUDAFailure(
                result,
                cudaErrorUnknown,
                OVTRTRG10PreprocessingStageStreamSynchronization
            );
            return OVTRTStatusPreprocessingFailure;
        }
#endif
        cudaError_t synchronizationStatus =
            owner->api.cudaStreamSynchronize(owner->stream);
        recordCleanupCUDAFailure(
            result,
            synchronizationStatus,
            OVTRTRG10PreprocessingStageStreamSynchronization
        );
        if (synchronizationStatus != cudaSuccess) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->inFlight = false;
    }
    if (owner->module != nullptr) {
        CUresult unloadStatus =
            owner->api.cuModuleUnload(owner->module);
        recordCleanupDriverFailure(
            result,
            unloadStatus,
            OVTRTRG10PreprocessingStageModuleUnload
        );
        if (unloadStatus != CUDA_SUCCESS) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->module = nullptr;
        owner->kernel = nullptr;
    }
    if (owner->deviceConfiguration != nullptr) {
        cudaError_t freeStatus =
            owner->api.cudaFree(owner->deviceConfiguration);
        recordCleanupCUDAFailure(
            result,
            freeStatus,
            OVTRTRG10PreprocessingStageDeviceDeallocation
        );
        if (freeStatus != cudaSuccess) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->deviceConfiguration = nullptr;
    }
    if (owner->deviceOutput != nullptr) {
        cudaError_t freeStatus =
            owner->api.cudaFree(owner->deviceOutput);
        recordCleanupCUDAFailure(
            result,
            freeStatus,
            OVTRTRG10PreprocessingStageDeviceDeallocation
        );
        if (freeStatus != cudaSuccess) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->deviceOutput = nullptr;
    }
    if (owner->deviceInput != nullptr) {
        cudaError_t freeStatus =
            owner->api.cudaFree(owner->deviceInput);
        recordCleanupCUDAFailure(
            result,
            freeStatus,
            OVTRTRG10PreprocessingStageDeviceDeallocation
        );
        if (freeStatus != cudaSuccess) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->deviceInput = nullptr;
    }
    if (owner->outputReadyEvent != nullptr) {
        cudaError_t destructionStatus =
            owner->api.cudaEventDestroy(owner->outputReadyEvent);
        recordCleanupCUDAFailure(
            result,
            destructionStatus,
            OVTRTRG10PreprocessingStageEventDestruction
        );
        if (destructionStatus != cudaSuccess) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->outputReadyEvent = nullptr;
    }
    if (owner->startEvent != nullptr) {
        cudaError_t destructionStatus =
            owner->api.cudaEventDestroy(owner->startEvent);
        recordCleanupCUDAFailure(
            result,
            destructionStatus,
            OVTRTRG10PreprocessingStageEventDestruction
        );
        if (destructionStatus != cudaSuccess) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->startEvent = nullptr;
    }
    if (owner->stream != nullptr) {
        cudaError_t destructionStatus =
            owner->api.cudaStreamDestroy(owner->stream);
        recordCleanupCUDAFailure(
            result,
            destructionStatus,
            OVTRTRG10PreprocessingStageStreamDestruction
        );
        if (destructionStatus != cudaSuccess) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->stream = nullptr;
    }
    if (owner->program != nullptr) {
        nvrtcResult destructionStatus =
            owner->api.nvrtcDestroyProgram(&owner->program);
        recordCleanupNVRTCFailure(
            result,
            destructionStatus,
            OVTRTRG10PreprocessingStageNVRTCProgramDestruction
        );
        if (destructionStatus != NVRTC_SUCCESS) {
            return OVTRTStatusPreprocessingFailure;
        }
    }
    if (owner->nvrtcLibrary != nullptr) {
        int closeStatus = dlclose(owner->nvrtcLibrary);
        recordCleanupDynamicLoaderFailure(
            result,
            closeStatus,
            OVTRTRG10PreprocessingStageLibraryClose
        );
        if (closeStatus != 0) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->nvrtcLibrary = nullptr;
    }
    if (owner->cudaDriverLibrary != nullptr) {
        int closeStatus = dlclose(owner->cudaDriverLibrary);
        recordCleanupDynamicLoaderFailure(
            result,
            closeStatus,
            OVTRTRG10PreprocessingStageLibraryClose
        );
        if (closeStatus != 0) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->cudaDriverLibrary = nullptr;
    }
    if (owner->cudaRuntimeLibrary != nullptr) {
        int closeStatus = dlclose(owner->cudaRuntimeLibrary);
        recordCleanupDynamicLoaderFailure(
            result,
            closeStatus,
            OVTRTRG10PreprocessingStageLibraryClose
        );
        if (closeStatus != 0) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->cudaRuntimeLibrary = nullptr;
    }
    return OVTRTStatusSuccess;
}

OVTRTStatus finishFailedCreation(
    OVTRTRG10Preprocessor *owner,
    OVTRTRG10Preprocessor **preprocessor,
    OVTRTRG10PreprocessingResult *result,
    OVTRTStatus operationStatus
) noexcept {
    OVTRTStatus cleanupStatus = cleanupOwner(owner, result);
    if (cleanupStatus == OVTRTStatusSuccess) {
        delete owner;
    } else {
        // A non-null output on failure transfers the partially initialized
        // owner to the caller so cleanup can be retried without freeing
        // resources that CUDA may still access.
        *preprocessor = owner;
    }
    return operationStatus;
}

OVTRTStatus compileKernel(
    OVTRTRG10Preprocessor *owner,
    OVTRTRG10PreprocessingResult *result
) {
    CUresult driverStatus = owner->api.cuInit(0);
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage =
            OVTRTRG10PreprocessingStageDriverInitialization;
        result->cudaDriverErrorCode =
            static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }
    CUdevice device = 0;
    driverStatus = owner->api.cuDeviceGet(&device, 0);
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage =
            OVTRTRG10PreprocessingStageDriverInitialization;
        result->cudaDriverErrorCode =
            static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }
    int computeMajor = 0;
    int computeMinor = 0;
    driverStatus = owner->api.cuDeviceGetAttribute(
        &computeMajor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        device
    );
    if (driverStatus == CUDA_SUCCESS) {
        driverStatus = owner->api.cuDeviceGetAttribute(
            &computeMinor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device
        );
    }
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage =
            OVTRTRG10PreprocessingStageDriverInitialization;
        result->cudaDriverErrorCode =
            static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }

    nvrtcResult nvrtcStatus = owner->api.nvrtcCreateProgram(
        &owner->program,
        RG10_KERNEL_SOURCE,
        "OpenVisionRG10.cu",
        0,
        nullptr,
        nullptr
    );
    if (nvrtcStatus != NVRTC_SUCCESS) {
        result->failureStage =
            OVTRTRG10PreprocessingStageNVRTCProgramCreation;
        result->nvrtcErrorCode =
            static_cast<int32_t>(nvrtcStatus);
        return OVTRTStatusNVRTCFailure;
    }

    std::string architecture =
        "--gpu-architecture=compute_" +
        std::to_string(computeMajor) +
        std::to_string(computeMinor);
    char const *options[] = {
        "--std=c++14",
        architecture.c_str()
    };
    nvrtcStatus = owner->api.nvrtcCompileProgram(
        owner->program,
        2,
        options
    );
    result->nvrtcCompilationCount = 1;
    if (nvrtcStatus != NVRTC_SUCCESS) {
        result->failureStage =
            OVTRTRG10PreprocessingStageNVRTCCompilation;
        result->nvrtcErrorCode =
            static_cast<int32_t>(nvrtcStatus);
        return OVTRTStatusNVRTCFailure;
    }

    size_t ptxSize = 0;
    nvrtcStatus = owner->api.nvrtcGetPTXSize(
        owner->program,
        &ptxSize
    );
    if (nvrtcStatus != NVRTC_SUCCESS || ptxSize == 0) {
        result->failureStage =
            OVTRTRG10PreprocessingStagePTXAccess;
        result->nvrtcErrorCode =
            static_cast<int32_t>(nvrtcStatus);
        return OVTRTStatusNVRTCFailure;
    }
    std::vector<char> ptx;
    try {
        ptx.resize(ptxSize);
    } catch (std::bad_alloc const &) {
        result->failureStage =
            OVTRTRG10PreprocessingStagePTXAccess;
        return OVTRTStatusAllocationFailed;
    }
    nvrtcStatus = owner->api.nvrtcGetPTX(
        owner->program,
        ptx.data()
    );
    if (nvrtcStatus != NVRTC_SUCCESS) {
        result->failureStage =
            OVTRTRG10PreprocessingStagePTXAccess;
        result->nvrtcErrorCode =
            static_cast<int32_t>(nvrtcStatus);
        return OVTRTStatusNVRTCFailure;
    }
    nvrtcResult destroyStatus =
        owner->api.nvrtcDestroyProgram(&owner->program);
    if (destroyStatus != NVRTC_SUCCESS) {
        recordCleanupNVRTCFailure(
            result,
            destroyStatus,
            OVTRTRG10PreprocessingStageNVRTCProgramDestruction
        );
        return OVTRTStatusPreprocessingFailure;
    }

    driverStatus = owner->api.cuModuleLoadData(
        &owner->module,
        ptx.data()
    );
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage =
            OVTRTRG10PreprocessingStageModuleLoad;
        result->cudaDriverErrorCode =
            static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }
    driverStatus = owner->api.cuModuleGetFunction(
        &owner->kernel,
        owner->module,
        "ovtrt_rg10_preprocess"
    );
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage =
            OVTRTRG10PreprocessingStageKernelLookup;
        result->cudaDriverErrorCode =
            static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }
    return OVTRTStatusSuccess;
}

}  // namespace
#else
struct OVTRTRG10Preprocessor {
    uint8_t unavailable;
};
#endif

OVTRTStatus ovtrt_rg10_preprocessor_create(
    OVTRTRG10PreprocessingConfiguration const *configuration,
    OVTRTRG10Preprocessor **preprocessor,
    OVTRTRG10PreprocessingResult *result
) {
    if (
        configuration == nullptr ||
        preprocessor == nullptr ||
        result == nullptr
    ) {
        return OVTRTStatusInvalidArgument;
    }
    *preprocessor = nullptr;
    resetResult(result, configuration);
    if (!isValidConfiguration(*configuration)) {
        result->failureStage =
            OVTRTRG10PreprocessingStageConfiguration;
        return OVTRTStatusInvalidArgument;
    }

#if OVTRT_HAS_RG10_PREPROCESSING
    auto *owner = new (std::nothrow) OVTRTRG10Preprocessor{};
    if (owner == nullptr) {
        return OVTRTStatusAllocationFailed;
    }
    owner->configuration = *configuration;
    char const *cudaRuntimeNames[] = {
        "libcudart.so.13",
        "libcudart.so"
    };
    char const *cudaDriverNames[] = {
        "libcuda.so.1",
        "libcuda.so"
    };
    char const *nvrtcNames[] = {
        "libnvrtc.so.13",
        "libnvrtc.so"
    };
    owner->cudaRuntimeLibrary = openLibrary(
        cudaRuntimeNames,
        sizeof(cudaRuntimeNames) / sizeof(cudaRuntimeNames[0])
    );
    owner->cudaDriverLibrary = openLibrary(
        cudaDriverNames,
        sizeof(cudaDriverNames) / sizeof(cudaDriverNames[0])
    );
    owner->nvrtcLibrary = openLibrary(
        nvrtcNames,
        sizeof(nvrtcNames) / sizeof(nvrtcNames[0])
    );
    if (
        owner->cudaRuntimeLibrary == nullptr ||
        owner->cudaDriverLibrary == nullptr ||
        owner->nvrtcLibrary == nullptr
    ) {
        if (owner->cudaRuntimeLibrary == nullptr) {
            result->libraryOpenFailureMask |=
                OVTRTRG10LibraryOpenFailureCUDARuntime;
        }
        if (owner->cudaDriverLibrary == nullptr) {
            result->libraryOpenFailureMask |=
                OVTRTRG10LibraryOpenFailureCUDADriver;
        }
        if (owner->nvrtcLibrary == nullptr) {
            result->libraryOpenFailureMask |=
                OVTRTRG10LibraryOpenFailureNVRTC;
        }
        result->failureStage =
            OVTRTRG10PreprocessingStageLibraryOpen;
        return finishFailedCreation(
            owner,
            preprocessor,
            result,
            OVTRTStatusUnavailable
        );
    }
    if (!loadRuntimeAPI(owner)) {
        result->failureStage =
            OVTRTRG10PreprocessingStageSymbolLoad;
        return finishFailedCreation(
            owner,
            preprocessor,
            result,
            OVTRTStatusUnavailable
        );
    }

    cudaError_t cudaStatus = owner->api.cudaStreamCreateWithFlags(
        &owner->stream,
        cudaStreamNonBlocking
    );
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTRG10PreprocessingStageStreamCreation;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
    }
    if (result->failureStage == OVTRTRG10PreprocessingStageNone) {
        cudaStatus = owner->api.cudaEventCreateWithFlags(
            &owner->startEvent,
            cudaEventDefault
        );
        if (cudaStatus == cudaSuccess) {
            cudaStatus = owner->api.cudaEventCreateWithFlags(
                &owner->outputReadyEvent,
                cudaEventDefault
            );
        }
        if (cudaStatus != cudaSuccess) {
            result->failureStage =
                OVTRTRG10PreprocessingStageEventCreation;
            result->cudaErrorCode =
                static_cast<int32_t>(cudaStatus);
        }
    }
    if (result->failureStage == OVTRTRG10PreprocessingStageNone) {
        cudaStatus = owner->api.cudaMalloc(
            &owner->deviceInput,
            static_cast<size_t>(configuration->sourceByteCount)
        );
        if (cudaStatus != cudaSuccess) {
            result->failureStage =
                OVTRTRG10PreprocessingStageInputAllocation;
            result->cudaErrorCode =
                static_cast<int32_t>(cudaStatus);
        } else {
            result->frameSizedDeviceAllocationCount += 1;
        }
    }
    size_t outputByteCount = static_cast<size_t>(
        result->outputElementCount * sizeof(float)
    );
    if (result->failureStage == OVTRTRG10PreprocessingStageNone) {
        cudaStatus = owner->api.cudaMalloc(
            &owner->deviceOutput,
            outputByteCount
        );
        if (cudaStatus != cudaSuccess) {
            result->failureStage =
                OVTRTRG10PreprocessingStageOutputAllocation;
            result->cudaErrorCode =
                static_cast<int32_t>(cudaStatus);
        } else {
            result->frameSizedDeviceAllocationCount += 1;
        }
    }
    if (result->failureStage == OVTRTRG10PreprocessingStageNone) {
        cudaStatus = owner->api.cudaMalloc(
            &owner->deviceConfiguration,
            KERNEL_CONFIGURATION_FLOAT_COUNT * sizeof(float)
        );
        if (cudaStatus != cudaSuccess) {
            result->failureStage =
                OVTRTRG10PreprocessingStageConfigurationAllocation;
            result->cudaErrorCode =
                static_cast<int32_t>(cudaStatus);
        }
    }

    float kernelConfiguration[KERNEL_CONFIGURATION_FLOAT_COUNT] = {
        configuration->blackLevelR,
        configuration->blackLevelGreenR,
        configuration->blackLevelGreenB,
        configuration->blackLevelB,
        configuration->whiteLevel,
        configuration->gainR,
        configuration->gainGreenR,
        configuration->gainGreenB,
        configuration->gainB,
        configuration->colorMatrix00,
        configuration->colorMatrix01,
        configuration->colorMatrix02,
        configuration->colorMatrix10,
        configuration->colorMatrix11,
        configuration->colorMatrix12,
        configuration->colorMatrix20,
        configuration->colorMatrix21,
        configuration->colorMatrix22,
        configuration->letterboxR,
        configuration->letterboxG,
        configuration->letterboxB,
        configuration->normalizationScale,
        configuration->normalizationBias
    };
    if (result->failureStage == OVTRTRG10PreprocessingStageNone) {
        cudaStatus = owner->api.cudaMemcpy(
            owner->deviceConfiguration,
            kernelConfiguration,
            sizeof(kernelConfiguration),
            cudaMemcpyHostToDevice
        );
        if (cudaStatus != cudaSuccess) {
            result->failureStage =
                OVTRTRG10PreprocessingStageConfigurationTransfer;
            result->cudaErrorCode =
                static_cast<int32_t>(cudaStatus);
        }
    }

    OVTRTStatus status = result->failureStage ==
            OVTRTRG10PreprocessingStageNone
        ? compileKernel(owner, result)
        : OVTRTStatusCUDARuntimeFailure;
    if (status != OVTRTStatusSuccess) {
        return finishFailedCreation(
            owner,
            preprocessor,
            result,
            status
        );
    }
    *preprocessor = owner;
    return OVTRTStatusSuccess;
#else
    result->failureStage =
        OVTRTRG10PreprocessingStageLibraryOpen;
    result->libraryOpenFailureMask =
        OVTRTRG10LibraryOpenFailureCUDARuntime |
        OVTRTRG10LibraryOpenFailureCUDADriver |
        OVTRTRG10LibraryOpenFailureNVRTC;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_rg10_preprocessor_submit(
    OVTRTRG10Preprocessor *preprocessor,
    void const *source,
    uint64_t sourceByteCount,
    OVTRTRG10Orientation orientation,
    OVTRTRG10PreprocessingResult *result
) {
    if (
        preprocessor == nullptr ||
        source == nullptr ||
        result == nullptr
    ) {
        return OVTRTStatusInvalidArgument;
    }
#if OVTRT_HAS_RG10_PREPROCESSING
    resetResult(result, &preprocessor->configuration);
    if (
        sourceByteCount < preprocessor->configuration.sourceByteCount ||
        orientation < OVTRTRG10OrientationUp ||
        orientation > OVTRTRG10OrientationLeft
    ) {
        result->failureStage =
            OVTRTRG10PreprocessingStageConfiguration;
        return OVTRTStatusInvalidArgument;
    }
    if (preprocessor->inFlight) {
        return OVTRTStatusResourceBusy;
    }
    cudaError_t cudaStatus = preprocessor->api.cudaEventRecord(
        preprocessor->startEvent,
        nullptr
    );
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTRG10PreprocessingStageEventTiming;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        return OVTRTStatusCUDARuntimeFailure;
    }
    // The source is a scoped OpenVision borrow. A synchronous H2D transfer is
    // required so no CUDA source read can escape the caller's borrow closure,
    // including error exits. The provider-owned device input remains valid for
    // the asynchronous fused kernel after this call returns.
    cudaStatus = preprocessor->api.cudaMemcpy(
        preprocessor->deviceInput,
        source,
        static_cast<size_t>(
            preprocessor->configuration.sourceByteCount
        ),
        cudaMemcpyHostToDevice
    );
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTRG10PreprocessingStageHostToDevice;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        return OVTRTStatusCUDARuntimeFailure;
    }
    result->fullFrameHostToDeviceCopyCount = 1;
    result->sourceReadCompleted = 1;
    result->sourceReadFencePassed = 1;

    unsigned int sourceWidth =
        preprocessor->configuration.sourceWidth;
    unsigned int sourceHeight =
        preprocessor->configuration.sourceHeight;
    unsigned int sourceBytesPerRow =
        preprocessor->configuration.sourceBytesPerRow;
    unsigned int outputWidth =
        preprocessor->configuration.outputWidth;
    unsigned int outputHeight =
        preprocessor->configuration.outputHeight;
    unsigned int orientationValue =
        static_cast<unsigned int>(orientation);
    unsigned int resizePolicy = static_cast<unsigned int>(
        preprocessor->configuration.resizePolicy
    );
    unsigned int tensorLayout = static_cast<unsigned int>(
        preprocessor->configuration.tensorLayout
    );
    unsigned int channelOrder = static_cast<unsigned int>(
        preprocessor->configuration.channelOrder
    );
    unsigned int applySRGBTransfer =
        preprocessor->configuration.applySRGBTransfer;
    void *arguments[] = {
        &preprocessor->deviceInput,
        &preprocessor->deviceOutput,
        &preprocessor->deviceConfiguration,
        &sourceWidth,
        &sourceHeight,
        &sourceBytesPerRow,
        &outputWidth,
        &outputHeight,
        &orientationValue,
        &resizePolicy,
        &tensorLayout,
        &channelOrder,
        &applySRGBTransfer
    };
    unsigned int blockWidth = 16;
    unsigned int blockHeight = 16;
    unsigned int gridWidth =
        (outputWidth + blockWidth - 1) / blockWidth;
    unsigned int gridHeight =
        (outputHeight + blockHeight - 1) / blockHeight;
    CUresult driverStatus = preprocessor->api.cuLaunchKernel(
        preprocessor->kernel,
        gridWidth,
        gridHeight,
        1,
        blockWidth,
        blockHeight,
        1,
        0,
        reinterpret_cast<CUstream>(preprocessor->stream),
        arguments,
        nullptr
    );
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage =
            OVTRTRG10PreprocessingStageKernelLaunch;
        result->cudaDriverErrorCode =
            static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }
    result->kernelLaunchCount = 1;
    cudaStatus = preprocessor->api.cudaEventRecord(
        preprocessor->outputReadyEvent,
        preprocessor->stream
    );
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTRG10PreprocessingStageOutputEvent;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        cudaError_t synchronizationStatus =
            preprocessor->api.cudaStreamSynchronize(
                preprocessor->stream
            );
        recordCleanupCUDAFailure(
            result,
            synchronizationStatus,
            OVTRTRG10PreprocessingStageStreamSynchronization
        );
        preprocessor->inFlight =
            synchronizationStatus != cudaSuccess;
        return OVTRTStatusCUDARuntimeFailure;
    }
    preprocessor->inFlight = true;
    result->explicitFrameSizedDeviceAllocationCountAfterPreparation = 0;
    return OVTRTStatusSuccess;
#else
    resetResult(result, nullptr);
    result->failureStage =
        OVTRTRG10PreprocessingStageLibraryOpen;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_rg10_preprocessor_wait(
    OVTRTRG10Preprocessor *preprocessor,
    OVTRTRG10PreprocessingResult *result
) {
    if (preprocessor == nullptr || result == nullptr) {
        return OVTRTStatusInvalidArgument;
    }
#if OVTRT_HAS_RG10_PREPROCESSING
    resetResult(result, &preprocessor->configuration);
    if (!preprocessor->inFlight) {
        return OVTRTStatusResourceBusy;
    }
    cudaError_t cudaStatus =
        preprocessor->api.cudaEventSynchronize(
            preprocessor->outputReadyEvent
        );
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTRG10PreprocessingStageOutputSynchronization;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        cudaError_t synchronizationStatus =
            preprocessor->api.cudaStreamSynchronize(
                preprocessor->stream
            );
        recordCleanupCUDAFailure(
            result,
            synchronizationStatus,
            OVTRTRG10PreprocessingStageStreamSynchronization
        );
        preprocessor->inFlight =
            synchronizationStatus != cudaSuccess;
        return OVTRTStatusCUDARuntimeFailure;
    }
    result->outputReadyEventPassed = 1;
    float milliseconds = 0.0F;
    cudaStatus = preprocessor->api.cudaEventElapsedTime(
        &milliseconds,
        preprocessor->startEvent,
        preprocessor->outputReadyEvent
    );
    preprocessor->inFlight = false;
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTRG10PreprocessingStageEventTiming;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        return OVTRTStatusCUDARuntimeFailure;
    }
    result->gpuMilliseconds = static_cast<double>(milliseconds);
    result->explicitFrameSizedDeviceAllocationCountAfterPreparation = 0;
    return OVTRTStatusSuccess;
#else
    resetResult(result, nullptr);
    result->failureStage =
        OVTRTRG10PreprocessingStageLibraryOpen;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_rg10_preprocessor_output(
    OVTRTRG10Preprocessor *preprocessor,
    OVTRTDeviceTensorView *output
) {
    if (preprocessor == nullptr || output == nullptr) {
        return OVTRTStatusInvalidArgument;
    }
    std::memset(output, 0, sizeof(*output));
#if OVTRT_HAS_RG10_PREPROCESSING
    if (preprocessor->inFlight) {
        return OVTRTStatusResourceBusy;
    }
    uint64_t elementCount =
        static_cast<uint64_t>(
            preprocessor->configuration.outputWidth
        ) *
        static_cast<uint64_t>(
            preprocessor->configuration.outputHeight
        ) *
        3ULL;
    output->deviceAddress = preprocessor->deviceOutput;
    output->elementCount = elementCount;
    output->byteCount = elementCount * sizeof(float);
    output->width = preprocessor->configuration.outputWidth;
    output->height = preprocessor->configuration.outputHeight;
    output->channelCount = 3;
    output->layout = preprocessor->configuration.tensorLayout;
    output->channelOrder =
        preprocessor->configuration.channelOrder;
    return OVTRTStatusSuccess;
#else
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_rg10_preprocessor_copy_output(
    OVTRTRG10Preprocessor *preprocessor,
    float *destination,
    uint64_t destinationElementCount,
    OVTRTRG10PreprocessingResult *result
) {
    if (
        preprocessor == nullptr ||
        destination == nullptr ||
        result == nullptr
    ) {
        return OVTRTStatusInvalidArgument;
    }
#if OVTRT_HAS_RG10_PREPROCESSING
    resetResult(result, &preprocessor->configuration);
    if (
        preprocessor->inFlight ||
        destinationElementCount < result->outputElementCount
    ) {
        result->failureStage =
            OVTRTRG10PreprocessingStageOutputReadback;
        return OVTRTStatusInvalidArgument;
    }
    cudaError_t cudaStatus = preprocessor->api.cudaMemcpy(
        destination,
        preprocessor->deviceOutput,
        static_cast<size_t>(
            result->outputElementCount * sizeof(float)
        ),
        cudaMemcpyDeviceToHost
    );
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTRG10PreprocessingStageOutputReadback;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        return OVTRTStatusCUDARuntimeFailure;
    }
    result->deviceToHostVerificationCopyCount = 1;
    return OVTRTStatusSuccess;
#else
    resetResult(result, nullptr);
    result->failureStage =
        OVTRTRG10PreprocessingStageLibraryOpen;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_rg10_preprocessor_destroy(
    OVTRTRG10Preprocessor **preprocessor,
    OVTRTRG10PreprocessingResult *result
) {
    if (preprocessor == nullptr || result == nullptr) {
        return OVTRTStatusInvalidArgument;
    }
    resetResult(result, nullptr);
    if (*preprocessor == nullptr) {
        return OVTRTStatusInvalidArgument;
    }
#if OVTRT_HAS_RG10_PREPROCESSING
    OVTRTRG10Preprocessor *owner = *preprocessor;
    OVTRTStatus status = cleanupOwner(owner, result);
    if (status != OVTRTStatusSuccess) {
        return status;
    }
    delete owner;
    *preprocessor = nullptr;
    return OVTRTStatusSuccess;
#else
    delete *preprocessor;
    *preprocessor = nullptr;
    return OVTRTStatusSuccess;
#endif
}

#if defined(OVTRT_ENABLE_TEST_HOOKS)
OVTRTStatus
ovtrt_rg10_preprocessor_test_fail_next_cleanup_synchronization(
    OVTRTRG10Preprocessor *preprocessor
) {
    if (preprocessor == nullptr) {
        return OVTRTStatusInvalidArgument;
    }
#if OVTRT_HAS_RG10_PREPROCESSING
    preprocessor->failNextCleanupSynchronization = true;
    return OVTRTStatusSuccess;
#else
    return OVTRTStatusUnavailable;
#endif
}
#endif
