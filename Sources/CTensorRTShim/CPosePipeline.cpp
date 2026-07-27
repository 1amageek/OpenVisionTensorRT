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
#define OVTRT_HAS_POSE_PIPELINE 1
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#else
#define OVTRT_HAS_POSE_PIPELINE 0
#endif

namespace {

constexpr uint32_t MAXIMUM_DETECTION_COUNT = 100;
constexpr uint32_t MAXIMUM_REGION_COUNT = 4;
constexpr uint32_t MAXIMUM_JOINT_COUNT = 256;
constexpr uint64_t MAXIMUM_POSE_ELEMENT_COUNT =
    static_cast<uint64_t>(MAXIMUM_REGION_COUNT) * 3ULL * 512ULL * 512ULL;
constexpr uint32_t DEVICE_CONFIGURATION_FLOAT_COUNT = 24;

bool isFinite(float value) noexcept {
    return std::isfinite(static_cast<double>(value));
}

bool isValidConfiguration(
    OVTRTPosePipelineConfiguration const &configuration
) noexcept {
    if (
        configuration.sourceWidth < 2 ||
        configuration.sourceHeight < 2 ||
        configuration.sourceBytesPerRow <
            configuration.sourceWidth * 2U ||
        configuration.detectorInputWidth == 0 ||
        configuration.detectorInputHeight == 0 ||
        configuration.poseInputWidth == 0 ||
        configuration.poseInputHeight == 0 ||
        configuration.maximumRegionCount == 0 ||
        configuration.maximumRegionCount > MAXIMUM_REGION_COUNT ||
        configuration.jointCount == 0 ||
        configuration.jointCount > MAXIMUM_JOINT_COUNT ||
        !isFinite(configuration.minimumDetectionConfidence) ||
        configuration.minimumDetectionConfidence < 0.0F ||
        configuration.minimumDetectionConfidence > 1.0F ||
        !isFinite(configuration.regionScale) ||
        configuration.regionScale <= 0.0F ||
        configuration.sourceWordLayout <
            OVTRTRG10WordLayoutLeastSignificantBits ||
        configuration.sourceWordLayout >
            OVTRTRG10WordLayoutMostSignificantBits ||
        configuration.applySRGBTransfer > 1
    ) {
        return false;
    }
    uint64_t poseElementCount =
        static_cast<uint64_t>(configuration.maximumRegionCount) *
        3ULL *
        static_cast<uint64_t>(configuration.poseInputWidth) *
        static_cast<uint64_t>(configuration.poseInputHeight);
    if (
        poseElementCount == 0 ||
        poseElementCount > MAXIMUM_POSE_ELEMENT_COUNT
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
    for (uint32_t index = 0; index < 4; ++index) {
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
    float remaining[] = {
        configuration.colorMatrix00,
        configuration.colorMatrix01,
        configuration.colorMatrix02,
        configuration.colorMatrix10,
        configuration.colorMatrix11,
        configuration.colorMatrix12,
        configuration.colorMatrix20,
        configuration.colorMatrix21,
        configuration.colorMatrix22,
        configuration.normalizationScaleR,
        configuration.normalizationScaleG,
        configuration.normalizationScaleB,
        configuration.normalizationBiasR,
        configuration.normalizationBiasG,
        configuration.normalizationBiasB
    };
    for (float value : remaining) {
        if (!isFinite(value)) {
            return false;
        }
    }
    return true;
}

void resetResult(
    OVTRTPosePipelineResult *result,
    OVTRTPosePipelineConfiguration const *configuration
) noexcept {
    *result = OVTRTPosePipelineResult{};
    if (configuration == nullptr) {
        return;
    }
    uint64_t poseBytes =
        static_cast<uint64_t>(configuration->maximumRegionCount) *
        3ULL *
        static_cast<uint64_t>(configuration->poseInputWidth) *
        static_cast<uint64_t>(configuration->poseInputHeight) *
        sizeof(float);
    uint64_t regionBytes =
        static_cast<uint64_t>(configuration->maximumRegionCount) *
        sizeof(OVTRTPoseRegion);
    uint64_t jointBytes =
        static_cast<uint64_t>(configuration->maximumRegionCount) *
        static_cast<uint64_t>(configuration->jointCount) *
        sizeof(OVTRTPoseJoint);
    result->persistentDeviceAllocationByteCount =
        poseBytes +
        regionBytes +
        jointBytes +
        sizeof(uint32_t) +
        DEVICE_CONFIGURATION_FLOAT_COUNT * sizeof(float);
}

}  // namespace

#if OVTRT_HAS_POSE_PIPELINE
namespace {

template <typename Function>
Function loadFunction(void *library, char const *name) noexcept {
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
using NVRTCDestroyProgramFunction =
    nvrtcResult (*)(nvrtcProgram *);

struct PoseRuntimeAPI {
    CUDAMallocFunction cudaMalloc{nullptr};
    CUDAFreeFunction cudaFree{nullptr};
    CUDAStreamCreateWithFlagsFunction cudaStreamCreateWithFlags{nullptr};
    CUDAStreamDestroyFunction cudaStreamDestroy{nullptr};
    CUDAStreamSynchronizeFunction cudaStreamSynchronize{nullptr};
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
    NVRTCDestroyProgramFunction nvrtcDestroyProgram{nullptr};

    bool isComplete() const noexcept {
        return
            cudaMalloc != nullptr &&
            cudaFree != nullptr &&
            cudaStreamCreateWithFlags != nullptr &&
            cudaStreamDestroy != nullptr &&
            cudaStreamSynchronize != nullptr &&
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
            nvrtcDestroyProgram != nullptr;
    }
};

char const *POSE_KERNEL_SOURCE = R"cuda(
extern "C" {

struct PoseRegion {
    float centerX;
    float centerY;
    float width;
    float height;
    float confidence;
};

struct PoseJoint {
    float normalizedX;
    float normalizedY;
    float confidence;
};

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
    unsigned int sampleBitShift,
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
    value = (value >> sampleBitShift) & 1023U;
    unsigned int site =
        ((static_cast<unsigned int>(y) & 1U) << 1U) |
        (static_cast<unsigned int>(x) & 1U);
    float blackLevel = configuration[site];
    return ovtrt_clamp(
        (static_cast<float>(value) - blackLevel) /
            (configuration[4] - blackLevel) *
            configuration[5 + site],
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
    unsigned int sampleBitShift,
    float const *configuration,
    float &red,
    float &green,
    float &blue
) {
    bool evenX = (x & 1) == 0;
    bool evenY = (y & 1) == 0;
    float center = ovtrt_raw(
        source, width, height, bytesPerRow, x, y, sampleBitShift, configuration
    );
    if (evenX && evenY) {
        red = center;
        green = 0.25F * (
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x, y - 1, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x, y + 1, sampleBitShift, configuration)
        );
        blue = 0.25F * (
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y - 1, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y - 1, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y + 1, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y + 1, sampleBitShift, configuration)
        );
        return;
    }
    if (!evenX && !evenY) {
        blue = center;
        green = 0.25F * (
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x, y - 1, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x, y + 1, sampleBitShift, configuration)
        );
        red = 0.25F * (
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y - 1, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y - 1, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y + 1, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y + 1, sampleBitShift, configuration)
        );
        return;
    }
    green = center;
    if (!evenX && evenY) {
        red = 0.5F * (
            ovtrt_raw(source, width, height, bytesPerRow, x - 1, y, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x + 1, y, sampleBitShift, configuration)
        );
        blue = 0.5F * (
            ovtrt_raw(source, width, height, bytesPerRow, x, y - 1, sampleBitShift, configuration) +
            ovtrt_raw(source, width, height, bytesPerRow, x, y + 1, sampleBitShift, configuration)
        );
        return;
    }
    red = 0.5F * (
        ovtrt_raw(source, width, height, bytesPerRow, x, y - 1, sampleBitShift, configuration) +
        ovtrt_raw(source, width, height, bytesPerRow, x, y + 1, sampleBitShift, configuration)
    );
    blue = 0.5F * (
        ovtrt_raw(source, width, height, bytesPerRow, x - 1, y, sampleBitShift, configuration) +
        ovtrt_raw(source, width, height, bytesPerRow, x + 1, y, sampleBitShift, configuration)
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

__global__ void ovtrt_select_regions(
    float const *detections,
    long long const *classes,
    unsigned int detectionCount,
    unsigned int detectorWidth,
    unsigned int detectorHeight,
    unsigned int orientedWidth,
    unsigned int orientedHeight,
    float minimumConfidence,
    float regionScale,
    unsigned int poseWidth,
    unsigned int poseHeight,
    unsigned int maximumRegionCount,
    PoseRegion *regions,
    unsigned int *regionCount
) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) {
        return;
    }
    unsigned int count = 0;
    float detectorScale = fminf(
        static_cast<float>(detectorWidth) /
            static_cast<float>(orientedWidth),
        static_cast<float>(detectorHeight) /
            static_cast<float>(orientedHeight)
    );
    float offsetX = (
        static_cast<float>(detectorWidth) -
        static_cast<float>(orientedWidth) * detectorScale
    ) * 0.5F;
    float offsetY = (
        static_cast<float>(detectorHeight) -
        static_cast<float>(orientedHeight) * detectorScale
    ) * 0.5F;
    float poseAspect =
        static_cast<float>(poseWidth) / static_cast<float>(poseHeight);

    for (unsigned int index = 0; index < detectionCount; ++index) {
        float const *detection = detections + index * 5U;
        float confidence = detection[4];
        if (
            classes[index] != 0LL ||
            !isfinite(confidence) ||
            confidence < minimumConfidence
        ) {
            continue;
        }
        float x1 = ovtrt_clamp(
            (detection[0] - offsetX) / detectorScale,
            0.0F,
            static_cast<float>(orientedWidth)
        );
        float y1 = ovtrt_clamp(
            (detection[1] - offsetY) / detectorScale,
            0.0F,
            static_cast<float>(orientedHeight)
        );
        float x2 = ovtrt_clamp(
            (detection[2] - offsetX) / detectorScale,
            0.0F,
            static_cast<float>(orientedWidth)
        );
        float y2 = ovtrt_clamp(
            (detection[3] - offsetY) / detectorScale,
            0.0F,
            static_cast<float>(orientedHeight)
        );
        float width = (x2 - x1) * regionScale;
        float height = (y2 - y1) * regionScale;
        if (
            !isfinite(width) ||
            !isfinite(height) ||
            width <= 1.0F ||
            height <= 1.0F
        ) {
            continue;
        }
        if (width > height * poseAspect) {
            height = width / poseAspect;
        } else {
            width = height * poseAspect;
        }
        PoseRegion region{
            (x1 + x2) * 0.5F,
            (y1 + y2) * 0.5F,
            width,
            height,
            confidence
        };
        unsigned int insertion = count;
        if (insertion > maximumRegionCount) {
            insertion = maximumRegionCount;
        }
        while (
            insertion > 0U &&
            regions[insertion - 1U].confidence < confidence
        ) {
            if (insertion < maximumRegionCount) {
                regions[insertion] = regions[insertion - 1U];
            }
            --insertion;
        }
        if (insertion < maximumRegionCount) {
            regions[insertion] = region;
            if (count < maximumRegionCount) {
                ++count;
            }
        }
    }
    *regionCount = count;
}

__global__ void ovtrt_region_affine(
    unsigned char const *source,
    float *destination,
    float const *configuration,
    PoseRegion const *regions,
    unsigned int regionCount,
    unsigned int sourceWidth,
    unsigned int sourceHeight,
    unsigned int sourceBytesPerRow,
    unsigned int outputWidth,
    unsigned int outputHeight,
    unsigned int orientation,
    unsigned int sampleBitShift,
    unsigned int applySRGBTransfer
) {
    unsigned int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int outputY = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int batch = blockIdx.z;
    if (
        outputX >= outputWidth ||
        outputY >= outputHeight ||
        batch >= regionCount
    ) {
        return;
    }
    PoseRegion region = regions[batch];
    float orientedX =
        (static_cast<float>(outputX) -
            static_cast<float>(outputWidth) * 0.5F) *
            region.width / static_cast<float>(outputWidth) +
        region.centerX;
    float orientedY =
        (static_cast<float>(outputY) -
            static_cast<float>(outputHeight) * 0.5F) *
            region.height / static_cast<float>(outputHeight) +
        region.centerY;
    float sourceX = 0.0F;
    float sourceY = 0.0F;
    ovtrt_oriented_source(
        orientedX,
        orientedY,
        sourceWidth,
        sourceHeight,
        orientation,
        sourceX,
        sourceY
    );
    sourceX = ovtrt_clamp(
        sourceX, 0.0F, static_cast<float>(sourceWidth - 1U)
    );
    sourceY = ovtrt_clamp(
        sourceY, 0.0F, static_cast<float>(sourceHeight - 1U)
    );
    int x0 = static_cast<int>(floorf(sourceX));
    int y0 = static_cast<int>(floorf(sourceY));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float fractionX = sourceX - static_cast<float>(x0);
    float fractionY = sourceY - static_cast<float>(y0);
    float samples[12];
    ovtrt_rgb_at(source, sourceWidth, sourceHeight, sourceBytesPerRow, x0, y0, sampleBitShift, configuration, samples[0], samples[1], samples[2]);
    ovtrt_rgb_at(source, sourceWidth, sourceHeight, sourceBytesPerRow, x1, y0, sampleBitShift, configuration, samples[3], samples[4], samples[5]);
    ovtrt_rgb_at(source, sourceWidth, sourceHeight, sourceBytesPerRow, x0, y1, sampleBitShift, configuration, samples[6], samples[7], samples[8]);
    ovtrt_rgb_at(source, sourceWidth, sourceHeight, sourceBytesPerRow, x1, y1, sampleBitShift, configuration, samples[9], samples[10], samples[11]);
    float camera[3];
    for (unsigned int channel = 0; channel < 3U; ++channel) {
        float top = samples[channel] +
            (samples[3U + channel] - samples[channel]) * fractionX;
        float bottom = samples[6U + channel] +
            (samples[9U + channel] - samples[6U + channel]) * fractionX;
        camera[channel] = top + (bottom - top) * fractionY;
    }
    float red =
        configuration[9] * camera[0] +
        configuration[10] * camera[1] +
        configuration[11] * camera[2];
    float green =
        configuration[12] * camera[0] +
        configuration[13] * camera[1] +
        configuration[14] * camera[2];
    float blue =
        configuration[15] * camera[0] +
        configuration[16] * camera[1] +
        configuration[17] * camera[2];
    red = ovtrt_clamp(red, 0.0F, 1.0F);
    green = ovtrt_clamp(green, 0.0F, 1.0F);
    blue = ovtrt_clamp(blue, 0.0F, 1.0F);
    if (applySRGBTransfer != 0U) {
        red = ovtrt_srgb(red);
        green = ovtrt_srgb(green);
        blue = ovtrt_srgb(blue);
    }
    float channels[3] = {
        red * configuration[18] + configuration[21],
        green * configuration[19] + configuration[22],
        blue * configuration[20] + configuration[23]
    };
    unsigned long long pixelCount =
        static_cast<unsigned long long>(outputWidth) * outputHeight;
    unsigned long long batchStride = pixelCount * 3ULL;
    unsigned long long pixel =
        static_cast<unsigned long long>(outputY) * outputWidth + outputX;
    for (unsigned int channel = 0; channel < 3U; ++channel) {
        destination[
            static_cast<unsigned long long>(batch) * batchStride +
            static_cast<unsigned long long>(channel) * pixelCount +
            pixel
        ] = channels[channel];
    }
}

__global__ void ovtrt_decode_simcc(
    float const *simCCX,
    float const *simCCY,
    PoseRegion const *regions,
    PoseJoint *joints,
    unsigned int regionCount,
    unsigned int jointCount,
    unsigned int xBinCount,
    unsigned int yBinCount,
    unsigned int poseWidth,
    unsigned int poseHeight,
    unsigned int orientedWidth,
    unsigned int orientedHeight
) {
    unsigned int flatIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = regionCount * jointCount;
    if (flatIndex >= total) {
        return;
    }
    unsigned int batch = flatIndex / jointCount;
    unsigned int joint = flatIndex % jointCount;
    float const *xValues =
        simCCX + static_cast<unsigned long long>(flatIndex) * xBinCount;
    float const *yValues =
        simCCY + static_cast<unsigned long long>(flatIndex) * yBinCount;
    unsigned int maximumXIndex = 0;
    unsigned int maximumYIndex = 0;
    float maximumX = -3.402823466e+38F;
    float maximumY = -3.402823466e+38F;
    for (unsigned int index = 0; index < xBinCount; ++index) {
        if (xValues[index] > maximumX) {
            maximumX = xValues[index];
            maximumXIndex = index;
        }
    }
    for (unsigned int index = 0; index < yBinCount; ++index) {
        if (yValues[index] > maximumY) {
            maximumY = yValues[index];
            maximumYIndex = index;
        }
    }
    float confidence = fminf(maximumX, maximumY);
    if (!isfinite(confidence) || confidence <= 0.0F) {
        joints[flatIndex] = PoseJoint{-1.0F, -1.0F, 0.0F};
        return;
    }
    float modelX = static_cast<float>(maximumXIndex) * 0.5F;
    float modelY = static_cast<float>(maximumYIndex) * 0.5F;
    PoseRegion region = regions[batch];
    float imageX =
        modelX / static_cast<float>(poseWidth) * region.width +
        region.centerX -
        region.width * 0.5F;
    float imageY =
        modelY / static_cast<float>(poseHeight) * region.height +
        region.centerY -
        region.height * 0.5F;
    joints[flatIndex] = PoseJoint{
        ovtrt_clamp(
            imageX / static_cast<float>(orientedWidth),
            0.0F,
            1.0F
        ),
        ovtrt_clamp(
            1.0F - imageY / static_cast<float>(orientedHeight),
            0.0F,
            1.0F
        ),
        ovtrt_clamp(confidence, 0.0F, 1.0F)
    };
}

}  // extern "C"
)cuda";

}  // namespace

struct OVTRTPosePipeline {
    OVTRTPosePipelineConfiguration configuration{};
    PoseRuntimeAPI api{};
    void *cudaRuntimeLibrary{nullptr};
    void *cudaDriverLibrary{nullptr};
    void *nvrtcLibrary{nullptr};
    cudaStream_t stream{nullptr};
    void *deviceConfiguration{nullptr};
    void *devicePoseInput{nullptr};
    OVTRTPoseRegion *deviceRegions{nullptr};
    uint32_t *deviceRegionCount{nullptr};
    OVTRTPoseJoint *deviceJoints{nullptr};
    CUmodule module{nullptr};
    CUfunction regionSelectionKernel{nullptr};
    CUfunction regionAffineKernel{nullptr};
    CUfunction simCCDecodeKernel{nullptr};
    uint32_t selectedRegionCount{0};
    uint32_t orientedWidth{0};
    uint32_t orientedHeight{0};
};

namespace {

bool loadRuntimeAPI(OVTRTPosePipeline *owner) noexcept {
    auto &api = owner->api;
    api.cudaMalloc = loadFunction<CUDAMallocFunction>(
        owner->cudaRuntimeLibrary, "cudaMalloc"
    );
    api.cudaFree = loadFunction<CUDAFreeFunction>(
        owner->cudaRuntimeLibrary, "cudaFree"
    );
    api.cudaStreamCreateWithFlags =
        loadFunction<CUDAStreamCreateWithFlagsFunction>(
            owner->cudaRuntimeLibrary, "cudaStreamCreateWithFlags"
        );
    api.cudaStreamDestroy = loadFunction<CUDAStreamDestroyFunction>(
        owner->cudaRuntimeLibrary, "cudaStreamDestroy"
    );
    api.cudaStreamSynchronize =
        loadFunction<CUDAStreamSynchronizeFunction>(
            owner->cudaRuntimeLibrary, "cudaStreamSynchronize"
        );
    api.cudaMemcpy = loadFunction<CUDAMemcpyFunction>(
        owner->cudaRuntimeLibrary, "cudaMemcpy"
    );
    api.cuInit = loadFunction<CUInitFunction>(
        owner->cudaDriverLibrary, "cuInit"
    );
    api.cuDeviceGet = loadFunction<CUDeviceGetFunction>(
        owner->cudaDriverLibrary, "cuDeviceGet"
    );
    api.cuDeviceGetAttribute =
        loadFunction<CUDeviceGetAttributeFunction>(
            owner->cudaDriverLibrary, "cuDeviceGetAttribute"
        );
    api.cuModuleLoadData = loadFunction<CUModuleLoadDataFunction>(
        owner->cudaDriverLibrary, "cuModuleLoadData"
    );
    api.cuModuleUnload = loadFunction<CUModuleUnloadFunction>(
        owner->cudaDriverLibrary, "cuModuleUnload"
    );
    api.cuModuleGetFunction =
        loadFunction<CUModuleGetFunctionFunction>(
            owner->cudaDriverLibrary, "cuModuleGetFunction"
        );
    api.cuLaunchKernel = loadFunction<CULaunchKernelFunction>(
        owner->cudaDriverLibrary, "cuLaunchKernel"
    );
    api.nvrtcCreateProgram =
        loadFunction<NVRTCCreateProgramFunction>(
            owner->nvrtcLibrary, "nvrtcCreateProgram"
        );
    api.nvrtcCompileProgram =
        loadFunction<NVRTCCompileProgramFunction>(
            owner->nvrtcLibrary, "nvrtcCompileProgram"
        );
    api.nvrtcGetPTXSize = loadFunction<NVRTCGetPTXSizeFunction>(
        owner->nvrtcLibrary, "nvrtcGetPTXSize"
    );
    api.nvrtcGetPTX = loadFunction<NVRTCGetPTXFunction>(
        owner->nvrtcLibrary, "nvrtcGetPTX"
    );
    api.nvrtcDestroyProgram =
        loadFunction<NVRTCDestroyProgramFunction>(
            owner->nvrtcLibrary, "nvrtcDestroyProgram"
        );
    return api.isComplete();
}

void recordCleanupCUDA(
    OVTRTPosePipelineResult *result,
    cudaError_t status,
    OVTRTPosePipelineStage stage
) noexcept {
    if (
        status != cudaSuccess &&
        result->cleanupFailureStage == OVTRTPosePipelineStageNone
    ) {
        result->cleanupCUDAErrorCode = static_cast<int32_t>(status);
        result->cleanupFailureStage = stage;
    }
}

void recordCleanupDriver(
    OVTRTPosePipelineResult *result,
    CUresult status,
    OVTRTPosePipelineStage stage
) noexcept {
    if (
        status != CUDA_SUCCESS &&
        result->cleanupFailureStage == OVTRTPosePipelineStageNone
    ) {
        result->cleanupCUDADriverErrorCode = static_cast<int32_t>(status);
        result->cleanupFailureStage = stage;
    }
}

OVTRTStatus cleanup(
    OVTRTPosePipeline *owner,
    OVTRTPosePipelineResult *result
) noexcept {
    if (owner->stream != nullptr) {
        cudaError_t status = owner->api.cudaStreamSynchronize(owner->stream);
        recordCleanupCUDA(
            result, status, OVTRTPosePipelineStageStreamSynchronization
        );
        if (status != cudaSuccess) {
            return OVTRTStatusPreprocessingFailure;
        }
    }
    if (owner->module != nullptr) {
        CUresult status = owner->api.cuModuleUnload(owner->module);
        recordCleanupDriver(
            result, status, OVTRTPosePipelineStageModuleUnload
        );
        if (status != CUDA_SUCCESS) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->module = nullptr;
        owner->regionSelectionKernel = nullptr;
        owner->regionAffineKernel = nullptr;
        owner->simCCDecodeKernel = nullptr;
    }
    void **allocations[] = {
        &owner->deviceConfiguration,
        &owner->devicePoseInput,
        reinterpret_cast<void **>(&owner->deviceRegions),
        reinterpret_cast<void **>(&owner->deviceRegionCount),
        reinterpret_cast<void **>(&owner->deviceJoints)
    };
    for (void **allocation : allocations) {
        if (*allocation == nullptr) {
            continue;
        }
        cudaError_t status = owner->api.cudaFree(*allocation);
        recordCleanupCUDA(
            result, status, OVTRTPosePipelineStageDeviceDeallocation
        );
        if (status != cudaSuccess) {
            return OVTRTStatusPreprocessingFailure;
        }
        *allocation = nullptr;
    }
    if (owner->stream != nullptr) {
        cudaError_t status = owner->api.cudaStreamDestroy(owner->stream);
        recordCleanupCUDA(
            result, status, OVTRTPosePipelineStageStreamDestruction
        );
        if (status != cudaSuccess) {
            return OVTRTStatusPreprocessingFailure;
        }
        owner->stream = nullptr;
    }
    void **libraries[] = {
        &owner->nvrtcLibrary,
        &owner->cudaDriverLibrary,
        &owner->cudaRuntimeLibrary
    };
    for (void **library : libraries) {
        if (*library == nullptr) {
            continue;
        }
        int status = dlclose(*library);
        if (
            status != 0 &&
            result->cleanupFailureStage == OVTRTPosePipelineStageNone
        ) {
            result->cleanupDynamicLoaderErrorCode = status;
            result->cleanupFailureStage =
                OVTRTPosePipelineStageLibraryClose;
        }
        if (status != 0) {
            return OVTRTStatusPreprocessingFailure;
        }
        *library = nullptr;
    }
    return OVTRTStatusSuccess;
}

OVTRTStatus compileKernels(
    OVTRTPosePipeline *owner,
    OVTRTPosePipelineResult *result
) {
    CUresult driverStatus = owner->api.cuInit(0);
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage =
            OVTRTPosePipelineStageDriverInitialization;
        result->cudaDriverErrorCode = static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }
    CUdevice device = 0;
    driverStatus = owner->api.cuDeviceGet(&device, 0);
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage =
            OVTRTPosePipelineStageDriverInitialization;
        result->cudaDriverErrorCode = static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }
    int major = 0;
    int minor = 0;
    driverStatus = owner->api.cuDeviceGetAttribute(
        &major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        device
    );
    if (driverStatus == CUDA_SUCCESS) {
        driverStatus = owner->api.cuDeviceGetAttribute(
            &minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device
        );
    }
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage =
            OVTRTPosePipelineStageDriverInitialization;
        result->cudaDriverErrorCode = static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }
    nvrtcProgram program = nullptr;
    nvrtcResult nvrtcStatus = owner->api.nvrtcCreateProgram(
        &program,
        POSE_KERNEL_SOURCE,
        "OpenVisionPosePipeline.cu",
        0,
        nullptr,
        nullptr
    );
    if (nvrtcStatus != NVRTC_SUCCESS) {
        result->failureStage =
            OVTRTPosePipelineStageNVRTCProgramCreation;
        result->nvrtcErrorCode = static_cast<int32_t>(nvrtcStatus);
        return OVTRTStatusNVRTCFailure;
    }
    std::string architecture =
        "--gpu-architecture=compute_" +
        std::to_string(major) +
        std::to_string(minor);
    char const *options[] = {"--std=c++14", architecture.c_str()};
    nvrtcStatus = owner->api.nvrtcCompileProgram(program, 2, options);
    if (nvrtcStatus != NVRTC_SUCCESS) {
        result->failureStage =
            OVTRTPosePipelineStageNVRTCCompilation;
        result->nvrtcErrorCode = static_cast<int32_t>(nvrtcStatus);
        owner->api.nvrtcDestroyProgram(&program);
        return OVTRTStatusNVRTCFailure;
    }
    size_t ptxSize = 0;
    nvrtcStatus = owner->api.nvrtcGetPTXSize(program, &ptxSize);
    if (nvrtcStatus != NVRTC_SUCCESS || ptxSize == 0) {
        result->failureStage = OVTRTPosePipelineStagePTXAccess;
        result->nvrtcErrorCode = static_cast<int32_t>(nvrtcStatus);
        owner->api.nvrtcDestroyProgram(&program);
        return OVTRTStatusNVRTCFailure;
    }
    std::vector<char> ptx;
    try {
        ptx.resize(ptxSize);
    } catch (std::bad_alloc const &) {
        owner->api.nvrtcDestroyProgram(&program);
        result->failureStage = OVTRTPosePipelineStagePTXAccess;
        return OVTRTStatusAllocationFailed;
    }
    nvrtcStatus = owner->api.nvrtcGetPTX(program, ptx.data());
    nvrtcResult destroyStatus =
        owner->api.nvrtcDestroyProgram(&program);
    if (
        nvrtcStatus != NVRTC_SUCCESS ||
        destroyStatus != NVRTC_SUCCESS
    ) {
        result->failureStage = OVTRTPosePipelineStagePTXAccess;
        result->nvrtcErrorCode = static_cast<int32_t>(
            nvrtcStatus != NVRTC_SUCCESS ? nvrtcStatus : destroyStatus
        );
        return OVTRTStatusNVRTCFailure;
    }
    driverStatus = owner->api.cuModuleLoadData(
        &owner->module, ptx.data()
    );
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage = OVTRTPosePipelineStageModuleLoad;
        result->cudaDriverErrorCode = static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }
    struct KernelBinding {
        CUfunction *destination;
        char const *name;
    };
    KernelBinding bindings[] = {
        {&owner->regionSelectionKernel, "ovtrt_select_regions"},
        {&owner->regionAffineKernel, "ovtrt_region_affine"},
        {&owner->simCCDecodeKernel, "ovtrt_decode_simcc"}
    };
    for (KernelBinding binding : bindings) {
        driverStatus = owner->api.cuModuleGetFunction(
            binding.destination, owner->module, binding.name
        );
        if (driverStatus != CUDA_SUCCESS) {
            result->failureStage =
                OVTRTPosePipelineStageKernelLookup;
            result->cudaDriverErrorCode =
                static_cast<int32_t>(driverStatus);
            return OVTRTStatusCUDADriverFailure;
        }
    }
    return OVTRTStatusSuccess;
}

OVTRTStatus operationFailure(
    OVTRTPosePipeline *owner,
    OVTRTPosePipeline **pipeline,
    OVTRTPosePipelineResult *result,
    OVTRTStatus status
) {
    OVTRTStatus cleanupStatus = cleanup(owner, result);
    if (cleanupStatus == OVTRTStatusSuccess) {
        delete owner;
    } else {
        *pipeline = owner;
    }
    return status;
}

}  // namespace
#else
struct OVTRTPosePipeline {
    OVTRTPosePipelineConfiguration configuration{};
};
#endif

OVTRTStatus ovtrt_pose_pipeline_create(
    OVTRTPosePipelineConfiguration const *configuration,
    OVTRTPosePipeline **pipeline,
    OVTRTPosePipelineResult *result
) {
    if (
        configuration == nullptr ||
        pipeline == nullptr ||
        result == nullptr
    ) {
        return OVTRTStatusInvalidArgument;
    }
    *pipeline = nullptr;
    resetResult(result, configuration);
    if (!isValidConfiguration(*configuration)) {
        result->failureStage = OVTRTPosePipelineStageConfiguration;
        return OVTRTStatusInvalidArgument;
    }
#if OVTRT_HAS_POSE_PIPELINE
    OVTRTPosePipeline *owner =
        new (std::nothrow) OVTRTPosePipeline();
    if (owner == nullptr) {
        return OVTRTStatusAllocationFailed;
    }
    owner->configuration = *configuration;
    char const *runtimeNames[] = {
        "libcudart.so.13",
        "libcudart.so"
    };
    char const *driverNames[] = {
        "libcuda.so.1",
        "libcuda.so"
    };
    char const *nvrtcNames[] = {
        "libnvrtc.so.13",
        "libnvrtc.so"
    };
    owner->cudaRuntimeLibrary = openLibrary(runtimeNames, 2);
    owner->cudaDriverLibrary = openLibrary(driverNames, 2);
    owner->nvrtcLibrary = openLibrary(nvrtcNames, 2);
    if (
        owner->cudaRuntimeLibrary == nullptr ||
        owner->cudaDriverLibrary == nullptr ||
        owner->nvrtcLibrary == nullptr
    ) {
        result->failureStage = OVTRTPosePipelineStageLibraryOpen;
        return operationFailure(
            owner, pipeline, result, OVTRTStatusUnavailable
        );
    }
    if (!loadRuntimeAPI(owner)) {
        result->failureStage = OVTRTPosePipelineStageSymbolLoad;
        return operationFailure(
            owner, pipeline, result, OVTRTStatusUnavailable
        );
    }
    cudaError_t cudaStatus = owner->api.cudaStreamCreateWithFlags(
        &owner->stream, cudaStreamNonBlocking
    );
    if (cudaStatus != cudaSuccess) {
        result->failureStage = OVTRTPosePipelineStageStreamCreation;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        return operationFailure(
            owner, pipeline, result, OVTRTStatusCUDARuntimeFailure
        );
    }
    uint64_t poseElementCount =
        static_cast<uint64_t>(configuration->maximumRegionCount) *
        3ULL *
        configuration->poseInputWidth *
        configuration->poseInputHeight;
    struct Allocation {
        void **address;
        size_t byteCount;
    };
    Allocation allocations[] = {
        {
            &owner->deviceConfiguration,
            DEVICE_CONFIGURATION_FLOAT_COUNT * sizeof(float)
        },
        {
            &owner->devicePoseInput,
            static_cast<size_t>(poseElementCount * sizeof(float))
        },
        {
            reinterpret_cast<void **>(&owner->deviceRegions),
            configuration->maximumRegionCount * sizeof(OVTRTPoseRegion)
        },
        {
            reinterpret_cast<void **>(&owner->deviceRegionCount),
            sizeof(uint32_t)
        },
        {
            reinterpret_cast<void **>(&owner->deviceJoints),
            configuration->maximumRegionCount *
                configuration->jointCount *
                sizeof(OVTRTPoseJoint)
        }
    };
    for (Allocation allocation : allocations) {
        cudaStatus = owner->api.cudaMalloc(
            allocation.address, allocation.byteCount
        );
        if (cudaStatus != cudaSuccess) {
            result->failureStage =
                OVTRTPosePipelineStageDeviceAllocation;
            result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
            return operationFailure(
                owner, pipeline, result, OVTRTStatusCUDARuntimeFailure
            );
        }
    }
    float deviceConfiguration[DEVICE_CONFIGURATION_FLOAT_COUNT] = {
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
        configuration->normalizationScaleR,
        configuration->normalizationScaleG,
        configuration->normalizationScaleB,
        configuration->normalizationBiasR,
        configuration->normalizationBiasG,
        configuration->normalizationBiasB
    };
    cudaStatus = owner->api.cudaMemcpy(
        owner->deviceConfiguration,
        deviceConfiguration,
        sizeof(deviceConfiguration),
        cudaMemcpyHostToDevice
    );
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTPosePipelineStageConfigurationTransfer;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        return operationFailure(
            owner, pipeline, result, OVTRTStatusCUDARuntimeFailure
        );
    }
    OVTRTStatus compileStatus = compileKernels(owner, result);
    if (compileStatus != OVTRTStatusSuccess) {
        return operationFailure(
            owner, pipeline, result, compileStatus
        );
    }
    *pipeline = owner;
    return OVTRTStatusSuccess;
#else
    result->failureStage = OVTRTPosePipelineStageLibraryOpen;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_pose_pipeline_prepare_input(
    OVTRTPosePipeline *pipeline,
    OVTRTRG10SourceView const *source,
    void const *detectionsDeviceAddress,
    uint64_t detectionCount,
    void const *classesDeviceAddress,
    uint64_t classCount,
    OVTRTRG10Orientation orientation,
    OVTRTPoseRegion *regions,
    uint32_t regionCapacity,
    OVTRTDeviceTensorView *poseInput,
    OVTRTPosePipelineResult *result
) {
    if (
        pipeline == nullptr ||
        source == nullptr ||
        detectionsDeviceAddress == nullptr ||
        classesDeviceAddress == nullptr ||
        regions == nullptr ||
        poseInput == nullptr ||
        result == nullptr
    ) {
        return OVTRTStatusInvalidArgument;
    }
    resetResult(result, &pipeline->configuration);
    std::memset(poseInput, 0, sizeof(*poseInput));
#if OVTRT_HAS_POSE_PIPELINE
    if (
        source->deviceAddress == nullptr ||
        source->width != pipeline->configuration.sourceWidth ||
        source->height != pipeline->configuration.sourceHeight ||
        source->bytesPerRow !=
            pipeline->configuration.sourceBytesPerRow ||
        source->byteCount <
            static_cast<uint64_t>(source->bytesPerRow) * source->height ||
        detectionCount == 0 ||
        detectionCount > MAXIMUM_DETECTION_COUNT ||
        classCount < detectionCount ||
        regionCapacity <
            pipeline->configuration.maximumRegionCount ||
        orientation < OVTRTRG10OrientationUp ||
        orientation > OVTRTRG10OrientationLeft
    ) {
        result->failureStage = OVTRTPosePipelineStageConfiguration;
        return OVTRTStatusInvalidArgument;
    }
    bool swapsDimensions =
        orientation >= OVTRTRG10OrientationLeftMirrored;
    pipeline->orientedWidth = swapsDimensions
        ? source->height
        : source->width;
    pipeline->orientedHeight = swapsDimensions
        ? source->width
        : source->height;
    unsigned int detectionCountValue =
        static_cast<unsigned int>(detectionCount);
    unsigned int detectorWidth =
        pipeline->configuration.detectorInputWidth;
    unsigned int detectorHeight =
        pipeline->configuration.detectorInputHeight;
    float minimumConfidence =
        pipeline->configuration.minimumDetectionConfidence;
    float regionScale = pipeline->configuration.regionScale;
    unsigned int poseWidth =
        pipeline->configuration.poseInputWidth;
    unsigned int poseHeight =
        pipeline->configuration.poseInputHeight;
    unsigned int maximumRegionCount =
        pipeline->configuration.maximumRegionCount;
    void const *detections = detectionsDeviceAddress;
    void const *classes = classesDeviceAddress;
    void *arguments[] = {
        &detections,
        &classes,
        &detectionCountValue,
        &detectorWidth,
        &detectorHeight,
        &pipeline->orientedWidth,
        &pipeline->orientedHeight,
        &minimumConfidence,
        &regionScale,
        &poseWidth,
        &poseHeight,
        &maximumRegionCount,
        &pipeline->deviceRegions,
        &pipeline->deviceRegionCount
    };
    CUresult driverStatus = pipeline->api.cuLaunchKernel(
        pipeline->regionSelectionKernel,
        1, 1, 1,
        1, 1, 1,
        0,
        reinterpret_cast<CUstream>(pipeline->stream),
        arguments,
        nullptr
    );
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage =
            OVTRTPosePipelineStageRegionSelection;
        result->cudaDriverErrorCode =
            static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }
    result->regionSelectionKernelLaunchCount = 1;
    cudaError_t cudaStatus =
        pipeline->api.cudaStreamSynchronize(pipeline->stream);
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTPosePipelineStageRegionReadback;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        return OVTRTStatusCUDARuntimeFailure;
    }
    cudaStatus = pipeline->api.cudaMemcpy(
        &pipeline->selectedRegionCount,
        pipeline->deviceRegionCount,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost
    );
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTPosePipelineStageRegionReadback;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        return OVTRTStatusCUDARuntimeFailure;
    }
    result->compactDeviceToHostCopyCount = 1;
    result->compactReadbackByteCount = sizeof(uint32_t);
    if (
        pipeline->selectedRegionCount >
        pipeline->configuration.maximumRegionCount
    ) {
        result->failureStage =
            OVTRTPosePipelineStageRegionReadback;
        return OVTRTStatusPreprocessingFailure;
    }
    result->selectedRegionCount = pipeline->selectedRegionCount;
    if (pipeline->selectedRegionCount == 0) {
        return OVTRTStatusSuccess;
    }
    size_t regionBytes =
        pipeline->selectedRegionCount * sizeof(OVTRTPoseRegion);
    cudaStatus = pipeline->api.cudaMemcpy(
        regions,
        pipeline->deviceRegions,
        regionBytes,
        cudaMemcpyDeviceToHost
    );
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTPosePipelineStageRegionReadback;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        return OVTRTStatusCUDARuntimeFailure;
    }
    ++result->compactDeviceToHostCopyCount;
    result->compactReadbackByteCount += regionBytes;
    unsigned int orientationValue =
        static_cast<unsigned int>(orientation);
    unsigned int applySRGBTransfer =
        pipeline->configuration.applySRGBTransfer;
    unsigned int sampleBitShift =
        pipeline->configuration.sourceWordLayout ==
            OVTRTRG10WordLayoutMostSignificantBits
        ? 6U
        : 0U;
    void const *sourceAddress = source->deviceAddress;
    void *affineArguments[] = {
        &sourceAddress,
        &pipeline->devicePoseInput,
        &pipeline->deviceConfiguration,
        &pipeline->deviceRegions,
        &pipeline->selectedRegionCount,
        &pipeline->configuration.sourceWidth,
        &pipeline->configuration.sourceHeight,
        &pipeline->configuration.sourceBytesPerRow,
        &poseWidth,
        &poseHeight,
        &orientationValue,
        &sampleBitShift,
        &applySRGBTransfer
    };
    unsigned int blockWidth = 16;
    unsigned int blockHeight = 16;
    driverStatus = pipeline->api.cuLaunchKernel(
        pipeline->regionAffineKernel,
        (poseWidth + blockWidth - 1U) / blockWidth,
        (poseHeight + blockHeight - 1U) / blockHeight,
        pipeline->selectedRegionCount,
        blockWidth,
        blockHeight,
        1,
        0,
        reinterpret_cast<CUstream>(pipeline->stream),
        affineArguments,
        nullptr
    );
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage =
            OVTRTPosePipelineStageRegionAffine;
        result->cudaDriverErrorCode =
            static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }
    result->regionAffineKernelLaunchCount = 1;
    cudaStatus = pipeline->api.cudaStreamSynchronize(pipeline->stream);
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTPosePipelineStagePoseSynchronization;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        return OVTRTStatusCUDARuntimeFailure;
    }
    uint64_t elementCount =
        static_cast<uint64_t>(pipeline->selectedRegionCount) *
        3ULL * poseWidth * poseHeight;
    poseInput->deviceAddress = pipeline->devicePoseInput;
    poseInput->byteCount = elementCount * sizeof(float);
    poseInput->elementCount = elementCount;
    poseInput->width = poseWidth;
    poseInput->height = poseHeight;
    poseInput->channelCount = 3;
    poseInput->layout = OVTRTTensorLayoutNCHW;
    poseInput->channelOrder = OVTRTTensorChannelOrderRGB;
    result->poseInputByteCount = poseInput->byteCount;
    result->explicitFrameDeviceAllocationCount = 0;
    return OVTRTStatusSuccess;
#else
    result->failureStage = OVTRTPosePipelineStageLibraryOpen;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_pose_pipeline_decode_simcc(
    OVTRTPosePipeline *pipeline,
    void const *simCCXDeviceAddress,
    uint64_t simCCXElementCount,
    void const *simCCYDeviceAddress,
    uint64_t simCCYElementCount,
    OVTRTPoseJoint *joints,
    uint64_t jointCapacity,
    OVTRTPosePipelineResult *result
) {
    if (
        pipeline == nullptr ||
        simCCXDeviceAddress == nullptr ||
        simCCYDeviceAddress == nullptr ||
        joints == nullptr ||
        result == nullptr
    ) {
        return OVTRTStatusInvalidArgument;
    }
    resetResult(result, &pipeline->configuration);
#if OVTRT_HAS_POSE_PIPELINE
    uint64_t compactJointCount =
        static_cast<uint64_t>(pipeline->selectedRegionCount) *
        pipeline->configuration.jointCount;
    uint32_t xBinCount =
        pipeline->configuration.poseInputWidth * 2U;
    uint32_t yBinCount =
        pipeline->configuration.poseInputHeight * 2U;
    if (
        pipeline->selectedRegionCount == 0 ||
        jointCapacity < compactJointCount ||
        simCCXElementCount != compactJointCount * xBinCount ||
        simCCYElementCount != compactJointCount * yBinCount
    ) {
        result->failureStage =
            OVTRTPosePipelineStageConfiguration;
        return OVTRTStatusInvalidArgument;
    }
    void const *simCCX = simCCXDeviceAddress;
    void const *simCCY = simCCYDeviceAddress;
    uint32_t jointCount = pipeline->configuration.jointCount;
    uint32_t poseWidth = pipeline->configuration.poseInputWidth;
    uint32_t poseHeight = pipeline->configuration.poseInputHeight;
    void *arguments[] = {
        &simCCX,
        &simCCY,
        &pipeline->deviceRegions,
        &pipeline->deviceJoints,
        &pipeline->selectedRegionCount,
        &jointCount,
        &xBinCount,
        &yBinCount,
        &poseWidth,
        &poseHeight,
        &pipeline->orientedWidth,
        &pipeline->orientedHeight
    };
    uint32_t threadCount = 128;
    CUresult driverStatus = pipeline->api.cuLaunchKernel(
        pipeline->simCCDecodeKernel,
        static_cast<unsigned int>(
            (compactJointCount + threadCount - 1U) / threadCount
        ),
        1,
        1,
        threadCount,
        1,
        1,
        0,
        reinterpret_cast<CUstream>(pipeline->stream),
        arguments,
        nullptr
    );
    if (driverStatus != CUDA_SUCCESS) {
        result->failureStage =
            OVTRTPosePipelineStageSimCCDecode;
        result->cudaDriverErrorCode =
            static_cast<int32_t>(driverStatus);
        return OVTRTStatusCUDADriverFailure;
    }
    result->simCCDecodeKernelLaunchCount = 1;
    cudaError_t cudaStatus =
        pipeline->api.cudaStreamSynchronize(pipeline->stream);
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTPosePipelineStageJointReadback;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        return OVTRTStatusCUDARuntimeFailure;
    }
    size_t jointBytes =
        static_cast<size_t>(compactJointCount * sizeof(OVTRTPoseJoint));
    cudaStatus = pipeline->api.cudaMemcpy(
        joints,
        pipeline->deviceJoints,
        jointBytes,
        cudaMemcpyDeviceToHost
    );
    if (cudaStatus != cudaSuccess) {
        result->failureStage =
            OVTRTPosePipelineStageJointReadback;
        result->cudaErrorCode = static_cast<int32_t>(cudaStatus);
        return OVTRTStatusCUDARuntimeFailure;
    }
    result->selectedRegionCount = pipeline->selectedRegionCount;
    result->compactDeviceToHostCopyCount = 1;
    result->compactReadbackByteCount = jointBytes;
    result->explicitFrameDeviceAllocationCount = 0;
    return OVTRTStatusSuccess;
#else
    result->failureStage = OVTRTPosePipelineStageLibraryOpen;
    return OVTRTStatusUnavailable;
#endif
}

OVTRTStatus ovtrt_pose_pipeline_destroy(
    OVTRTPosePipeline **pipeline,
    OVTRTPosePipelineResult *result
) {
    if (pipeline == nullptr || result == nullptr) {
        return OVTRTStatusInvalidArgument;
    }
    OVTRTPosePipeline *owner = *pipeline;
    if (owner == nullptr) {
        resetResult(result, nullptr);
        return OVTRTStatusSuccess;
    }
    resetResult(result, &owner->configuration);
#if OVTRT_HAS_POSE_PIPELINE
    OVTRTStatus status = cleanup(owner, result);
    if (status != OVTRTStatusSuccess) {
        return status;
    }
    delete owner;
    *pipeline = nullptr;
    return OVTRTStatusSuccess;
#else
    delete owner;
    *pipeline = nullptr;
    return OVTRTStatusSuccess;
#endif
}
