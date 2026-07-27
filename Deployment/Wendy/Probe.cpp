#include "CTensorRTShim.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr float MAXIMUM_DIFFERENCE = 0.00002F;

struct RGB {
    float red;
    float green;
    float blue;
};

int reflected(int coordinate, int limit) {
    while (coordinate < 0 || coordinate >= limit) {
        coordinate = coordinate < 0
            ? -coordinate
            : 2 * limit - 2 - coordinate;
    }
    return coordinate;
}

float rawValue(
    std::vector<uint8_t> const &source,
    OVTRTRG10PreprocessingConfiguration const &configuration,
    int x,
    int y
) {
    x = reflected(x, static_cast<int>(configuration.sourceWidth));
    y = reflected(y, static_cast<int>(configuration.sourceHeight));
    size_t offset =
        static_cast<size_t>(y) * configuration.sourceBytesPerRow +
        static_cast<size_t>(x) * 2;
    uint32_t value =
        static_cast<uint32_t>(source[offset]) |
        (static_cast<uint32_t>(source[offset + 1]) << 8U);
    uint32_t sampleBitShift =
        configuration.wordLayout ==
            OVTRTRG10WordLayoutMostSignificantBits
        ? 6U
        : 0U;
    value = (value >> sampleBitShift) & 1023U;
    uint32_t site =
        ((static_cast<uint32_t>(y) & 1U) << 1U) |
        (static_cast<uint32_t>(x) & 1U);
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
    return std::clamp(
        (static_cast<float>(value) - blackLevels[site]) /
            (configuration.whiteLevel - blackLevels[site]) *
            gains[site],
        0.0F,
        1.0F
    );
}

RGB demosaiced(
    std::vector<uint8_t> const &source,
    OVTRTRG10PreprocessingConfiguration const &configuration,
    int x,
    int y
) {
    bool evenX = (x & 1) == 0;
    bool evenY = (y & 1) == 0;
    float center = rawValue(source, configuration, x, y);
    if (evenX && evenY) {
        return RGB{
            center,
            0.25F * (
                rawValue(source, configuration, x - 1, y) +
                rawValue(source, configuration, x + 1, y) +
                rawValue(source, configuration, x, y - 1) +
                rawValue(source, configuration, x, y + 1)
            ),
            0.25F * (
                rawValue(source, configuration, x - 1, y - 1) +
                rawValue(source, configuration, x + 1, y - 1) +
                rawValue(source, configuration, x - 1, y + 1) +
                rawValue(source, configuration, x + 1, y + 1)
            )
        };
    }
    if (!evenX && !evenY) {
        return RGB{
            0.25F * (
                rawValue(source, configuration, x - 1, y - 1) +
                rawValue(source, configuration, x + 1, y - 1) +
                rawValue(source, configuration, x - 1, y + 1) +
                rawValue(source, configuration, x + 1, y + 1)
            ),
            0.25F * (
                rawValue(source, configuration, x - 1, y) +
                rawValue(source, configuration, x + 1, y) +
                rawValue(source, configuration, x, y - 1) +
                rawValue(source, configuration, x, y + 1)
            ),
            center
        };
    }
    if (!evenX && evenY) {
        return RGB{
            0.5F * (
                rawValue(source, configuration, x - 1, y) +
                rawValue(source, configuration, x + 1, y)
            ),
            center,
            0.5F * (
                rawValue(source, configuration, x, y - 1) +
                rawValue(source, configuration, x, y + 1)
            )
        };
    }
    return RGB{
        0.5F * (
            rawValue(source, configuration, x, y - 1) +
            rawValue(source, configuration, x, y + 1)
        ),
        center,
        0.5F * (
            rawValue(source, configuration, x - 1, y) +
            rawValue(source, configuration, x + 1, y)
        )
    };
}

void sourceCoordinate(
    float orientedX,
    float orientedY,
    OVTRTRG10PreprocessingConfiguration const &configuration,
    OVTRTRG10Orientation orientation,
    float &sourceX,
    float &sourceY
) {
    float maximumX =
        static_cast<float>(configuration.sourceWidth - 1U);
    float maximumY =
        static_cast<float>(configuration.sourceHeight - 1U);
    switch (orientation) {
    case OVTRTRG10OrientationUp:
        sourceX = orientedX;
        sourceY = orientedY;
        break;
    case OVTRTRG10OrientationUpMirrored:
        sourceX = maximumX - orientedX;
        sourceY = orientedY;
        break;
    case OVTRTRG10OrientationDown:
        sourceX = maximumX - orientedX;
        sourceY = maximumY - orientedY;
        break;
    case OVTRTRG10OrientationDownMirrored:
        sourceX = orientedX;
        sourceY = maximumY - orientedY;
        break;
    case OVTRTRG10OrientationLeftMirrored:
        sourceX = orientedY;
        sourceY = orientedX;
        break;
    case OVTRTRG10OrientationRight:
        sourceX = orientedY;
        sourceY = maximumY - orientedX;
        break;
    case OVTRTRG10OrientationRightMirrored:
        sourceX = maximumX - orientedY;
        sourceY = maximumY - orientedX;
        break;
    case OVTRTRG10OrientationLeft:
        sourceX = maximumX - orientedY;
        sourceY = orientedX;
        break;
    }
}

RGB interpolated(
    std::vector<uint8_t> const &source,
    OVTRTRG10PreprocessingConfiguration const &configuration,
    float x,
    float y
) {
    x = std::clamp(
        x,
        0.0F,
        static_cast<float>(configuration.sourceWidth - 1U)
    );
    y = std::clamp(
        y,
        0.0F,
        static_cast<float>(configuration.sourceHeight - 1U)
    );
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float fractionX = x - static_cast<float>(x0);
    float fractionY = y - static_cast<float>(y0);
    RGB topLeft = demosaiced(source, configuration, x0, y0);
    RGB topRight = demosaiced(source, configuration, x1, y0);
    RGB bottomLeft = demosaiced(source, configuration, x0, y1);
    RGB bottomRight = demosaiced(source, configuration, x1, y1);
    auto interpolateChannel = [=](
        float topLeftValue,
        float topRightValue,
        float bottomLeftValue,
        float bottomRightValue
    ) {
        float top = topLeftValue +
            (topRightValue - topLeftValue) * fractionX;
        float bottom = bottomLeftValue +
            (bottomRightValue - bottomLeftValue) * fractionX;
        return top + (bottom - top) * fractionY;
    };
    return RGB{
        interpolateChannel(
            topLeft.red,
            topRight.red,
            bottomLeft.red,
            bottomRight.red
        ),
        interpolateChannel(
            topLeft.green,
            topRight.green,
            bottomLeft.green,
            bottomRight.green
        ),
        interpolateChannel(
            topLeft.blue,
            topRight.blue,
            bottomLeft.blue,
            bottomRight.blue
        )
    };
}

float srgb(float linear) {
    linear = std::clamp(linear, 0.0F, 1.0F);
    if (linear <= 0.0031308F) {
        return linear * 12.92F;
    }
    return 1.055F * std::pow(linear, 1.0F / 2.4F) - 0.055F;
}

std::vector<float> referenceOutput(
    std::vector<uint8_t> const &source,
    OVTRTRG10PreprocessingConfiguration const &configuration,
    OVTRTRG10Orientation orientation
) {
    uint64_t pixelCount =
        static_cast<uint64_t>(configuration.outputWidth) *
        configuration.outputHeight;
    std::vector<float> output(pixelCount * 3ULL);
    bool swapsDimensions =
        orientation >= OVTRTRG10OrientationLeftMirrored;
    float orientedWidth = static_cast<float>(
        swapsDimensions
            ? configuration.sourceHeight
            : configuration.sourceWidth
    );
    float orientedHeight = static_cast<float>(
        swapsDimensions
            ? configuration.sourceWidth
            : configuration.sourceHeight
    );
    float scaleX =
        static_cast<float>(configuration.outputWidth) /
        orientedWidth;
    float scaleY =
        static_cast<float>(configuration.outputHeight) /
        orientedHeight;

    for (uint32_t outputY = 0;
         outputY < configuration.outputHeight;
         ++outputY) {
        for (uint32_t outputX = 0;
             outputX < configuration.outputWidth;
             ++outputX) {
            float sampleX = 0.0F;
            float sampleY = 0.0F;
            bool letterbox = false;
            if (
                configuration.resizePolicy ==
                OVTRTRG10ResizePolicyScaleFill
            ) {
                sampleX =
                    (static_cast<float>(outputX) + 0.5F) /
                    scaleX - 0.5F;
                sampleY =
                    (static_cast<float>(outputY) + 0.5F) /
                    scaleY - 0.5F;
            } else {
                float scale =
                    configuration.resizePolicy ==
                        OVTRTRG10ResizePolicyScaleFit
                    ? std::min(scaleX, scaleY)
                    : std::max(scaleX, scaleY);
                float offsetX =
                    (static_cast<float>(configuration.outputWidth) -
                        orientedWidth * scale) * 0.5F;
                float offsetY =
                    (static_cast<float>(configuration.outputHeight) -
                        orientedHeight * scale) * 0.5F;
                float centerX = static_cast<float>(outputX) + 0.5F;
                float centerY = static_cast<float>(outputY) + 0.5F;
                letterbox =
                    configuration.resizePolicy ==
                        OVTRTRG10ResizePolicyScaleFit &&
                    (
                        centerX < offsetX ||
                        centerX >=
                            static_cast<float>(
                                configuration.outputWidth
                            ) - offsetX ||
                        centerY < offsetY ||
                        centerY >=
                            static_cast<float>(
                                configuration.outputHeight
                            ) - offsetY
                    );
                sampleX = (centerX - offsetX) / scale - 0.5F;
                sampleY = (centerY - offsetY) / scale - 0.5F;
            }

            RGB color{
                configuration.letterboxR,
                configuration.letterboxG,
                configuration.letterboxB
            };
            if (!letterbox) {
                float sourceX = 0.0F;
                float sourceY = 0.0F;
                sourceCoordinate(
                    sampleX,
                    sampleY,
                    configuration,
                    orientation,
                    sourceX,
                    sourceY
                );
                RGB camera = interpolated(
                    source,
                    configuration,
                    sourceX,
                    sourceY
                );
                color = RGB{
                    std::clamp(
                        configuration.colorMatrix00 * camera.red +
                        configuration.colorMatrix01 * camera.green +
                        configuration.colorMatrix02 * camera.blue,
                        0.0F,
                        1.0F
                    ),
                    std::clamp(
                        configuration.colorMatrix10 * camera.red +
                        configuration.colorMatrix11 * camera.green +
                        configuration.colorMatrix12 * camera.blue,
                        0.0F,
                        1.0F
                    ),
                    std::clamp(
                        configuration.colorMatrix20 * camera.red +
                        configuration.colorMatrix21 * camera.green +
                        configuration.colorMatrix22 * camera.blue,
                        0.0F,
                        1.0F
                    )
                };
                if (configuration.applySRGBTransfer != 0) {
                    color.red = srgb(color.red);
                    color.green = srgb(color.green);
                    color.blue = srgb(color.blue);
                }
            }
            float channels[] = {
                color.red * configuration.normalizationScaleR +
                    configuration.normalizationBiasR,
                color.green * configuration.normalizationScaleG +
                    configuration.normalizationBiasG,
                color.blue * configuration.normalizationScaleB +
                    configuration.normalizationBiasB
            };
            if (
                configuration.channelOrder ==
                OVTRTTensorChannelOrderBGR
            ) {
                std::swap(channels[0], channels[2]);
            }
            uint64_t pixelIndex =
                static_cast<uint64_t>(outputY) *
                configuration.outputWidth + outputX;
            for (uint64_t channel = 0; channel < 3; ++channel) {
                uint64_t index =
                    configuration.tensorLayout ==
                        OVTRTTensorLayoutNCHW
                    ? channel * pixelCount + pixelIndex
                    : pixelIndex * 3ULL + channel;
                output[index] = channels[channel];
            }
        }
    }
    return output;
}

OVTRTRG10PreprocessingConfiguration configuration(
    uint32_t sourceWidth,
    uint32_t sourceHeight,
    uint32_t outputWidth,
    uint32_t outputHeight,
    OVTRTRG10ResizePolicy resizePolicy
) {
    OVTRTRG10PreprocessingConfiguration value{};
    value.sourceWidth = sourceWidth;
    value.sourceHeight = sourceHeight;
    value.sourceBytesPerRow = sourceWidth * 2U;
    value.sourceByteCount =
        static_cast<uint64_t>(sourceWidth) * sourceHeight * 2ULL;
    value.outputWidth = outputWidth;
    value.outputHeight = outputHeight;
    value.resizePolicy = resizePolicy;
    value.tensorLayout = OVTRTTensorLayoutNCHW;
    value.channelOrder = OVTRTTensorChannelOrderRGB;
    value.wordLayout = OVTRTRG10WordLayoutLeastSignificantBits;
    value.whiteLevel = 1023.0F;
    value.gainR = 1.0F;
    value.gainGreenR = 1.0F;
    value.gainGreenB = 1.0F;
    value.gainB = 1.0F;
    value.colorMatrix00 = 1.0F;
    value.colorMatrix11 = 1.0F;
    value.colorMatrix22 = 1.0F;
    value.letterboxR = 0.125F;
    value.letterboxG = 0.25F;
    value.letterboxB = 0.5F;
    value.normalizationScaleR = 1.0F;
    value.normalizationScaleG = 1.0F;
    value.normalizationScaleB = 1.0F;
    return value;
}

std::vector<uint8_t> fixture(
    OVTRTRG10PreprocessingConfiguration const &configuration
) {
    std::vector<uint8_t> source(
        static_cast<size_t>(configuration.sourceByteCount)
    );
    for (uint32_t y = 0; y < configuration.sourceHeight; ++y) {
        for (uint32_t x = 0; x < configuration.sourceWidth; ++x) {
            uint16_t value = static_cast<uint16_t>(
                (
                    x * 97U +
                    y * 53U +
                    ((y & 1U) << 1U | (x & 1U)) * 181U
                ) & 1023U
            );
            if (
                configuration.wordLayout ==
                OVTRTRG10WordLayoutMostSignificantBits
            ) {
                value = static_cast<uint16_t>(
                    (value << 6U) | (value >> 4U)
                );
            }
            size_t offset =
                static_cast<size_t>(y) *
                configuration.sourceBytesPerRow +
                static_cast<size_t>(x) * 2;
            source[offset] = static_cast<uint8_t>(value & 0xFFU);
            source[offset + 1] =
                static_cast<uint8_t>(value >> 8U);
        }
    }
    return source;
}

bool verifiedFrame(
    OVTRTRG10Preprocessor *preprocessor,
    std::vector<uint8_t> const &source,
    OVTRTRG10PreprocessingConfiguration const &configuration,
    OVTRTRG10Orientation orientation,
    float &maximumDifference
) {
    OVTRTRG10PreprocessingResult submission{};
    if (
        ovtrt_rg10_preprocessor_submit(
            preprocessor,
            source.data(),
            source.size(),
            orientation,
            &submission
        ) != OVTRTStatusSuccess
    ) {
        std::fprintf(
            stderr,
            "preprocess_submit_failed stage=%d cuda=%d driver=%d "
            "cleanup_stage=%d\n",
            static_cast<int>(submission.failureStage),
            submission.cudaErrorCode,
            submission.cudaDriverErrorCode,
            static_cast<int>(submission.cleanupFailureStage)
        );
        return false;
    }
    OVTRTRG10PreprocessingResult completion{};
    if (
        ovtrt_rg10_preprocessor_wait(
            preprocessor,
            &completion
        ) != OVTRTStatusSuccess
    ) {
        std::fprintf(
            stderr,
            "preprocess_wait_failed stage=%d cuda=%d\n",
            static_cast<int>(completion.failureStage),
            completion.cudaErrorCode
        );
        return false;
    }
    OVTRTDeviceTensorView tensor{};
    if (
        ovtrt_rg10_preprocessor_output(
            preprocessor,
            &tensor
        ) != OVTRTStatusSuccess ||
        tensor.deviceAddress == nullptr ||
        tensor.elementCount != completion.outputElementCount ||
        tensor.byteCount != tensor.elementCount * sizeof(float) ||
        tensor.width != configuration.outputWidth ||
        tensor.height != configuration.outputHeight ||
        tensor.channelCount != 3 ||
        tensor.layout != configuration.tensorLayout ||
        tensor.channelOrder != configuration.channelOrder
    ) {
        return false;
    }
    std::vector<float> actual(
        static_cast<size_t>(completion.outputElementCount)
    );
    OVTRTRG10PreprocessingResult readback{};
    if (
        ovtrt_rg10_preprocessor_copy_output(
            preprocessor,
            actual.data(),
            actual.size(),
            &readback
        ) != OVTRTStatusSuccess
    ) {
        return false;
    }
    std::vector<float> expected = referenceOutput(
        source,
        configuration,
        orientation
    );
    for (size_t index = 0; index < expected.size(); ++index) {
        maximumDifference = std::max(
            maximumDifference,
            std::abs(actual[index] - expected[index])
        );
    }
    return
        submission.fullFrameHostToDeviceCopyCount == 1 &&
        submission.kernelLaunchCount == 1 &&
        submission
                .explicitFrameSizedDeviceAllocationCountAfterPreparation ==
            0 &&
        submission.sourceReadCompleted == 1 &&
        submission.sourceReadFencePassed == 1 &&
        completion.outputReadyEventPassed == 1 &&
        readback.deviceToHostVerificationCopyCount == 1 &&
        maximumDifference <= MAXIMUM_DIFFERENCE;
}

bool verifiedGoldenFrame(float &maximumDifference) {
    auto testConfiguration = configuration(
        2,
        2,
        2,
        2,
        OVTRTRG10ResizePolicyScaleFill
    );
    std::vector<uint8_t> source{
        0xFF, 0x03,
        0x00, 0x02,
        0x00, 0x02,
        0x00, 0x00
    };
    OVTRTRG10Preprocessor *preprocessor = nullptr;
    OVTRTRG10PreprocessingResult preparation{};
    if (
        ovtrt_rg10_preprocessor_create(
            &testConfiguration,
            &preprocessor,
            &preparation
        ) != OVTRTStatusSuccess
    ) {
        return false;
    }
    OVTRTRG10PreprocessingResult submission{};
    OVTRTRG10PreprocessingResult completion{};
    bool passed =
        ovtrt_rg10_preprocessor_submit(
            preprocessor,
            source.data(),
            source.size(),
            OVTRTRG10OrientationUp,
            &submission
        ) == OVTRTStatusSuccess &&
        ovtrt_rg10_preprocessor_wait(
            preprocessor,
            &completion
        ) == OVTRTStatusSuccess;
    std::vector<float> actual(12);
    OVTRTRG10PreprocessingResult readback{};
    passed =
        passed &&
        ovtrt_rg10_preprocessor_copy_output(
            preprocessor,
            actual.data(),
            actual.size(),
            &readback
        ) == OVTRTStatusSuccess;
    float expected[] = {
        1.0F, 1.0F, 1.0F, 1.0F,
        512.0F / 1023.0F,
        512.0F / 1023.0F,
        512.0F / 1023.0F,
        512.0F / 1023.0F,
        0.0F, 0.0F, 0.0F, 0.0F
    };
    if (passed) {
        for (size_t index = 0; index < actual.size(); ++index) {
            maximumDifference = std::max(
                maximumDifference,
                std::abs(actual[index] - expected[index])
            );
        }
        passed = maximumDifference <= MAXIMUM_DIFFERENCE;
    }
    OVTRTRG10PreprocessingResult cleanup{};
    passed =
        ovtrt_rg10_preprocessor_destroy(
            &preprocessor,
            &cleanup
        ) == OVTRTStatusSuccess &&
        preprocessor == nullptr &&
        passed;
    return passed;
}

double percentile(
    std::vector<double> values,
    double fraction
) {
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(
        std::ceil(fraction * static_cast<double>(values.size()))
    ) - 1;
    return values[index];
}

bool destroyed(
    OVTRTRG10Preprocessor *&preprocessor
) {
    OVTRTRG10PreprocessingResult result{};
    return
        ovtrt_rg10_preprocessor_destroy(
            &preprocessor,
            &result
        ) == OVTRTStatusSuccess &&
        preprocessor == nullptr &&
        result.cleanupFailureStage ==
            OVTRTRG10PreprocessingStageNone;
}

bool verifiedRetryableCleanup() {
    auto testConfiguration = configuration(
        8,
        6,
        7,
        5,
        OVTRTRG10ResizePolicyScaleFit
    );
    OVTRTRG10Preprocessor *preprocessor = nullptr;
    OVTRTRG10PreprocessingResult preparation{};
    if (
        ovtrt_rg10_preprocessor_create(
            &testConfiguration,
            &preprocessor,
            &preparation
        ) != OVTRTStatusSuccess ||
        preprocessor == nullptr
    ) {
        return false;
    }
    if (
        ovtrt_rg10_preprocessor_test_fail_next_cleanup_synchronization(
            preprocessor
        ) != OVTRTStatusSuccess
    ) {
        if (!destroyed(preprocessor)) {
            std::fprintf(
                stderr,
                "cleanup_after_test_hook_failure_failed\n"
            );
        }
        return false;
    }

    OVTRTRG10PreprocessingResult injectedFailure{};
    OVTRTStatus firstStatus = ovtrt_rg10_preprocessor_destroy(
        &preprocessor,
        &injectedFailure
    );
    bool retainedAfterFailure =
        firstStatus == OVTRTStatusPreprocessingFailure &&
        preprocessor != nullptr &&
        injectedFailure.cleanupFailureStage ==
            OVTRTRG10PreprocessingStageStreamSynchronization &&
        injectedFailure.cleanupCUDAErrorCode != 0;
    bool retryPassed = destroyed(preprocessor);
    return retainedAfterFailure && retryPassed;
}

}  // namespace

int main() {
    if (ovtrt_probe(nullptr) != OVTRTStatusInvalidArgument) {
        return 20;
    }
    OVTRTProbeResult probe{};
    if (ovtrt_probe(&probe) != OVTRTStatusSuccess) {
        return 21;
    }

    OVTRTRuntime *runtime = nullptr;
    if (
        ovtrt_runtime_create(&runtime) != OVTRTStatusSuccess ||
        runtime == nullptr
    ) {
        return 22;
    }
    ovtrt_runtime_destroy(runtime);

    OVTRTCUDATransferProbeConfiguration transferConfiguration{
        1920ULL * 1080ULL * 2ULL,
        10,
        100
    };
    OVTRTCUDATransferProbeResult transfer{};
    if (
        ovtrt_cuda_transfer_probe(
            &transferConfiguration,
            &transfer
        ) != OVTRTStatusSuccess
    ) {
        return 23;
    }

    float maximumDifference = 0.0F;
    uint32_t verifiedCaseCount = 0;
    for (int layoutValue = 0; layoutValue < 2; ++layoutValue) {
      for (int policyValue = 0; policyValue < 3; ++policyValue) {
        auto policy = static_cast<OVTRTRG10ResizePolicy>(policyValue);
        auto testConfiguration = configuration(8, 6, 7, 5, policy);
        testConfiguration.wordLayout =
            static_cast<OVTRTRG10WordLayout>(layoutValue);
        if (policy == OVTRTRG10ResizePolicyScaleFit) {
            testConfiguration.blackLevelR = 64.0F;
            testConfiguration.blackLevelGreenR = 60.0F;
            testConfiguration.blackLevelGreenB = 62.0F;
            testConfiguration.blackLevelB = 66.0F;
            testConfiguration.gainR = 1.125F;
            testConfiguration.gainGreenR = 1.0F;
            testConfiguration.gainGreenB = 0.975F;
            testConfiguration.gainB = 1.25F;
            testConfiguration.colorMatrix00 = 1.05F;
            testConfiguration.colorMatrix01 = -0.025F;
            testConfiguration.colorMatrix02 = -0.025F;
            testConfiguration.colorMatrix10 = -0.01F;
            testConfiguration.colorMatrix11 = 1.02F;
            testConfiguration.colorMatrix12 = -0.01F;
            testConfiguration.colorMatrix20 = -0.02F;
            testConfiguration.colorMatrix21 = -0.02F;
            testConfiguration.colorMatrix22 = 1.04F;
            testConfiguration.applySRGBTransfer = 1;
        } else if (
            policy == OVTRTRG10ResizePolicyCenterCrop
        ) {
            testConfiguration.tensorLayout =
                OVTRTTensorLayoutNHWC;
            testConfiguration.channelOrder =
                OVTRTTensorChannelOrderBGR;
            testConfiguration.normalizationScaleR = 2.0F;
            testConfiguration.normalizationScaleG = 3.0F;
            testConfiguration.normalizationScaleB = 4.0F;
            testConfiguration.normalizationBiasR = -1.0F;
            testConfiguration.normalizationBiasG = -0.5F;
            testConfiguration.normalizationBiasB = 0.25F;
        }
        std::vector<uint8_t> source = fixture(testConfiguration);
        OVTRTRG10Preprocessor *preprocessor = nullptr;
        OVTRTRG10PreprocessingResult preparation{};
        OVTRTStatus createStatus = ovtrt_rg10_preprocessor_create(
            &testConfiguration,
            &preprocessor,
            &preparation
        );
        if (
            createStatus != OVTRTStatusSuccess ||
            preprocessor == nullptr ||
            preparation.frameSizedDeviceAllocationCount != 2 ||
            preparation.nvrtcCompilationCount != 1
        ) {
            std::fprintf(
                stderr,
                "preprocess_create_failed status=%d stage=%d "
                "cuda=%d driver=%d nvrtc=%d cleanup_stage=%d\n",
                static_cast<int>(createStatus),
                static_cast<int>(preparation.failureStage),
                preparation.cudaErrorCode,
                preparation.cudaDriverErrorCode,
                preparation.nvrtcErrorCode,
                static_cast<int>(preparation.cleanupFailureStage)
            );
            return 30;
        }
        int firstOrientation = 1;
        int lastOrientation = 8;
        for (
            int orientationValue = firstOrientation;
            orientationValue <= lastOrientation;
            ++orientationValue
        ) {
            if (
                !verifiedFrame(
                    preprocessor,
                    source,
                    testConfiguration,
                    static_cast<OVTRTRG10Orientation>(
                        orientationValue
                    ),
                    maximumDifference
                )
            ) {
                return 31;
            }
            ++verifiedCaseCount;
        }
        if (!destroyed(preprocessor)) {
            return 32;
        }
      }
    }
    if (!verifiedGoldenFrame(maximumDifference)) {
        return 38;
    }
    ++verifiedCaseCount;

    auto performanceConfiguration = configuration(
        1920,
        1080,
        256,
        256,
        OVTRTRG10ResizePolicyScaleFit
    );
    std::vector<uint8_t> performanceSource =
        fixture(performanceConfiguration);
    OVTRTRG10Preprocessor *performancePreprocessor = nullptr;
    OVTRTRG10PreprocessingResult performancePreparation{};
    if (
        ovtrt_rg10_preprocessor_create(
            &performanceConfiguration,
            &performancePreprocessor,
            &performancePreparation
        ) != OVTRTStatusSuccess
    ) {
        return 33;
    }
    std::vector<double> timings;
    std::vector<double> wallTimings;
    timings.reserve(100);
    wallTimings.reserve(100);
    for (uint32_t iteration = 0; iteration < 110; ++iteration) {
        auto wallStart = std::chrono::steady_clock::now();
        OVTRTRG10PreprocessingResult submission{};
        if (
            ovtrt_rg10_preprocessor_submit(
                performancePreprocessor,
                performanceSource.data(),
                performanceSource.size(),
                OVTRTRG10OrientationUp,
                &submission
            ) != OVTRTStatusSuccess
        ) {
            return 34;
        }
        OVTRTRG10PreprocessingResult completion{};
        if (
            ovtrt_rg10_preprocessor_wait(
                performancePreprocessor,
                &completion
            ) != OVTRTStatusSuccess
        ) {
            return 35;
        }
        if (
            submission.fullFrameHostToDeviceCopyCount != 1 ||
            submission.kernelLaunchCount != 1 ||
            submission
                    .explicitFrameSizedDeviceAllocationCountAfterPreparation !=
                0
        ) {
            return 36;
        }
        if (iteration >= 10) {
            timings.push_back(completion.gpuMilliseconds);
            double wallMilliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - wallStart
                ).count();
            wallTimings.push_back(wallMilliseconds);
        }
    }
    double preprocessingP50 = percentile(timings, 0.50);
    double preprocessingP95 = percentile(timings, 0.95);
    double wallP50 = percentile(wallTimings, 0.50);
    double wallP95 = percentile(wallTimings, 0.95);
    if (!destroyed(performancePreprocessor)) {
        return 37;
    }
    if (!verifiedRetryableCleanup()) {
        return 39;
    }

    std::printf(
        "{\"status\":\"available\","
        "\"tensorRTVersion\":%d,"
        "\"cudaRuntimeVersion\":%d,"
        "\"cudaDriverVersion\":%d,"
        "\"cudaDeviceCount\":%d,"
        "\"runtimeLifecycle\":\"passed\","
        "\"transfer\":{\"p50Milliseconds\":%.6f,"
        "\"p95Milliseconds\":%.6f,\"contract\":\"passed\"},"
        "\"rg10Preprocessing\":{\"verifiedCases\":%u,"
        "\"maximumAbsoluteDifference\":%.8f,"
        "\"fullFrameHostToDeviceCopiesPerFrame\":1,"
        "\"kernelLaunchesPerFrame\":1,"
        "\"explicitFrameSizedDeviceAllocationsAfterPreparation\":0,"
        "\"p50Milliseconds\":%.6f,"
        "\"p95Milliseconds\":%.6f,"
        "\"endToEndP50Milliseconds\":%.6f,"
        "\"endToEndP95Milliseconds\":%.6f,"
        "\"contract\":\"passed\"},"
        "\"retryableCleanup\":\"passed\"}\n",
        probe.tensorRTVersion,
        probe.cudaRuntimeVersion,
        probe.cudaDriverVersion,
        probe.cudaDeviceCount,
        transfer.p50Milliseconds,
        transfer.p95Milliseconds,
        verifiedCaseCount,
        maximumDifference,
        preprocessingP50,
        preprocessingP95,
        wallP50,
        wallP95
    );
    return 0;
}
