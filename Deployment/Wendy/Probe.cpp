#include "CTensorRTShim.h"

#include <cstdio>

int main() {
    if (ovtrt_probe(nullptr) != OVTRTStatusInvalidArgument) {
        std::fputs("probe_null_contract_failed\n", stderr);
        return 20;
    }
    OVTRTCUDATransferProbeConfiguration nullContractConfiguration{
        1,
        0,
        1
    };
    OVTRTCUDATransferProbeResult nullContractResult{};
    if (
        ovtrt_cuda_transfer_probe(
            nullptr,
            &nullContractResult
        ) != OVTRTStatusInvalidArgument ||
        ovtrt_cuda_transfer_probe(
            &nullContractConfiguration,
            nullptr
        ) != OVTRTStatusInvalidArgument
    ) {
        std::fputs(
            "transfer_probe_null_contract_failed\n",
            stderr
        );
        return 25;
    }

    OVTRTProbeResult probe{};
    OVTRTStatus probeStatus = ovtrt_probe(&probe);
    if (probeStatus != OVTRTStatusSuccess) {
        std::fprintf(
            stderr,
            "probe_failed status=%d\n",
            static_cast<int>(probeStatus)
        );
        return 21;
    }

    OVTRTRuntime *runtime = nullptr;
    OVTRTStatus createStatus = ovtrt_runtime_create(&runtime);
    if (
        createStatus != OVTRTStatusSuccess ||
        runtime == nullptr
    ) {
        std::fprintf(
            stderr,
            "runtime_create_failed status=%d\n",
            static_cast<int>(createStatus)
        );
        return 22;
    }
    ovtrt_runtime_destroy(runtime);

    OVTRTCUDATransferProbeConfiguration transferConfiguration{
        1920ULL * 1080ULL * 2ULL,
        10,
        100
    };
    OVTRTCUDATransferProbeResult transfer{};
    OVTRTStatus transferStatus = ovtrt_cuda_transfer_probe(
        &transferConfiguration,
        &transfer
    );
    if (transferStatus != OVTRTStatusSuccess) {
        std::fprintf(
            stderr,
            "transfer_probe_failed status=%d stage=%d "
            "cuda=%d cleanup_stage=%d cleanup_cuda=%d\n",
            static_cast<int>(transferStatus),
            static_cast<int>(transfer.failureStage),
            transfer.cudaErrorCode,
            static_cast<int>(transfer.cleanupFailureStage),
            transfer.cleanupCUDAErrorCode
        );
        return 23;
    }
    uint32_t expectedHostToDeviceCopies =
        transferConfiguration.warmupIterationCount +
        transferConfiguration.measuredIterationCount;
    bool transferContractPassed =
        transfer.hostToDeviceCopyCount ==
            expectedHostToDeviceCopies &&
        transfer.deviceToHostVerificationCopyCount == 1 &&
        transfer.hostFrameAllocationCount == 2 &&
        transfer.deviceFrameAllocationCount == 1 &&
        transfer.frameSizedAllocationCountAfterWarmup == 0 &&
        transfer.failureStage ==
            OVTRTCUDATransferStageNone &&
        transfer.cleanupFailureStage ==
            OVTRTCUDATransferStageNone &&
        transfer.hostRegistrationPassed == 1 &&
        transfer.sourceAddressPreserved == 1 &&
        transfer.inputConsumedEventPassed == 1 &&
        transfer.verificationPassed == 1;
    if (!transferContractPassed) {
        std::fputs(
            "transfer_contract_failed\n",
            stderr
        );
        return 24;
    }

    std::printf(
        "{\"status\":\"available\","
        "\"tensorRTVersion\":%d,"
        "\"cudaRuntimeVersion\":%d,"
        "\"cudaDriverVersion\":%d,"
        "\"cudaDeviceCount\":%d,"
        "\"runtimeLifecycle\":\"passed\","
        "\"transfer\":{"
        "\"byteCount\":%llu,"
        "\"warmupIterationCount\":%u,"
        "\"measuredIterationCount\":%u,"
        "\"hostToDeviceCopyCount\":%u,"
        "\"deviceToHostVerificationCopyCount\":%u,"
        "\"hostFrameAllocationCount\":%u,"
        "\"deviceFrameAllocationCount\":%u,"
        "\"frameSizedAllocationCountAfterWarmup\":%u,"
        "\"p50Milliseconds\":%.6f,"
        "\"p95Milliseconds\":%.6f,"
        "\"p50GigabytesPerSecond\":%.6f,"
        "\"p95GigabytesPerSecond\":%.6f,"
        "\"hostRegistrationPassed\":true,"
        "\"sourceAddressPreserved\":true,"
        "\"inputConsumedEventPassed\":true,"
        "\"verificationPassed\":true,"
        "\"contract\":\"passed\"}}\n",
        probe.tensorRTVersion,
        probe.cudaRuntimeVersion,
        probe.cudaDriverVersion,
        probe.cudaDeviceCount,
        static_cast<unsigned long long>(transfer.byteCount),
        transfer.warmupIterationCount,
        transfer.measuredIterationCount,
        transfer.hostToDeviceCopyCount,
        transfer.deviceToHostVerificationCopyCount,
        transfer.hostFrameAllocationCount,
        transfer.deviceFrameAllocationCount,
        transfer.frameSizedAllocationCountAfterWarmup,
        transfer.p50Milliseconds,
        transfer.p95Milliseconds,
        transfer.p50GigabytesPerSecond,
        transfer.p95GigabytesPerSecond
    );
    return 0;
}
