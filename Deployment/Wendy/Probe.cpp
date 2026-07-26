#include "CTensorRTShim.h"

#include <cstdio>

int main() {
    if (ovtrt_probe(nullptr) != OVTRTStatusInvalidArgument) {
        std::fputs("probe_null_contract_failed\n", stderr);
        return 20;
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

    std::printf(
        "{\"status\":\"available\","
        "\"tensorRTVersion\":%d,"
        "\"cudaRuntimeVersion\":%d,"
        "\"cudaDriverVersion\":%d,"
        "\"cudaDeviceCount\":%d,"
        "\"runtimeLifecycle\":\"passed\"}\n",
        probe.tensorRTVersion,
        probe.cudaRuntimeVersion,
        probe.cudaDriverVersion,
        probe.cudaDeviceCount
    );
    return 0;
}
