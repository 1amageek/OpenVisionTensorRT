# Wendy TensorRT Runtime Probe

This deployment validates the real `CTensorRTShim` path against the
CUDA/TensorRT libraries exposed by Wendy's GPU entitlement. It does not contain
or select a vision model.

Prepare the exact package sources as the Docker build context:

```bash
./scripts/prepare-wendy-runtime-probe.sh
```

Then build and run the arm64 container:

```bash
wendy run \
  --device 169.254.58.23 \
  --prefix Deployment/Wendy \
  --builder apple-container \
  --no-restart \
  --yes
```

Success requires CUDA device enumeration, one real TensorRT runtime
creation/destruction cycle, and the representative frame transfer contract.
The container first executes the same source with Address Sanitizer and
Undefined Behavior Sanitizer, then executes the optimized binary used for the
reported performance measurements. Leak detection is disabled because Wendy's
device supervision is incompatible with LeakSanitizer; address-boundary and
undefined-behavior checks remain enabled.
The transfer contract requires one H2D copy per iteration, no frame-sized
allocation after warm-up, source-address preservation, a synchronized
input-consumed event, and byte-for-byte device-content verification. Missing
libraries, inaccessible GPU devices, transfer failures, and runtime creation
failures exit nonzero.

Verified on WendyOS 0.18.1 / JetPack 7.2:

```json
{"status":"available","tensorRTVersion":101602,"cudaRuntimeVersion":13020,"cudaDriverVersion":13020,"cudaDeviceCount":1,"runtimeLifecycle":"passed","transfer":{"byteCount":4147200,"warmupIterationCount":10,"measuredIterationCount":100,"hostToDeviceCopyCount":110,"deviceToHostVerificationCopyCount":1,"hostFrameAllocationCount":2,"deviceFrameAllocationCount":1,"frameSizedAllocationCountAfterWarmup":0,"p50Milliseconds":0.170080,"p95Milliseconds":0.171040,"p50GigabytesPerSecond":24.383819,"p95GigabytesPerSecond":24.246960,"hostRegistrationPassed":true,"sourceAddressPreserved":true,"inputConsumedEventPassed":true,"verificationPassed":true,"contract":"passed"}}
```

This proves the runtime boundary and a representative camera-sized host-to-GPU
transfer. It does not claim that an actual camera lease, model engine,
preprocessing kernel, or pose inference has run.
