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
  --restart-on-failure \
  --yes
```

Success requires CUDA device enumeration, one real TensorRT runtime
creation/destruction cycle, the representative frame transfer contract, and
the fused RG10 preprocessing contract.
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

The RG10 contract compares CUDA output against an independent CPU reference for
all eight orientations under all three resize policies. It also covers Bayer
black/gain calibration, a 3x3 color matrix, sRGB transfer, affine
normalization, NCHW/NHWC, RGB/BGR, and an independent RGGB golden fixture. The
performance path converts one 1920x1080 SRGGB10 frame into a 256x256 tensor
with one H2D copy, one fused kernel, and no explicit frame-sized device
allocation after preparation. A second executable verifies the same operation
through the public Swift `VisionImageInput` and device-tensor lease API.

Verified on WendyOS 0.18.1 / JetPack 7.2:

```json
{"cProbe":{"status":"available","tensorRTVersion":101602,"cudaRuntimeVersion":13020,"cudaDriverVersion":13020,"cudaDeviceCount":1,"runtimeLifecycle":"passed","transfer":{"p50Milliseconds":0.169664,"p95Milliseconds":0.170720,"contract":"passed"},"rg10Preprocessing":{"verifiedCases":25,"maximumAbsoluteDifference":0.00000012,"fullFrameHostToDeviceCopiesPerFrame":1,"kernelLaunchesPerFrame":1,"explicitFrameSizedDeviceAllocationsAfterPreparation":0,"p50Milliseconds":0.657248,"p95Milliseconds":0.675616,"endToEndP50Milliseconds":0.667455,"endToEndP95Milliseconds":0.686560,"contract":"passed"},"retryableCleanup":"passed"},"swiftProbe":{"status":"available","swiftPublicPath":"passed","deviceAddressNonzero":true,"inputReleased":true,"h2dCopies":1,"kernelLaunches":1}}
```

This proves the runtime boundary, a representative camera-sized host-to-GPU
transfer, and the fused preprocessing kernel through both the C ABI and public
Swift API. It does not claim that an actual camera lease, model engine, or pose
inference has run.
