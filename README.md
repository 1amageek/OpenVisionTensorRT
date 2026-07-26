# OpenVisionTensorRT

OpenVisionTensorRT is the NVIDIA CUDA and TensorRT provider package for
OpenVision. Jetson is the first deployment environment.

The current implementation owns:

- CUDA and TensorRT runtime probing;
- dynamic loading of the installed Jetson CUDA/TensorRT libraries;
- an opaque C ABI for TensorRT runtime ownership;
- exactly-once runtime destruction;
- a registered-host-memory CUDA transfer probe with explicit event lifetime;
- measured H2D copy, allocation, latency, throughput, and content evidence;
- fused SRGGB10 GPU preprocessing into a reusable device tensor;
- typed calibration, orientation, resize, normalization, and tensor layout;
- a completion-fenced device tensor lease exposed through a scoped Swift API;
- retryable cleanup that retains CUDA owners after destruction failure;
- a validated RTMDet-nano plus DWPose-m semantic model manifest;
- semantic-model versus TensorRT-engine artifact metadata.

The semantic model has been selected for bring-up, but successful body or hand
pose inference is not implemented until matching TensorRT engines and the
provider execution path are verified. The package does not report an alternate
CPU provider.

```text
VisionImageInput (RG10)
    -> RG10Preprocessor
        -> one H2D + fused CUDA kernel
            -> RTMDet input tensor
                -> bounded person regions
                    -> DWPose region-affine tensor (next phase)
                        -> TensorRT provider (next phase)
```

The standalone transfer probe uses a 1920x1080 frame stored in a 16-bit raw
container. It registers the existing page-aligned host address, performs
exactly one asynchronous H2D copy per iteration, and retains the host owner
until the CUDA completion event has synchronized. A separate D2H copy is used
only after measurement to verify the device contents byte-for-byte.

```text
VisionImageInput scoped bytes
    -> cudaMemcpy(H2D) exactly once
        -> scoped source read completes
            -> fused CUDA kernel
                -> RG10DeviceTensor lease
```

On Jetson Orin Nano, the fused 1920x1080 RG10 to 256x256 tensor path measured
0.653184 ms p50 and 0.696288 ms p95 including the one H2D copy. The complete
public preprocessing API path measured 0.663581 ms p50 and 0.706149 ms p95.
Twenty-four differential cases covering all orientations and resize policies,
plus an independent RGGB golden fixture, matched the CPU reference within
0.00000012. The public Swift Jetson path also verified a nonzero device
address, input release, tensor release, and shutdown.
The Jetson probe also fault-injected a cleanup synchronization failure and
proved that a second destruction attempt consumes the retained owner.

Build and run the exact runtime boundary on a configured Wendy Jetson:

```bash
./scripts/prepare-wendy-runtime-probe.sh
wendy run \
  --device 169.254.58.23 \
  --prefix Deployment/Wendy \
  --builder apple-container \
  --restart-on-failure \
  --yes
```
