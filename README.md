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
- semantic-model versus TensorRT-engine artifact metadata;
- memory-mapped, SHA-256-verified TensorRT plan deserialization;
- typed runtime, compute-capability, tensor-name, type, shape, and profile
  compatibility validation;
- reusable TensorRT execution contexts, CUDA streams, timing events, and
  manifest-bounded output allocations;
- completion-fenced detector output leases with stable device addresses and no
  explicit per-frame device allocation.

The public Swift path now executes the detector engine on Jetson. Successful
body or hand pose inference is not implemented until detector decoding, GPU
region-affine preprocessing, DWPose execution, SimCC decoding, and the provider
execution path are verified. The package does not report an alternate CPU
provider.

```text
VisionImageInput (RG10)
    -> RG10Preprocessor
        -> one H2D + fused CUDA kernel
            -> RTMDet input tensor
                -> TensorRT detector execution (verified)
                    -> reusable device outputs (verified)
                    -> bounded person regions (next phase)
                    -> DWPose region-affine tensor (next phase)
                        -> DWPose execution/provider (next phase)
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

The same Swift deployment accepted the exact Jetson plans below after
memory-mapped SHA-256 verification and TensorRT deserialization:

| Stage | Plan SHA-256 | Verified tensors |
|---|---|---|
| RTMDet | `be0b54389dd01b4b571a06623978032f6d2e20a125872670ce7233bf22204775` | `input [1,3,320,320]`, `dets [1,-1,5]`, `labels [1,-1]` |
| DWPose | `32d56cc71e84c9d6fbde29cd782c0b6e3fcfcd80b67e2717f0bab42c212ba674` | `input [1...4,3,256,192]`, `simcc_x [B,133,384]`, `simcc_y [B,133,512]` |

Negative Jetson probes also proved that a wrong digest fails at the typed
`checksum` stage and swapped semantic output bindings fail with a typed element
type incompatibility. Neither condition falls back to another engine.

The final detector execution path was measured three times with 10 warm-up and
100 measured submissions. GPU inference p50 was 2.533344–2.569632 ms and p95
was 2.574208–2.859296 ms. The two dynamic detector outputs retain 2,800 device
bytes, reuse identical device addresses, and perform zero explicit per-frame
device allocations after preparation.

Build and run the exact runtime boundary on a configured Wendy Jetson:

```bash
./scripts/prepare-wendy-runtime-probe.sh
wendy run \
  --device 169.254.58.23 \
  --prefix Deployment/Wendy \
  --builder apple-container \
  --no-restart \
  --yes
```
