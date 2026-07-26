# OpenVisionTensorRT Design

## Boundary

OpenVisionTensorRT imports OpenVision. OpenVision never imports this package.
Camera capture remains in OpenAVFoundation camera drivers.

The C++ boundary exposes opaque owners and a non-owning device-tensor view.
Swift actors own the opaque handles. The public Swift tensor exposes its device
address only through a scoped closure and retains the provider owner until its
explicit completion-fenced lease is released.

On Linux, the shim resolves CUDA 13 and TensorRT 10 symbols with `dlopen` and
`dlsym`. This keeps the Swift package buildable on non-NVIDIA hosts without
inventing a successful fallback. Missing libraries, missing symbols, version
probe failures, and runtime creation failures remain distinct typed failures.
The scoped C++ loader owns every library handle and closes it only after all
TensorRT objects have been destroyed.

## Runtime owner

`OVTRTRuntime` owns its logger and `nvinfer1::IRuntime`. The logger is declared
first so it outlives the runtime. `ovtrt_runtime_create` transfers ownership
only after all allocations and TensorRT initialization succeed.
`ovtrt_runtime_destroy` consumes the owner exactly once.

The Swift `TensorRTRuntime` actor serializes lifecycle transitions. Explicit
shutdown clears the handle before calling the C destructor, preventing
reentrant double destruction. Actor deinitialization destroys an unconsumed
handle.

## Implemented RG10 preprocessing path

```text
RG10 V4L2 lease
    -> borrow the existing OpenVision bytes
        -> one synchronous H2D transfer inside the borrow
            -> release source input
                -> fused CUDA linearize / demosaic / orient / resize
                    -> color transform / sRGB / normalize / layout
                        -> completion-fenced device tensor lease
```

NVRTC compiles the model-independent fused kernel once during preprocessor
creation. PTX loading, the non-blocking stream, two events, the RG10 input
buffer, output tensor, and calibration/configuration buffer are all prepared
before frame submission. A frame performs one H2D transfer and one kernel
launch without an intermediate RGB/RGBA frame or a frame-sized allocation.
The H2D operation is synchronous because `VisionImageInput` provides a scoped
borrow: no CUDA host read may outlive that closure, including an error path.
After the copy, all work uses provider-owned device allocations.

The kernel accepts the eight OpenVision/EXIF orientations, `scaleFill`,
`scaleFit`, and `centerCrop`, NCHW and NHWC layouts, RGB and BGR ordering,
per-Bayer-site black levels and gains, a 3x3 color transform, optional sRGB
transfer, and independent red/green/blue affine normalization. Unsupported
hosts return typed unavailable evidence; no CPU fallback is selected.

`regionAffine` is intentionally rejected by this full-frame preprocessor. The
DWPose stage requires a detector-derived, aspect-preserving affine ROI and is
not equivalent to any full-frame resize mode. Its dedicated GPU implementation
belongs to the TensorRT execution milestone.

## Semantic model manifest

The bring-up manifest fixes two independently verifiable semantic stages:

```text
RTMDet-nano
  320x320 / BGR / NCHW / ImageNet channel normalization
    -> at most four person regions above 0.3 confidence
      -> 1.25x region-affine crop
        -> DWPose-m
          256x192 / RGB / NCHW / 133 COCO-WholeBody joints
            -> SimCC X[batch,133,384] + Y[batch,133,512]
```

Both checkpoint source URLs, source revisions, exact SHA-256 digests, training
datasets, and citations are part of the manifest. The output mapping covers all
OpenVision body joints and both complete hand vocabularies. Neck and root are
derived from shoulder and hip midpoints using the minimum input confidence.

The manifest is model meaning, not an engine. TensorRT plan checksums, TensorRT
and CUDA versions, compute capability, precision, memory workspace, and
execution-context ownership remain in provider artifact contracts. The
ceiling-camera domain and dataset-license review are explicit acceptance gates,
not inferred from successful checkpoint conversion.

Direct import will be advertised only after a DMA-BUF or external-memory path
is proven on the actual Jetson camera storage.

## CUDA transfer probe

The transfer probe establishes the ownership and synchronization contract
before camera integration. It allocates and initializes one page-aligned host
source, registers that same address with CUDA, allocates one reusable device
buffer, and submits one H2D operation per iteration on a non-blocking stream.
The source borrow ends only when the recorded CUDA event has synchronized.

The source, verification, and device allocations all occur before warm-up.
Timed iterations allocate no frame-sized storage. The D2H verification copy is
outside the timed H2D path and exists only to compare the final GPU contents
with the original source. After stream synchronization, CUDA resources are
released in dependency-safe order, and primary versus cleanup failures retain
separate typed stages.

The probe proves a one-copy CUDA ingestion path. It does not claim direct GPU
import or zero-copy camera-to-GPU access. OpenVision's CPU boundary remains
zero-copy because the existing owner is borrowed and registered without
materializing another CPU frame.

## Performance acceptance budget

The first production pipeline has the following measurable budget:

| Item | Budget |
|---|---|
| CPU frame materialization | 0 copies |
| Host-to-device transfer | exactly 1 full-frame transfer |
| Frame-sized allocation after warm-up | 0 |
| In-flight inference | 1, with explicit busy failure |
| Sustained input rate | 1920x1080 RG10 at 30 FPS |
| End-to-end inference latency | p95 below 33.3 ms |

The source lease may be released only after the synchronous H2D transfer or an
equivalent target-specific fence proves that GPU work no longer reads host
memory. Preprocessing, inference, and decoding reuse a prepared execution
slot. The eventual model-specific implementation must report copy count,
allocation count, p50/p95 latency, host/device resident memory, power, and
thermal state; runtime creation alone does not satisfy this budget.

The 2026-07-26 Jetson probe measured the representative 4,147,200-byte source
at 0.170016 ms p50 and 0.171136 ms p95, corresponding to 24.393 GB/s and
24.233 GB/s. All 110 H2D submissions used one copy each, no frame-sized
allocation occurred after warm-up, and address preservation, completion-event
ownership, and byte verification passed.

The fused 1920x1080 RG10 to 256x256 NCHW pipeline, including the one H2D copy,
measured 0.653184 ms p50 and 0.696288 ms p95 on the same Jetson. The complete
public preprocessing API path measured 0.663581 ms p50 and 0.706149 ms p95.
Twenty-four orientation/resize differential cases, including independent
red/green/blue normalization coefficients, plus one independent RGGB golden
fixture matched the CPU reference with a maximum absolute difference of
0.00000012.

The Swift deployment also passed the public path from `VisionImageInput`
through the scoped borrow, one H2D copy, fused kernel, nonzero device-tensor
address, input release, tensor release, and actor shutdown. A direct DMA-BUF or
external-memory camera import remains a separate capability and will not be
advertised until its owner and fence contract is proven on the real camera.
The deployment fault-injected one cleanup synchronization failure, verified
that the opaque owner remained non-null, then retried destruction successfully.

## Shared-state review matrix

| Logical state | Native storage/isolation | WASM storage/isolation | Embedded storage/isolation | Read/mutation | Shutdown/release |
|---|---|---|---|---|---|
| C preprocessor address and operation lease | `Mutex<State>` | `Mutex<State>` | `Mutex<State>` | short lease acquisition/release under the same mutex; CUDA work runs outside the critical section | destruction is excluded while a lease is active; C destroy clears the address only after successful dependency-ordered cleanup |
| Frame sequencing and active tensor lease | `RG10Preprocessor` actor | same actor contract | same actor contract | actor-isolated `process` and `shutdown`; a live output rejects overwrite | tensor release requires the consumer's completion fence; shutdown rejects a live lease |
| Deferred failed cleanup | `Mutex<[Entry]>` registry | same mutex contract | same mutex contract | entries are removed under lock and cleanup is attempted outside the critical section | failed owners remain retained and are retried before a new preprocessor is created |
| CUDA work state | C owner, serialized by the Swift actor | unavailable typed boundary | unavailable typed boundary | submit/wait/destroy state machine | stream sync precedes source unregister, module unload, buffer/event/stream destruction |

There is no `hasFeature(Embedded)` or `canImport(Synchronization)` branch in
this package. The same owner, mutex, actor, and typed failure contracts compile
for Native, regular WASM, and Embedded WASM. CUDA capability is selected by the
C runtime boundary, not by weakening shared-state isolation.

## Verified Jetson runtime boundary

The Wendy deployment cross-compiles the exact package shim with pinned
TensorRT 10.16 headers and executes it using the Jetson GPU entitlement. On
WendyOS 0.18.1 / JetPack 7.2 it verified TensorRT 10.16.2, CUDA runtime and
driver 13.2, one CUDA device, and one real runtime creation/destruction cycle.
This now proves the runtime ownership boundary and fused RG10 preprocessing
through both the C ABI and public Swift API. It does not prove semantic pose
inference or a real camera lease; those remain separate milestones.
