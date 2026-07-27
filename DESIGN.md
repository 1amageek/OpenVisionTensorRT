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

## Engine artifact owner

`OVTRTEngine` memory-maps each plan read-only, computes SHA-256 directly over
the mapped bytes, and deserializes only when the digest matches. The mapped
range is released immediately after `deserializeCudaEngine` returns. The
TensorRT engine is destroyed before its runtime, and both are destroyed before
the CUDA and TensorRT dynamic-library handles close.

The Swift `TensorRTEngine` actor accepts the opaque owner only after checking
the exact TensorRT and CUDA versions, compute capability, tensor names, I/O
modes, element types, declared shapes, and dynamic input profile. A digest
mismatch, incompatible semantic binding, unsupported TensorRT element type, or
deserialization failure remains a typed failure and destroys every partially
created owner.

```text
plan file
  -> read-only mmap
    -> in-process SHA-256
      -> TensorRT deserialize
        -> exact semantic tensor validation
          -> Swift actor owns engine
```

Plan bytes are not copied into `Data`, `Array`, or another frame-sized Swift
container.

### Semantic bounds and execution capacity

TensorRT's dynamic output allocator may request storage for an internal graph
bound that is larger than the final model output. RTMDet's exported graph has
2,100 candidates before its final top-k, while the semantic contract accepts at
most 100 detections. `TensorRTEngineOutputBinding.executionElementCapacity`
therefore records the artifact-specific allocation bound separately from the
semantic tensor shape. The provider requires capacities of 10,500 `Float`
elements for `dets` and 2,100 `Int64` elements for `labels`.

An absent, zero, or insufficient execution capacity is a typed configuration
failure. TensorRT may write only within the persistent artifact capacity, and
the final runtime shape must still satisfy the semantic maximum. Execution
capacity never enlarges the meaning accepted by OpenVision.

## TensorRT execution owner

`TensorRTEngine.prepareExecution()` creates one TensorRT execution context,
non-blocking CUDA stream, and timing-event pair. It derives maximum output
capacities from the validated artifact binding, allocates each output once, and
installs a fixed TensorRT output allocator. An actual runtime shape that exceeds
its declared semantic bound fails with
`outputCapacityExceeded`; it never falls back to an enqueue-time allocation.

`TensorRTEngine.execute(_:)` borrows an existing CUDA input address and binds it
directly. The C boundary synchronizes the completion event before publishing
output views, so the returned device addresses are ready for a downstream CUDA
consumer. A shared lease prevents another submission or shutdown while the
output or any individual tensor view remains live.

```text
prepared CUDA input
    -> scoped address borrow
        -> TensorRT enqueueV3
            -> completion event
                -> stable engine-owned output addresses
                    -> scoped output borrow
```

The detector path performs no tensor-sized input or output copy and no explicit
per-frame device allocation. It still constructs small Swift report, shape,
and lease values per submission; those are not frame-byte materializations.

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

The source word layout is explicit. Standard V4L2 expanded RG10 places the
sample in bits 0...9, while Jetson Xavier and Orin VI expose RAW10 through
NVIDIA T_R16 with the sample in bits 6...15 and replicated low bits. Both the
full-frame detector kernel and region-affine pose kernel consume the same
`RG10WordLayout`; neither guesses a layout from the RG10 FourCC. Layout
selection changes only the bit extraction in the fused kernels and introduces
no additional allocation or copy.

`regionAffine` remains intentionally rejected by the full-frame preprocessor.
The dedicated `TensorRTPosePipeline` receives detector outputs and the original
RG10 device source, selects at most four confidence-sorted person regions on
the GPU, applies the DWPose aspect and 1.25 scale rules, and writes batched
256x192 NCHW pose input directly into persistent device storage. No RGB frame,
crop image, or host-side pose tensor is materialized.

## Implemented pose and observation path

```text
RTMDet device outputs
    + original RG10 device source
        -> CUDA person ROI selection
            -> compact count + regions D2H (84-byte maximum)
            -> CUDA RG10 region affine
                -> persistent DWPose input
                    -> TensorRT DWPose
                        -> CUDA SimCC argmax/coordinate decode
                            -> compact joints D2H (6,384-byte maximum)
                                -> OpenVision observations
```

The pose actor admits one prepared frame at a time. Detector output leases and
the RG10 source owner remain alive until ROI selection and region affine have
synchronized. A pose-input lease then prevents overwrite until TensorRT has
finished consuming the prepared batch. SimCC output leases remain active until
the decode kernel and compact readback complete.

Region and joint readback arrays are allocated once with the pose pipeline.
Each frame writes only the selected prefix. New arrays are created only at the
OpenVision output boundary because observations must own values after the
provider execution returns. Pointer borrows are synchronous, scoped to their
owner, and never cross an actor suspension point.

## Semantic model manifest

The bring-up manifest fixes two independently verifiable semantic stages:

```text
RTMDet-nano
  320x320 / BGR / NCHW / ImageNet channel normalization
    -> dets[1,N,5] + labels[1,N] int64, bounded to N <= 100
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
| Fixture end-to-end inference latency | p95 below 33.3 ms |

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

The final detector run used 10 warm-up and 100 measured submissions. TensorRT
GPU inference measured 2.550144 ms p50 and 2.595040 ms p95. Its two dynamic
outputs retain 58,800 device bytes for the graph's 2,100-candidate execution
bound, reuse identical addresses, and require zero explicit per-frame device
allocations.

One prepared provider session completed 5 warm-up and 30 measured fixture
executions with stable observation counts. End-to-end latency measured
11.659648 ms p50 and 11.748704 ms p95, below the 33.3 ms fixture budget. At the
maximum batch of four, the package-owned persistent device allocations total
9,707,444 bytes (about 9.26 MiB): RG10 source, detector input and outputs, pose
input and decode storage. TensorRT-owned execution workspace is not included
in that number and must be measured separately during sustained camera
operation.

The maximum per-frame D2H payload is 6,468 bytes across three compact copies:
a 4-byte region count, 80 bytes of regions, and 6,384 bytes of decoded joint
tuples. No image or inference tensor crosses D2H. The current host RG10 input
requires exactly one full-frame H2D. A future DMA-BUF path may remove it only
after external-memory ownership and camera release fences are proven.

## Shared-state review matrix

| Logical state | Native storage/isolation | WASM storage/isolation | Embedded storage/isolation | Read/mutation | Shutdown/release |
|---|---|---|---|---|---|
| C preprocessor address and operation lease | `Mutex<State>` | `Mutex<State>` | `Mutex<State>` | short lease acquisition/release under the same mutex; CUDA work runs outside the critical section | destruction is excluded while a lease is active; C destroy clears the address only after successful dependency-ordered cleanup |
| Frame sequencing and active tensor lease | `RG10Preprocessor` actor | same actor contract | same actor contract | actor-isolated `process` and `shutdown`; a live output rejects overwrite | tensor release requires the consumer's completion fence; shutdown rejects a live lease |
| Deferred failed cleanup | `Mutex<[Entry]>` registry | same mutex contract | same mutex contract | entries are removed under lock and cleanup is attempted outside the critical section | failed owners remain retained and are retried before a new preprocessor is created |
| TensorRT engine owner | `TensorRTEngine` actor plus `Mutex<UInt?>` opaque-handle owner | same actor and mutex contract; runtime returns typed unavailable | same actor and mutex contract; runtime returns typed unavailable | actor serializes inspection and shutdown; the mutex protects exactly-once address consumption | engine precedes runtime, and both precede dynamic-library close |
| TensorRT output lease | `Mutex<State>` holder and borrow counts | same mutex contract | same mutex contract | a synchronous scoped borrow prevents release; every output holder retains the same lease | another enqueue and shutdown reject an unreleased output; the last holder closes the lease |
| Pose pipeline state and readback storage | `TensorRTPosePipeline` actor plus `Mutex<UInt?>` opaque-handle owner | same actor and mutex contract; runtime returns typed unavailable | same actor and mutex contract; runtime returns typed unavailable | actor serializes prepare/decode/discard; persistent region and joint arrays are synchronously borrowed without suspension | shutdown rejects a pending decode or live pose-input lease and consumes the C owner exactly once |
| Pose input lease | `Mutex<State>` | same mutex contract | same mutex contract | TensorRT receives a scoped device-address borrow; release cannot overlap the borrow | a new pose preparation and pipeline shutdown reject an unreleased input |
| Provider session state | `OpenVisionTensorRTProviderSession` actor | same actor contract | same actor contract | actor serializes preparation, one in-flight request, cancellation, cleanup, and shutdown | cleanup attempts every owned intermediate and reports the first typed failure; shutdown visits every backend owner |
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
It also deserialized the exact RTMDet and DWPose plans through the public Swift
loader and validated every declared tensor and dynamic pose batch profile.
Wrong-checksum and swapped-semantic-binding probes failed at their expected
typed boundaries. Independent `trtexec` runs prove that both plans execute on
this GPU. The public provider completed detector decoding, region-affine pose
input, DWPose execution, SimCC decoding, and OpenVision observation
construction, then sustained 30 measured executions on one session with stable
counts. The real camera lease and sustained capture remain the next milestone;
fixture success is not presented as camera evidence.
