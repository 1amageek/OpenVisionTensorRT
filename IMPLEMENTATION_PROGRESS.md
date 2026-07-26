# OpenVisionTensorRT Implementation Progress

## Implemented

- [x] Separate Swift 6.4 package
- [x] Conditional CUDA/TensorRT C++ runtime boundary
- [x] CUDA runtime, driver, device-count, and TensorRT version probe
- [x] Opaque TensorRT runtime creation and exactly-once destruction
- [x] Typed unavailable and lifecycle failures
- [x] TensorRT engine artifact compatibility descriptor
- [x] macOS unavailable-path behavior tests
- [x] Dynamic CUDA/TensorRT symbol loading without a silent fallback
- [x] Wendy arm64 deployment with pinned CUDA image and TensorRT headers
- [x] Jetson CUDA device enumeration and real TensorRT runtime lifecycle probe
- [x] Page-aligned host storage registration without address replacement
- [x] One-copy asynchronous H2D transfer with input-consumed CUDA event
- [x] Pre-warm-up host/device allocation and zero frame allocation thereafter
- [x] Test-only D2H byte verification outside the measured H2D path
- [x] Typed primary and cleanup transfer failure stages
- [x] Jetson p50/p95 transfer latency and throughput measurement
- [x] Reusable RG10 input, tensor output, and configuration device buffers
- [x] One NVRTC compilation and PTX module load per prepared preprocessor
- [x] Fused SRGGB10 linearization, bilinear demosaic, orientation, and resize
- [x] Letterbox/crop, color matrix, sRGB, normalization, layout, and channel order
- [x] One H2D copy and one kernel launch per frame
- [x] Scoped OpenVision borrow with synchronous H2D source-read fence
- [x] Output-ready CUDA event and explicit device-tensor lease
- [x] Public Swift API probe on the real Jetson GPU
- [x] Retryable dependency-ordered CUDA cleanup without premature owner release
- [x] Jetson fault injection proving failed cleanup retains the retry owner
- [x] Differential GPU/CPU fixtures for all orientations and resize policies
- [x] Independent RGGB golden fixture
- [x] Jetson preprocessing p50/p95 and end-to-end frame-path measurements
- [x] Backend-neutral two-stage semantic model manifest
- [x] RTMDet-nano detector input, output, and exact provenance contract
- [x] DWPose-m region-affine, SimCC, and exact provenance contract
- [x] Complete OpenVision body and bilateral hand joint mapping
- [x] Per-channel normalization propagated to the fused CUDA kernel ABI
- [x] Typed rejection of region-affine input by the full-frame preprocessor
- [x] Reproducible revision- and digest-pinned RTMDet and pose ONNX export
- [x] ONNX checker and ONNX Runtime verification for exact tensor contracts
- [x] Jetson FP16 detector and pose TensorRT plan builds
- [x] Memory-mapped plan loading with in-process SHA-256 verification
- [x] Real TensorRT runtime deserialization with exactly-once engine ownership
- [x] Runtime, CUDA, compute-capability, tensor, shape, and profile validation
- [x] Swift Jetson engine-load probe for both selected model stages
- [x] Jetson typed checksum and semantic-binding negative probes
- [x] Reusable TensorRT execution context, CUDA stream, and timing events
- [x] Manifest-bounded persistent output allocations for dynamic tensors
- [x] Scoped device-input and device-output leases without tensor copies
- [x] TensorRT detector submission through the public Swift API
- [x] Output-address reuse across warm-up and 100 measured submissions
- [x] Typed output-capacity, execution-stage, borrow, and cleanup failures
- [x] Retryable explicit execution-resource cleanup

## Incomplete production work

- [ ] Implement detector decode, bounded ROI selection, and GPU region affine.
- [ ] Execute DWPose through the Swift path after GPU region-affine preparation.
- [ ] Implement SimCC GPU decode and compact observation readback.
- [ ] Conform the production provider to `VisionProvider`.
- [ ] Run the public provider with a real camera frame lease.
- [ ] Measure sustained 1920x1080 RG10 at 30 FPS on Jetson.
- [ ] Prove a DMA-BUF/external-memory import before advertising direct import.

No callable pose provider is declared yet, so no incomplete inference branch can
be mistaken for successful production execution.

## Verification evidence

| Check | Current evidence |
|---|---|
| macOS unavailable, configuration, lifecycle, semantic manifest, engine-binding, and output-lease behavior | 23 tests passed with `xcodebuild test` |
| macOS sanitizer behavior | The same 23 tests passed with Address Sanitizer and Thread Sanitizer |
| Regular WASM | Debug and release `OpenVisionTensorRT` target builds passed with `swift-6.4.x-DEVELOPMENT-SNAPSHOT-2026-07-17-a` toolchain and matching SDK |
| Embedded WASM | Debug and release `OpenVisionTensorRT` target builds passed with the same 2026-07-17 toolchain and Embedded SDK |
| Exact snapshot identity | Swift `9517428e7f4b63e`, LLVM `3704913b9103f85`, target `wasm32-unknown-wasip1` |
| Jetson runtime | TensorRT 10.16.2 and CUDA 13.2 reported one device |
| Jetson sanitizer behavior | The full runtime and transfer path passed ASan and UBSan; leak detection is unavailable under Wendy device supervision |
| TensorRT ownership | Real `IRuntime` creation and destruction passed on Jetson |
| Representative H2D transfer | 4,147,200 bytes; 110/110 copies; p50 0.170016 ms; p95 0.171136 ms |
| Representative H2D throughput | p50 24.393 GB/s; p95 24.233 GB/s |
| Transfer ownership probe | Host registration, source-address preservation, input-consumed event, and byte verification passed |
| Transfer allocation | Two pre-warm-up host buffers, one device buffer, zero frame-sized allocations after warm-up |
| RG10 differential verification | 24 GPU/CPU orientation/resize cases, including distinct per-channel normalization, plus one independent RGGB golden fixture; maximum absolute difference 0.00000012 |
| RG10 GPU pipeline | 1920x1080 RG10 to 256x256 tensor including one H2D; p50 0.653184 ms; p95 0.696288 ms |
| RG10 end-to-end preprocessing | p50 0.663581 ms; p95 0.706149 ms; one H2D, one kernel, zero explicit post-prepare frame-sized device allocations |
| Public Swift Jetson path | `VisionImageInput` borrow, CUDA tensor lease, nonzero device address, input release, tensor release, and shutdown passed |
| Retryable cleanup | Injected stream-synchronization failure retained a non-null owner; second destruction succeeded |
| Semantic model | RTMDet-nano plus DWPose-m manifest fixes exact preprocessing, bounded runtime-variable detector output shapes, `int64` labels, 61 body/hand mappings, source revisions, and SHA-256 digests |
| ONNX exchange graphs | Reproducible export produced detector `68f9b14e...` and pose `9597eb65...`; ONNX checker passed and ONNX Runtime produced the declared detector outputs and pose batches one and four |
| TensorRT plans | RTMDet plan `be0b5438...` and DWPose plan `32d56cc7...` built for TensorRT 10.16.2 / CUDA 13.2 / compute capability 8.7 |
| Independent engine execution | `trtexec` executed both plans on the Jetson GPU: detector batch 1 p50 3.46277 ms; pose batch 1 p50 2.33325 ms; pose batch 4 p50 5.73291 ms |
| Swift engine acceptance | Both plans passed mmap SHA-256, deserialization, exact runtime/device compatibility, tensor name/mode/type/shape, and pose profile validation |
| Typed Jetson failures | Wrong SHA-256 produced `engineChecksumMismatch` at `checksum`; swapped detector output meanings produced `incompatibleArtifact` for `labels` element type |
| Swift detector execution | Three final-code runs used 10 warm-up and 100 measured submissions each. GPU inference p50 was 2.533344–2.569632 ms and p95 was 2.574208–2.859296 ms |
| Detector execution memory | The two dynamic detector outputs use 2,800 persistent device bytes; output addresses remained identical and the package performed zero explicit per-frame device allocations across 110 submissions |
| Current complete GPU slice | `VisionImageInput` borrow -> one RG10 H2D -> fused CUDA preprocessing -> device-tensor lease -> TensorRT detector enqueue -> completion-fenced reusable device outputs |
| Remaining inference path | Detector decode, GPU ROI affine, DWPose Swift execution, SimCC decode, observation construction, and the OpenVision provider are not implemented yet |

## Release gates beyond implementation

- The selected checkpoints are a bring-up baseline. Ceiling-view camera
  accuracy and false-positive behavior remain unmeasured.
- `licenseIdentifier` remains absent for both checkpoint provenances until the
  model and training-dataset usage review is complete.
- Neither condition is converted into a successful production-readiness claim.
