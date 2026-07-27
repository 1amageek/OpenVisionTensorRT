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
- [x] Explicit standard V4L2 and NVIDIA T_R16 RG10 word-layout selection
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
- [x] GPU detector decode with bounded confidence-sorted person ROI selection
- [x] GPU person-region duplicate suppression before batched pose inference
- [x] Fused RG10-to-pose region-affine CUDA preprocessing
- [x] Batched DWPose execution through the public Swift path
- [x] SimCC GPU decode with compact region and joint readback
- [x] Complete body and bilateral-hand OpenVision observation construction
- [x] Production `VisionProvider` and `VisionProviderSession` conformance
- [x] Typed cancellation, busy, release, discard, and shutdown paths
- [x] Persistent GPU allocations and bounded host readback storage
- [x] Sustained provider execution on Jetson with stable observation counts
- [x] Stratified overhead-image screening through the production RG10 provider
      path with per-frame observation and latency reports
- [x] ActionRecognition integration target with stateful OpenVision tracking,
      bounded compact history, and typed horizontal-swipe lifecycle output
- [x] Dataset evaluator preserves generic pose, action, pointing, gesture,
      cancellation, and ambiguity decisions from the portable
      `RecognitionObservation` contract

## Incomplete production work

- [ ] Run the public provider with a real camera frame lease.
- [ ] Measure sustained 1920x1080 RG10 at 30 FPS on Jetson.
- [ ] Prove a DMA-BUF/external-memory import before advertising direct import.
- [ ] Quantify ceiling-view keypoint accuracy and false positives with captured
      WAVESHARE-26185 product-domain data and keypoint ground truth.
- [ ] Complete the checkpoint and training-dataset license review.

The first launch immediately after the final image transfer reported one typed
detector `outputAllocation` failure. The same immutable image then passed a
manual full-probe run and a subsequent cold-start full-probe run. Treat a
recurrence as a release-blocking reliability defect; do not convert it into an
automatic or silent fallback.

## Verification evidence

| Check | Current evidence |
|---|---|
| macOS unavailable, configuration, lifecycle, semantic manifest, engine-binding, output-capacity, provider, decoder, and lease behavior | 29 tests in three suites passed with `xcodebuild test` |
| macOS sanitizer behavior | The same 29 tests passed independently with Address Sanitizer and Thread Sanitizer |
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
| Swift detector execution | The final cold-start probe used 10 warm-up and 100 measured detector submissions. GPU inference measured p50 2.550144 ms and p95 2.595040 ms |
| Detector execution memory | The dynamic detector allocator reserves the graph-required 2,100 candidates separately from the semantic maximum of 100 detections: 58,800 persistent device bytes, stable output addresses, and zero explicit per-frame device allocations |
| Complete fixture provider path | A real 1920x1080 RG10 fixture passed `VisionImageInput` borrow -> one H2D -> RTMDet -> bounded ROI -> RG10 region affine -> DWPose -> SimCC decode -> four `HumanBodyPoseObservation` values containing 74 body and 138 hand joints |
| Duplicate-region contract | Detector candidates with intersection-over-union above 0.5 are suppressed on the GPU in favor of the higher-confidence candidate before pose batching; the configured threshold has typed range validation |
| Sustained provider path | One prepared provider session completed 5 warm-up and 30 measured executions with stable observation counts; end-to-end p50 11.659648 ms and p95 11.748704 ms |
| Compact readback | No image or tensor is copied back to the host. At the four-person bound, only one 4-byte count, 80 bytes of regions, and 6,384 bytes of joint tuples cross D2H |
| Rough overhead screening | 48 WEPDTOF frames from 16 scenes ran through one provider session. Thirteen frames produced poses; capacity-adjusted count recall proxy was 15.6%, with no false positive in the single negative frame. The 1–4-person subset produced a pose in only 1 of 10 frames, showing a ceiling/fisheye domain gap beyond the four-person cap. Warm execution averaged 7.53 ms, with p50 5.24 ms and maximum 14.58 ms |
| Current complete GPU slice | `VisionImageInput` borrow -> one RG10 H2D -> fused CUDA detector preprocessing -> RTMDet -> GPU ROI affine -> DWPose -> GPU SimCC decode -> compact D2H -> OpenVision observations |
| M4 temporal integration | The current evaluator completed four contiguous IPN slices totaling 236 frames. Pose was present in 236/236 frames and every slice retained one actor on one track. `D0X` produced no recognition, but `G05` was misclassified as rotary, `G06` ended a swipe in the preparatory direction, and `B0B` ended two swipes without a pointing observation. This proves execution, not semantic accuracy |
| M4 temporal latency | Across the four slices, end-to-end p95 ranged from 8.548128 to 9.371936 ms and ActionRecognition p95 ranged from 0.035296 to 0.036672 ms |
| M4 bounded history | Maximum 61 compact samples and 4,392 retained feature bytes; no full frame was retained by ActionRecognition |
| Temporal evaluator acceptance | The current top-level `passed` status checks only whether any gesture reached `ended`. It does not score identifier, direction, label mapping, or temporal overlap and therefore is not an accuracy verdict |
| Expanded ActionRecognition contract | Dataset evaluator builds against ActionRecognition `49e0e2f` and serializes non-gesture observations and ambiguous candidates without coercing them into gesture or no-match results |
| Remaining integration path | Actual camera lease and storage negotiation, then sustained 30 FPS capture. DMA-BUF direct import is a separate capability and is not advertised |

## Release gates beyond implementation

- The selected checkpoints are a bring-up baseline. Rough third-party
  ceiling-view screening shows inadequate recall, but product-domain keypoint
  accuracy and false-positive behavior remain unmeasured.
- `licenseIdentifier` remains absent for both checkpoint provenances until the
  model and training-dataset usage review is complete.
- Neither condition is converted into a successful production-readiness claim.

## M4 execution boundary

M4 proves complete offline temporal execution on the Jetson GPU, while the
expanded dataset screening exposes unresolved semantic failures:

```text
236 contiguous RG10 frames across four IPN slices
    -> production TensorRT pose provider
    -> OpenVision stateful body tracking
    -> ActionRecognition semantic observations
    -> typed JSON latency/memory and decision evidence
         execution path: passed
         dataset semantic match: not passed
```

Wendy CLI `2026.07.17-173113` lost the OCI entrypoint and image environment
variables when its default chunk-diff deployment path was used. Repeating the
same deployment with `--chunking off` preserved
`/usr/local/bin/run-openvision-probes` and both exact plan digests. That normal
Wendy launch completed with exit code zero. Deployment instructions therefore
pin the verified non-chunked path until the CLI defect is fixed.
