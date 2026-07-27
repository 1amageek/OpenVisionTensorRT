# Wendy TensorRT Runtime Probe

This deployment validates the real `CTensorRTShim` path against the
CUDA/TensorRT libraries exposed by Wendy's GPU entitlement. Model plans are
optional inputs and are never stored in the repository.

Prepare the exact package sources as the Docker build context:

```bash
OPENVISION_TRT_DETECTOR_PLAN=/path/to/rtmdet-fp16.plan \
OPENVISION_TRT_POSE_PLAN=/path/to/dwpose-fp16.plan \
./scripts/prepare-wendy-runtime-probe.sh
```

An optional stratified RG10 dataset evaluation can be included without
changing the production provider:

```bash
OPENVISION_TRT_DETECTOR_PLAN=/path/to/rtmdet-fp16.plan \
OPENVISION_TRT_POSE_PLAN=/path/to/dwpose-fp16.plan \
OPENVISION_TRT_RG10_FIXTURE=/path/to/person.rg10 \
OPENVISION_TRT_EVALUATION_ROOT=/path/to/evaluation \
./scripts/prepare-wendy-runtime-probe.sh
```

The evaluation root must contain `runtime-manifest.tsv` and `fixtures/*.rg10`.
The evaluator keeps one TensorRT provider session alive across all frames and
reports count proxies, joint coverage, and p50/p95 latency. These metrics are
screening evidence; they are not pose accuracy because the WEPDTOF base release
does not contain joint ground truth.

A contiguous temporal RG10 sequence can exercise the complete M4 path from
TensorRT pose inference through OpenVision tracking and ActionRecognition:

```bash
OPENVISION_TRT_DETECTOR_PLAN=/path/to/rtmdet-fp16.plan \
OPENVISION_TRT_POSE_PLAN=/path/to/dwpose-fp16.plan \
OPENVISION_TRT_RG10_FIXTURE=/path/to/person.rg10 \
OPENVISION_TRT_TEMPORAL_EVALUATION_ROOT=/path/to/temporal-evaluation \
./scripts/prepare-wendy-runtime-probe.sh
```

The temporal root has the same `runtime-manifest.tsv` and `fixtures/*.rg10`
layout. Frames must be ordered in the manifest. Annotation labels are reported
only as evaluation metadata and are never passed to the recognizer. A
`completedNoGesture` status means the entire execution path completed but no
gesture reached the `ended` phase; it is not a successful recognition result.

Then build and run the arm64 container:

```bash
wendy run \
  --device 169.254.58.23 \
  --prefix Deployment/Wendy \
  --builder apple-container \
  --chunking off \
  --no-restart \
  --yes
```

`--chunking off` is currently required with Wendy CLI
`2026.07.17-173113`. Its chunk-diff deployment path imported all image layers
but replaced the OCI entrypoint with `/bin/sh` and omitted image environment
variables. The non-chunked deployment preserved the image entrypoint and both
plan digests, ran the complete probe, and exited with code zero.

Success requires CUDA device enumeration, one real TensorRT runtime
creation/destruction cycle, the representative frame transfer contract, and
the fused RG10 preprocessing contract. When both plans are supplied, success
also requires their exact SHA-256 digests, TensorRT deserialization, semantic
tensor compatibility, detector execution through the public Swift API,
reusable output addresses, GPU region-affine pose preparation, DWPose and SimCC
execution, OpenVision observation construction, sustained provider execution,
and the two expected negative failure paths.
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
black/gain calibration, a 3x3 color matrix, sRGB transfer, independent
red/green/blue affine normalization, NCHW/NHWC, RGB/BGR, and an independent
RGGB golden fixture. The
performance path converts one 1920x1080 SRGGB10 frame into a 256x256 tensor
with one H2D copy, one fused kernel, and no explicit frame-sized device
allocation after preparation. A second executable verifies the same operation
through the public Swift `VisionImageInput` and device-tensor lease API.

Verified on WendyOS 0.18.1 / JetPack 7.2:

```json
{"swiftProbe":{"status":"available","swiftPublicPath":"passed","h2dCopies":1,"kernelLaunches":1,"detectorInference":{"status":"passed","warmupIterations":10,"measuredIterations":100,"gpuP50Milliseconds":2.550144,"gpuP95Milliseconds":2.595040,"submissionCount":110,"persistentBytes":58800,"explicitFrameAllocations":0,"reusedOutputAddresses":true},"provider":{"status":"available","providerPath":"passed","inputReleased":true,"warmupIterations":5,"measuredIterations":30,"endToEndP50Milliseconds":11.659648,"endToEndP95Milliseconds":11.748704,"stableObservationCounts":true,"observationCount":4,"bodyJointCount":74,"handJointCount":138}},"engineProbe":{"status":"available","detector":[{"name":"input","shape":[1,3,320,320]},{"name":"dets","shape":[1,-1,5]},{"name":"labels","shape":[1,-1]}],"pose":[{"name":"input","shape":[-1,3,256,192],"profile":{"min":[1,3,256,192],"opt":[2,3,256,192],"max":[4,3,256,192]}},{"name":"simcc_x","shape":[-1,133,384]},{"name":"simcc_y","shape":[-1,133,512]}]},"checksumFailureProbe":{"status":"passed","failure":"engineChecksumMismatch","stage":"checksum"},"semanticMismatchProbe":{"status":"passed","failure":"incompatibleArtifact","tensor":"labels","reason":"elementType"}}
```

This proves the runtime boundary, a representative camera-sized host-to-GPU
transfer, the fused preprocessing kernel, and exact engine acceptance through
both the C ABI and public Swift API. It also proves sustained detector execution
with stable output allocations, GPU region-affine pose preparation, semantic
pose inference, and sustained provider execution from the RG10 fixture. It does
not claim that an actual camera lease or sustained camera capture has run.
