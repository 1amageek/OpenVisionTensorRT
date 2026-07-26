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

## Incomplete production work

- [ ] Select the semantic body/hand pose model and joint schema.
- [ ] Deserialize and validate a matching TensorRT engine.
- [ ] Implement reusable CUDA buffers and execution contexts.
- [ ] Implement RG10 GPU preprocessing.
- [ ] Implement request execution and observation decoding.
- [ ] Connect input-consumed CUDA synchronization to a camera frame lease.
- [ ] Conform the production provider to `VisionProvider`.
- [ ] Measure 1920x1080 RG10 at 30 FPS on Jetson.

No callable pose provider is declared yet, so no incomplete inference branch can
be mistaken for successful production execution.

## Verification evidence

| Check | Current evidence |
|---|---|
| macOS unavailable, configuration, and lifecycle behavior | 8 tests passed with `xcodebuild test` and the 2026-07-17 Swift 6.4 snapshot |
| macOS sanitizer behavior | The same 8 tests passed with Address Sanitizer and Thread Sanitizer |
| Regular WASM | Debug and release `OpenVisionTensorRT` target builds passed with the matching 2026-07-17 SDK |
| Embedded WASM | Debug and release `OpenVisionTensorRT` target builds passed with the matching 2026-07-17 SDK |
| Jetson runtime | TensorRT 10.16.2 and CUDA 13.2 reported one device |
| Jetson sanitizer behavior | The full runtime and transfer path passed ASan and UBSan; leak detection is unavailable under Wendy device supervision |
| TensorRT ownership | Real `IRuntime` creation and destruction passed on Jetson |
| Representative H2D transfer | 4,147,200 bytes; 110/110 copies; p50 0.170080 ms; p95 0.171040 ms |
| Representative H2D throughput | p50 24.384 GB/s; p95 24.247 GB/s |
| Transfer ownership | Host registration, source-address preservation, input-consumed event, and byte verification passed |
| Transfer allocation | Two pre-warm-up host buffers, one device buffer, zero frame-sized allocations after warm-up |
| Model inference | Not implemented; no semantic model has been selected |
