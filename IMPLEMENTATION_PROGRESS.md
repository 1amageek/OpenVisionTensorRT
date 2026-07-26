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

## Incomplete production work

- [ ] Select the semantic body/hand pose model and joint schema.
- [ ] Deserialize and validate a matching TensorRT engine.
- [ ] Implement reusable CUDA buffers and execution contexts.
- [ ] Implement RG10 GPU preprocessing.
- [ ] Implement request execution and observation decoding.
- [ ] Implement input-consumed CUDA synchronization.
- [ ] Conform the production provider to `VisionProvider`.
- [ ] Measure 1920x1080 RG10 at 30 FPS on Jetson.

No callable pose provider is declared yet, so no incomplete inference branch can
be mistaken for successful production execution.

## Verification evidence

| Check | Current evidence |
|---|---|
| macOS unavailable and lifecycle behavior | 3 tests passed with `xcodebuild test` and the 2026-07-17 Swift 6.4 snapshot |
| Jetson runtime | TensorRT 10.16.2 and CUDA 13.2 reported one device |
| TensorRT ownership | Real `IRuntime` creation and destruction passed on Jetson |
| Model inference | Not implemented; no semantic model has been selected |
