# OpenVisionTensorRT implementation guidance

- Read `../OpenVision/SPEC.md`, `DESIGN.md`, and
  `IMPLEMENTATION_PROGRESS.md`.
- Keep every CUDA, TensorRT, Linux, and Jetson type in this package.
- Public provider boundaries use OpenVision types only.
- C and C++ pointers stay behind opaque handles owned by Swift actors.
- Document owner, lifetime, initialized range, alignment, binding, aliasing,
  and synchronization before adding unsafe memory access.
- Do not advertise direct import without runtime evidence from the actual
  camera storage.
- Do not create a production provider that returns empty pose observations.
- Model choice requires an explicit user decision.
