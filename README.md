# OpenVisionTensorRT

OpenVisionTensorRT is the NVIDIA CUDA and TensorRT provider package for
OpenVision. Jetson is the first deployment environment.

The current implementation owns:

- CUDA and TensorRT runtime probing;
- dynamic loading of the installed Jetson CUDA/TensorRT libraries;
- an opaque C ABI for TensorRT runtime ownership;
- exactly-once runtime destruction;
- semantic-model versus TensorRT-engine artifact metadata.

Successful body or hand pose inference is not implemented until a semantic
model is selected. The package does not report an alternate CPU provider.

```text
OpenVision request
    -> OpenVisionTensorRT provider (next phase)
        -> CTensorRTShim
            -> CUDA / TensorRT
```

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
