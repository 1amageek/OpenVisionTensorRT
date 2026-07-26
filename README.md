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
- semantic-model versus TensorRT-engine artifact metadata.

Successful body or hand pose inference is not implemented until a semantic
model is selected. The package does not report an alternate CPU provider.

```text
OpenVision request
    -> OpenVisionTensorRT provider (next phase)
        -> CTensorRTShim
            -> CUDA / TensorRT
```

The transfer probe uses a 1920x1080 frame stored in a 16-bit raw container. It
registers the existing page-aligned host address, performs exactly one
asynchronous H2D copy per iteration, and retains the host owner until the CUDA
completion event has synchronized. A separate D2H copy is used only after
measurement to verify the device contents byte-for-byte.

```text
owned host storage
    -> cudaHostRegister(same address)
        -> cudaMemcpyAsync(H2D)
            -> cudaEventSynchronize(input consumed)
                -> source owner may be released
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
