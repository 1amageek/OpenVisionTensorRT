# Wendy TensorRT Runtime Probe

This deployment validates the real `CTensorRTShim` path against the
CUDA/TensorRT libraries exposed by Wendy's GPU entitlement. It does not contain
or select a vision model.

Prepare the exact package sources as the Docker build context:

```bash
./scripts/prepare-wendy-runtime-probe.sh
```

Then build and run the arm64 container:

```bash
wendy run \
  --device 169.254.58.23 \
  --prefix Deployment/Wendy \
  --builder apple-container \
  --no-restart \
  --yes
```

Success requires CUDA device enumeration plus one real TensorRT runtime
creation/destruction cycle. Missing libraries, inaccessible GPU devices, and
runtime creation failures exit nonzero.

Verified on WendyOS 0.18.1 / JetPack 7.2:

```json
{"status":"available","tensorRTVersion":101602,"cudaRuntimeVersion":13020,"cudaDriverVersion":13020,"cudaDeviceCount":1,"runtimeLifecycle":"passed"}
```

This is runtime-boundary evidence only. It does not claim that a model engine,
camera-frame transfer, preprocessing, or pose inference has run.
