# OpenVisionTensorRT Design

## Boundary

OpenVisionTensorRT imports OpenVision. OpenVision never imports this package.
Camera capture remains in OpenAVFoundation camera drivers.

The C++ boundary exposes opaque handles only. Swift actors own those handles,
and raw TensorRT or CUDA pointers never appear in public Swift API.

On Linux, the shim resolves CUDA 13 and TensorRT 10 symbols with `dlopen` and
`dlsym`. This keeps the Swift package buildable on non-NVIDIA hosts without
inventing a successful fallback. Missing libraries, missing symbols, version
probe failures, and runtime creation failures remain distinct typed failures.
The scoped C++ loader owns every library handle and closes it only after all
TensorRT objects have been destroyed.

## Runtime owner

`OVTRTRuntime` owns its logger and `nvinfer1::IRuntime`. The logger is declared
first so it outlives the runtime. `ovtrt_runtime_create` transfers ownership
only after all allocations and TensorRT initialization succeed.
`ovtrt_runtime_destroy` consumes the owner exactly once.

The Swift `TensorRTRuntime` actor serializes lifecycle transitions. Explicit
shutdown clears the handle before calling the C destructor, preventing
reentrant double destruction. Actor deinitialization destroys an unconsumed
handle.

## Planned hot path

```text
RG10 V4L2 lease
    -> one synchronous H2D transfer inside scoped Span borrow
        -> release source input
            -> CUDA demosaic / rotate / resize / normalize
                -> preallocated TensorRT execution context
                    -> compact OpenVision joints
```

Input, preprocessing, output, and workspace device allocations will be created
during model preparation and reused. No frame-sized allocation is allowed after
warm-up. Direct import will be advertised only after a DMA-BUF or external
memory path is proven on the actual Jetson camera storage.

## Performance acceptance budget

The first production pipeline has the following measurable budget:

| Item | Budget |
|---|---|
| CPU frame materialization | 0 copies |
| Host-to-device transfer | exactly 1 full-frame transfer |
| Frame-sized allocation after warm-up | 0 |
| In-flight inference | 1, with explicit busy failure |
| Sustained input rate | 1920x1080 RG10 at 30 FPS |
| End-to-end inference latency | p95 below 33.3 ms |

The source lease may be released only after the CUDA event proving that the H2D
transfer no longer reads host memory. Preprocessing, inference, and decoding
reuse a prepared execution slot. The eventual model-specific implementation
must report copy count, allocation count, p50/p95 latency, host/device resident
memory, power, and thermal state; runtime creation alone does not satisfy this
budget.

## Verified Jetson runtime boundary

The Wendy deployment cross-compiles the exact package shim with pinned
TensorRT 10.16 headers and executes it using the Jetson GPU entitlement. On
WendyOS 0.18.1 / JetPack 7.2 it verified TensorRT 10.16.2, CUDA runtime and
driver 13.2, one CUDA device, and one real runtime creation/destruction cycle.
This proves the runtime ownership boundary, not semantic pose inference.
