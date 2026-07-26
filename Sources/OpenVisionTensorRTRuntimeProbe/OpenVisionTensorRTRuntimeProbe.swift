import OpenVisionTensorRT

@main
enum OpenVisionTensorRTRuntimeProbe {
    static func main() async throws {
        let probe = TensorRTRuntimeProbe.current()
        print("status=\(probe.status)")
        print("tensorrt=\(probe.tensorRTVersion)")
        print("cuda-runtime=\(probe.cudaRuntimeVersion)")
        print("cuda-driver=\(probe.cudaDriverVersion)")
        print("cuda-devices=\(probe.cudaDeviceCount)")

        guard probe.isAvailable else {
            throw TensorRTRuntimeError.unavailable(probe)
        }

        let runtime = try TensorRTRuntime()
        try await runtime.shutdown()
    }
}
