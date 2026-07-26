import OpenVision
import OpenVisionTensorRT
import Testing

@Suite("TensorRT runtime boundary")
struct TensorRTRuntimeContractTests {
    @Test("Non-NVIDIA host reports unavailable without fallback")
    func unavailableHost() {
        let probe = TensorRTRuntimeProbe.current()

        #if os(macOS)
        #expect(probe.status == .unavailable)
        #expect(!probe.isAvailable)
        #expect(probe.cudaDeviceCount == 0)
        #endif
    }

    @Test("Runtime creation is a typed failure when unavailable")
    func unavailableRuntimeCreation() {
        #if os(macOS)
        let probe = TensorRTRuntimeProbe.current()
        #expect(throws: TensorRTRuntimeError.unavailable(probe)) {
            _ = try TensorRTRuntime()
        }
        #endif
    }

    @Test("Engine artifact rejects missing compatibility evidence")
    func artifactValidation() throws {
        let model = VisionModelDescriptor(
            id: "fixture",
            revision: "1",
            request: .detectHumanBodyPoseRequest(.revision2),
            input: try VisionModelInputDescriptor(
                width: 256,
                height: 256,
                pixelFormat: .rgba32,
                resizePolicy: .scaleFit,
                normalization: .zeroToOne
            ),
            outputSchemaRevision: "body-1"
        )

        #expect(throws: TensorRTEngineArtifactError.emptyChecksum) {
            _ = try TensorRTEngineArtifactDescriptor(
                semanticModel: model,
                checksum: "",
                tensorRTVersion: 10_000,
                cudaRuntimeVersion: 12_000,
                computeCapabilityMajor: 8,
                computeCapabilityMinor: 7,
                precision: .float16
            )
        }
    }
}
