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

    @Test("Transfer configuration rejects an empty frame")
    func emptyTransferFrame() {
        #expect(
            throws:
                CUDATransferProbeConfigurationError
                    .invalidByteCount(0)
        ) {
            _ = try CUDATransferProbeConfiguration(
                byteCount: 0,
                warmupIterationCount: 0,
                measuredIterationCount: 1
            )
        }
    }

    @Test("Transfer configuration requires measurements")
    func missingMeasuredIterations() {
        #expect(
            throws:
                CUDATransferProbeConfigurationError
                    .invalidMeasuredIterationCount(0)
        ) {
            _ = try CUDATransferProbeConfiguration(
                byteCount: 4_147_200,
                warmupIterationCount: 10,
                measuredIterationCount: 0
            )
        }
    }

    @Test("Transfer configuration enforces resource bounds")
    func transferResourceBounds() {
        let excessiveByteCount =
            CUDATransferProbeConfiguration.maximumByteCount + 1
        #expect(
            throws:
                CUDATransferProbeConfigurationError
                    .invalidByteCount(excessiveByteCount)
        ) {
            _ = try CUDATransferProbeConfiguration(
                byteCount: excessiveByteCount,
                warmupIterationCount: 0,
                measuredIterationCount: 1
            )
        }

        let excessiveIterationCount =
            CUDATransferProbeConfiguration
                .maximumIterationCount + 1
        #expect(
            throws:
                CUDATransferProbeConfigurationError
                    .invalidWarmupIterationCount(
                        excessiveIterationCount
                    )
        ) {
            _ = try CUDATransferProbeConfiguration(
                byteCount: 1,
                warmupIterationCount:
                    excessiveIterationCount,
                measuredIterationCount: 1
            )
        }
        #expect(
            throws:
                CUDATransferProbeConfigurationError
                    .invalidMeasuredIterationCount(
                        excessiveIterationCount
                    )
        ) {
            _ = try CUDATransferProbeConfiguration(
                byteCount: 1,
                warmupIterationCount: 0,
                measuredIterationCount:
                    excessiveIterationCount
            )
        }
    }

    @Test("RG10 transfer shape matches camera storage")
    func rg10TransferShape() throws {
        let configuration =
            try CUDATransferProbeConfiguration.rg10FullHD()

        #expect(configuration.byteCount == 4_147_200)
        #expect(configuration.warmupIterationCount == 10)
        #expect(configuration.measuredIterationCount == 100)
    }

    @Test("Transfer probe exposes typed unavailable evidence")
    func unavailableTransferProbe() throws {
        #if os(macOS)
        let configuration =
            try CUDATransferProbeConfiguration.rg10FullHD()

        do {
            _ = try CUDATransferProbe.run(
                configuration: configuration
            )
            Issue.record("Expected the CUDA runtime to be unavailable")
        } catch let error {
            guard case .unavailable(let result) = error else {
                Issue.record("Expected typed unavailable evidence")
                return
            }
            #expect(result.byteCount == configuration.byteCount)
            #expect(result.failureStage == .libraryOpen)
            #expect(!result.isTransferContractSatisfied)
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
