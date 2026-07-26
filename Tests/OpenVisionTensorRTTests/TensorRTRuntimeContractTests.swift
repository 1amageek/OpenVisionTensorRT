import OpenVision
@testable import OpenVisionTensorRT
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

    @Test("RG10 preprocessing configuration validates camera layout")
    func rg10PreprocessingConfiguration() throws {
        let configuration = try makeRG10Configuration()

        #expect(configuration.sourceWidth == 1920)
        #expect(configuration.sourceHeight == 1080)
        #expect(configuration.sourceBytesPerRow == 3840)
        #expect(configuration.sourceByteCount == 4_147_200)
        #expect(configuration.outputWidth == 256)
        #expect(configuration.outputHeight == 256)
    }

    @Test("RG10 preprocessing rejects a truncated source layout")
    func truncatedRG10Source() {
        #expect(
            throws:
                RG10PreprocessingConfigurationError
                    .invalidSourceByteCount(4_147_199)
        ) {
            _ = try RG10PreprocessingConfiguration(
                sourceWidth: 1920,
                sourceHeight: 1080,
                sourceBytesPerRow: 3840,
                sourceByteCount: 4_147_199,
                outputWidth: 256,
                outputHeight: 256,
                resizePolicy: .scaleFit,
                tensorLayout: .channelsFirst,
                channelOrder: .rgb,
                blackLevels: .init(
                    red: 0,
                    greenOnRedRow: 0,
                    greenOnBlueRow: 0,
                    blue: 0
                ),
                whiteLevel: 1023,
                gains: .init(
                    red: 1,
                    greenOnRedRow: 1,
                    greenOnBlueRow: 1,
                    blue: 1
                ),
                colorMatrix: .identity,
                normalization: .zeroToOne,
                appliesSRGBTransfer: false
            )
        }
    }

    @Test("RG10 GPU preparation is a typed unavailable failure on Mac")
    func unavailableRG10Preprocessor() throws {
        #if os(macOS)
        let configuration = try makeRG10Configuration()
        do {
            _ = try RG10Preprocessor(configuration: configuration)
            Issue.record("Expected the CUDA preprocessor to be unavailable")
        } catch let error {
            guard case .unavailable(let report) = error else {
                Issue.record("Expected typed unavailable evidence")
                return
            }
            #expect(report.failureStage == .libraryOpen)
            #expect(
                report.libraryOpenFailures == [
                    .cudaRuntime,
                    .cudaDriver,
                    .nvrtc
                ]
            )
            #expect(report.outputElementCount == 196_608)
        }
        #endif
    }

    @Test("RG10 configuration enforces C ABI and resource bounds")
    func rg10ConfigurationBounds() {
        #expect(
            throws:
                RG10PreprocessingConfigurationError
                    .invalidSourceDimensions(
                        width: Int(UInt32.max) + 1,
                        height: 2
                    )
        ) {
            _ = try makeRG10Configuration(
                sourceWidth: Int(UInt32.max) + 1,
                sourceHeight: 2,
                sourceBytesPerRow: 4,
                sourceByteCount: 8
            )
        }
        #expect(
            throws:
                RG10PreprocessingConfigurationError
                    .invalidSourceBytesPerRow(
                        Int(UInt32.max) + 1
                    )
        ) {
            _ = try makeRG10Configuration(
                sourceWidth: 2,
                sourceHeight: 2,
                sourceBytesPerRow: Int(UInt32.max) + 1,
                sourceByteCount: 8
            )
        }
        #expect(
            throws:
                RG10PreprocessingConfigurationError
                    .invalidSourceByteCount(
                        RG10PreprocessingConfiguration
                            .maximumSourceByteCount + 1
                    )
        ) {
            _ = try makeRG10Configuration(
                sourceWidth: 2,
                sourceHeight: 2,
                sourceBytesPerRow: 4,
                sourceByteCount:
                    RG10PreprocessingConfiguration
                        .maximumSourceByteCount + 1
            )
        }
        #expect(
            throws:
                RG10PreprocessingConfigurationError
                    .invalidOutputDimensions(
                        width: 4097,
                        height: 4096
                    )
        ) {
            _ = try makeRG10Configuration(
                outputWidth: 4097,
                outputHeight: 4096
            )
        }
    }

    @Test("Tensor lease excludes release while borrowed")
    func tensorLease() throws {
        let lease = RG10TensorLeaseState()
        _ = try lease.withBorrow {
            #expect(
                throws: RG10DeviceTensorError.borrowInProgress
            ) {
                try lease.release()
            }
        }
        try lease.release()
        #expect(lease.isReleased)
        #expect(throws: RG10DeviceTensorError.released) {
            try lease.withBorrow {}
        }
    }

    private func makeRG10Configuration()
        throws -> RG10PreprocessingConfiguration
    {
        try makeRG10Configuration(
            sourceWidth: 1920,
            sourceHeight: 1080,
            sourceBytesPerRow: 3840,
            sourceByteCount: 4_147_200,
            outputWidth: 256,
            outputHeight: 256
        )
    }

    private func makeRG10Configuration(
        sourceWidth: Int = 1920,
        sourceHeight: Int = 1080,
        sourceBytesPerRow: Int = 3840,
        sourceByteCount: Int = 4_147_200,
        outputWidth: Int = 256,
        outputHeight: Int = 256
    ) throws -> RG10PreprocessingConfiguration {
        try RG10PreprocessingConfiguration(
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            sourceBytesPerRow: sourceBytesPerRow,
            sourceByteCount: sourceByteCount,
            outputWidth: outputWidth,
            outputHeight: outputHeight,
            resizePolicy: .scaleFit,
            tensorLayout: .channelsFirst,
            channelOrder: .rgb,
            blackLevels: .init(
                red: 0,
                greenOnRedRow: 0,
                greenOnBlueRow: 0,
                blue: 0
            ),
            whiteLevel: 1023,
            gains: .init(
                red: 1,
                greenOnRedRow: 1,
                greenOnBlueRow: 1,
                blue: 1
            ),
            colorMatrix: .identity,
            normalization: .zeroToOne,
            appliesSRGBTransfer: false
        )
    }
}
