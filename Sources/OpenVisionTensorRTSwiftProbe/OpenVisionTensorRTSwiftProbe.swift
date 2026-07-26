import OpenVision
import OpenVisionTensorRT
import Synchronization

@main
enum OpenVisionTensorRTSwiftProbe {
    static func main() async throws {
        #if os(Linux)
        try await run()
        #else
        print("{\"status\":\"unavailable\"}")
        #endif
    }

    #if os(Linux)
    private static func run() async throws {
            guard
                CommandLine.arguments.count == 1 || CommandLine.arguments.count == 3
            else {
                throw SwiftProbeError.invalidArguments
            }
            let runsDetector = CommandLine.arguments.count == 3
            let modelInput: VisionModelInputDescriptor?
            if runsDetector {
                let manifest =
                    try RTMDetDWPoseBodyPoseManifest.manifest()
                guard
                    let stage = manifest.stage(
                        identifiedBy:
                            RTMDetDWPoseBodyPoseManifest
                            .personDetectionStage
                    )
                else {
                    throw SwiftProbeError.contractViolation
                }
                modelInput = stage.input
            } else {
                modelInput = nil
            }
        let width = 1920
        let height = 1080
        let bytesPerRow = width * 2
        let byteCount = bytesPerRow * height
        let storage = try RG10ProbeStorage(
            byteCount: byteCount,
            alignment: 4096
        )
        try storage.initializeRG10(
            width: width,
            height: height,
            bytesPerRow: bytesPerRow
        )
        let dimensions = try CVPixelDimensions(
            width: width,
            height: height
        )
        let pixelFormat = CVPixelFormatType(
            rawValue:
                RG10PreprocessingConfiguration.pixelFormatRawValue
        )
        let layout = try CVPackedPixelBufferLayout(
            dimensions: dimensions,
            pixelFormat: pixelFormat,
            bytesPerPixel: 2,
            bytesPerRow: bytesPerRow
        )
        let pixelBuffer = PortableRG10PixelBuffer(
            layout: layout,
            storage: storage
        )
        let sample = try CMImageSampleBuffer(
            imageBuffer: pixelBuffer,
            formatDescription: CMImmutableVideoFormatDescription(
                dimensions: dimensions,
                pixelFormat: pixelFormat
            ),
            timing: CMSampleTimingInfo(
                duration: CMTime(value: 1, timescale: 30),
                presentationTimeStamp: .zero,
                decodeTimeStamp: .invalid
            )
        )
        let clock = VisionClockDomain(
            id: "jetson-swift-probe",
            epoch: 1,
            kind: .deviceMonotonic
        )
        let validity = try VisionTimeRange(
            range: CMTimeRange(
                start: .zero,
                duration: CMTime(value: 1, timescale: 1)
            ),
            clockDomain: clock
        )
        let calibrationReference = VisionCalibrationReference(
            id: "jetson-swift-probe",
            revision: 1
        )
        let calibration = try VisionCameraCalibration(
            reference: calibrationReference,
            source: "jetson-swift-probe",
            calibratedAt: try VisionTimestamp(
                time: .zero,
                clockDomain: clock
            ),
            validity: validity,
            intrinsics: VisionCameraIntrinsics(
                matrix: .identity,
                referenceDimensions: dimensions
            )
        )
        let input = try VisionImageInput(
            sampleBuffer: sample,
            frameID: VisionFrameID(
                source: "jetson-swift-probe",
                sequence: 1
            ),
            clockDomain: clock,
            calibration: calibration
        )
        let configuration = try RG10PreprocessingConfiguration(
            sourceWidth: width,
            sourceHeight: height,
            sourceBytesPerRow: bytesPerRow,
            sourceByteCount: byteCount,
                outputWidth: modelInput?.width ?? 256,
                outputHeight: modelInput?.height ?? 256,
                resizePolicy: modelInput?.resizePolicy ?? .scaleFit,
                tensorLayout:
                    modelInput?.tensorLayout ?? .channelsFirst,
                channelOrder:
                    modelInput?.channelOrder ?? .rgb,
            blackLevels: RG10BayerValues(
                red: 0,
                greenOnRedRow: 0,
                greenOnBlueRow: 0,
                blue: 0
            ),
            whiteLevel: 1023,
            gains: RG10BayerValues(
                red: 1,
                greenOnRedRow: 1,
                greenOnBlueRow: 1,
                blue: 1
            ),
            colorMatrix: .identity,
                letterboxColor: RGBTriplet(
                    red: modelInput?.letterboxColor.red ?? 0,
                    green: modelInput?.letterboxColor.green ?? 0,
                    blue: modelInput?.letterboxColor.blue ?? 0
                ),
                normalization:
                    modelInput?.normalization ?? .zeroToOne,
                appliesSRGBTransfer:
                    modelInput?.transferFunction == .sRGB
        )
        let preprocessor = try RG10Preprocessor(
            configuration: configuration
        )
        let output = try await preprocessor.process(input)
        guard
            output.report.satisfiesFrameContract,
                output.tensor.width == (modelInput?.width ?? 256),
                output.tensor.height == (modelInput?.height ?? 256),
            output.tensor.channelCount == 3,
                output.tensor.elementCount == (modelInput?.width ?? 256)
                    * (modelInput?.height ?? 256)
                    * 3,
            input.isReleased
        else {
            throw SwiftProbeError.contractViolation
        }
        let addressIsNonzero = try output.tensor.withDeviceAddress {
            address,
            byteCount in
                guard
                    address != 0,
                    byteCount == (modelInput?.width ?? 256)
                        * (modelInput?.height ?? 256)
                        * 3 * 4
                else {
                throw SwiftProbeError.contractViolation
            }
            return true
        }
            let inferenceDescription: String
            if runsDetector {
                inferenceDescription = try await runDetector(
                    input: output.tensor,
                    path: CommandLine.arguments[1],
                    checksum: CommandLine.arguments[2]
                )
            } else {
                inferenceDescription = "\"notRequested\""
            }
        try output.tensor.release()
        try await preprocessor.shutdown()
        print(
            "{\"status\":\"available\","
                + "\"swiftPublicPath\":\"passed\","
                + "\"deviceAddressNonzero\":"
                + (addressIsNonzero ? "true" : "false")
                + ",\"inputReleased\":true,"
                + "\"h2dCopies\":"
                + String(
                    output.report.fullFrameHostToDeviceCopyCount
                )
                + ",\"kernelLaunches\":"
                + String(output.report.kernelLaunchCount)
                    + ",\"detectorInference\":"
                    + inferenceDescription
                + "}"
        )
    }

        private static func runDetector(
            input: RG10DeviceTensor,
            path: String,
            checksum: String
        ) async throws -> String {
            let manifest =
                try RTMDetDWPoseBodyPoseManifest.manifest()
            let artifact = try TensorRTEngineArtifactDescriptor(
                semanticModel: manifest,
                checksum: checksum,
                tensorRTVersion: 101_602,
                cudaRuntimeVersion: 13_020,
                computeCapabilityMajor: 8,
                computeCapabilityMinor: 7,
                precision: .float16
            )
            let stage = try TensorRTStageEngineArtifactDescriptor(
                artifact: artifact,
                stageID:
                    RTMDetDWPoseBodyPoseManifest
                    .personDetectionStage,
                inputTensorName: "input",
                outputBindings: [
                    try TensorRTEngineOutputBinding(
                        semanticTensorID:
                            RTMDetDWPoseBodyPoseManifest
                            .detectionsTensor,
                        engineTensorName: "dets"
                    ),
                    try TensorRTEngineOutputBinding(
                        semanticTensorID:
                            RTMDetDWPoseBodyPoseManifest
                            .classesTensor,
                        engineTensorName: "labels"
                    ),
                ]
            )
            let engine = try TensorRTEngine(
                path: path,
                artifact: stage
            )
            let preparation = try await engine.prepareExecution()
            guard
                preparation.persistentDeviceAllocationCount == 2,
                preparation.explicitFrameDeviceAllocationCount == 0
            else {
                throw SwiftProbeError.contractViolation
            }
            let warmupIterationCount = 10
            let measuredIterationCount = 100
            let totalIterationCount =
                warmupIterationCount + measuredIterationCount
            var measurements: [Float] = []
            measurements.reserveCapacity(measuredIterationCount)
            var outputAddresses: [UInt]?
            var detectionCount = 0
            for iteration in 0..<totalIterationCount {
                let inference = try await engine.execute(input)
                guard
                    inference.report.explicitFrameDeviceAllocationCount == 0,
                    inference.report.submissionCount == UInt64(iteration + 1),
                    inference.tensors.count == 2,
                    inference.tensors[0].name == "dets",
                    inference.tensors[0].shape.count == 3,
                    inference.tensors[0].shape[0] == 1,
                    inference.tensors[0].shape[2] == 5,
                    inference.tensors[1].name == "labels",
                    inference.tensors[1].shape.count == 2,
                    inference.tensors[1].shape[0] == 1,
                    inference.tensors[1].shape[1] == inference.tensors[0].shape[1]
                else {
                    throw SwiftProbeError.contractViolation
                }
                var currentAddresses: [UInt] = []
                currentAddresses.reserveCapacity(
                    inference.tensors.count
                )
                for tensor in inference.tensors {
                    currentAddresses.append(
                        try tensor.withDeviceAddress {
                            address,
                            _ in address
                        }
                    )
                }
                if let outputAddresses {
                    guard currentAddresses == outputAddresses else {
                        throw SwiftProbeError.contractViolation
                    }
                } else {
                    outputAddresses = currentAddresses
                }
                detectionCount = inference.tensors[0].shape[1]
                if iteration >= warmupIterationCount {
                    measurements.append(
                        inference.report.inferenceMilliseconds
                    )
                }
                try inference.release()
            }
            guard
                measurements.count == measuredIterationCount,
                let outputAddresses,
                outputAddresses.count == 2
            else {
                throw SwiftProbeError.contractViolation
            }
            let sortedMeasurements = measurements.sorted()
            let p50 = percentile(
                sortedMeasurements,
                percentile: 0.50
            )
            let p95 = percentile(
                sortedMeasurements,
                percentile: 0.95
            )
            try await engine.shutdown()
            return "{\"status\":\"passed\","
                + "\"warmupIterations\":"
                + String(warmupIterationCount)
                + ",\"measuredIterations\":"
                + String(measuredIterationCount)
                + ",\"gpuP50Milliseconds\":"
                + String(p50)
                + ",\"gpuP95Milliseconds\":"
                + String(p95)
                + ",\"submissionCount\":"
                + String(totalIterationCount)
                + ",\"persistentBytes\":"
                + String(
                    preparation.persistentDeviceAllocationByteCount
                )
                + ",\"explicitFrameAllocations\":"
                + String(preparation.explicitFrameDeviceAllocationCount)
                + ",\"reusedOutputAddresses\":true"
                + ",\"detectionCount\":"
                + String(detectionCount)
                + "}"
        }

        private static func percentile(
            _ sortedValues: [Float],
            percentile: Double
        ) -> Float {
            precondition(!sortedValues.isEmpty)
            let scaledIndex =
                percentile
                * Double(sortedValues.count - 1)
            let index = Int(scaledIndex.rounded(.up))
            return sortedValues[index]
        }
    #endif
}

private enum SwiftProbeError: Error {
    case invalidArguments
    case allocationFailed
    case storageReleased
    case contractViolation
}

private final class RG10ProbeStorage: Sendable {
    // This object is the exactly-once allocation owner. State stores the
    // address as an integer so no pointer crosses a Sendable boundary.
    // Pointer reconstruction is scoped to the borrow closures, which validate
    // byteCount before constructing a Span. Deinitialization clears the
    // address under the same Mutex before deallocating it once.
    private struct State: Sendable {
        var address: UInt?
    }

    let byteCount: Int
    private let state: Mutex<State>

    init(
        byteCount: Int,
        alignment: Int
    ) throws(SwiftProbeError) {
        guard byteCount > 0, alignment > 0 else {
            throw .allocationFailed
        }
        let pointer = UnsafeMutableRawPointer.allocate(
            byteCount: byteCount,
            alignment: alignment
        )
        self.byteCount = byteCount
        state = Mutex(State(address: UInt(bitPattern: pointer)))
    }

    deinit {
        let address = state.withLock { state -> UInt? in
            let address = state.address
            state.address = nil
            return address
        }
        if let address,
           let pointer = UnsafeMutableRawPointer(bitPattern: address)
        {
            pointer.deallocate()
        }
    }

    func initializeRG10(
        width: Int,
        height: Int,
        bytesPerRow: Int
    ) throws(SwiftProbeError) {
        try withMutableBytes { bytes in
            for y in 0..<height {
                for x in 0..<width {
                    let value = UInt16(
                        (x * 17 + y * 31 + (x ^ y)) & 1023
                    )
                    let offset = y * bytesPerRow + x * 2
                    bytes[offset] = UInt8(truncatingIfNeeded: value)
                    bytes[offset + 1] = UInt8(
                        truncatingIfNeeded: value >> 8
                    )
                }
            }
        }
    }

    func withReadBytes(
        _ body: (borrowing Span<UInt8>) -> Void
    ) throws(SwiftProbeError) {
        let address = state.withLock { $0.address }
        guard
            let address,
            let pointer = UnsafeRawPointer(bitPattern: address)
        else {
            throw .storageReleased
        }
        body(
            Span(
                _unsafeStart:
                    pointer.assumingMemoryBound(to: UInt8.self),
                count: byteCount
            )
        )
    }

    private func withMutableBytes(
        _ body: (inout MutableSpan<UInt8>) -> Void
    ) throws(SwiftProbeError) {
        let address = state.withLock { $0.address }
        guard
            let address,
            let pointer =
                UnsafeMutableRawPointer(bitPattern: address)
        else {
            throw .storageReleased
        }
        var bytes = MutableSpan(
            _unsafeStart:
                pointer.assumingMemoryBound(to: UInt8.self),
            count: byteCount
        )
        body(&bytes)
    }
}

private final class PortableRG10PixelBuffer: CVPixelBuffer {
    let layout: CVPackedPixelBufferLayout
    let accessCapabilities: CVPixelBufferAccessCapabilities = [.read]
    let attachments = CVBufferAttachments()

    private let storage: RG10ProbeStorage

    var dimensions: CVPixelDimensions {
        layout.dimensions
    }

    var pixelFormat: CVPixelFormatType {
        layout.pixelFormat
    }

    var bytesPerRow: Int {
        layout.bytesPerRow
    }

    var byteCount: Int {
        storage.byteCount
    }

    init(
        layout: CVPackedPixelBufferLayout,
        storage: RG10ProbeStorage
    ) {
        self.layout = layout
        self.storage = storage
    }

    func withReadBytes(
        _ body: (borrowing Span<UInt8>) -> Void
    ) throws(CVPixelBufferError) {
        do {
            try storage.withReadBytes(body)
        } catch {
            throw .storageReleased
        }
    }

    func withWriteBytes(
        _ body: (inout MutableSpan<UInt8>) -> Void
    ) throws(CVPixelBufferError) {
        throw .unsupportedAccess(.write)
    }
}
