import ActionRecognition
import OpenVision
import OpenVisionTensorRT
import OpenVisionTensorRTEvaluation
import Synchronization

#if canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

@main
enum OpenVisionTensorRTDatasetEvaluator {
    static func main() async {
        #if os(Linux)
        do {
            let output = try await DatasetEvaluationRunner(
                arguments: Array(CommandLine.arguments.dropFirst())
            ).run()
            print(output)
        } catch {
            let failure = DatasetEvaluationFailure(
                status: "failed",
                reason: String(describing: error)
            )
            print(failure.json)
            exit(1)
        }
        #else
        print("{\"status\":\"unavailable\",\"reason\":\"unsupportedPlatform\"}")
        #endif
    }
}

#if os(Linux)
private struct DatasetEvaluationRunner {
    private enum Mode: Sendable {
        case independentFrames
        case temporalHorizontalSwipe
    }

    private let mode: Mode
    private let detectorPath: String
    private let detectorChecksum: String
    private let posePath: String
    private let poseChecksum: String
    private let manifestPath: String

    init(arguments: [String]) throws {
        guard arguments.count == 6 else {
            throw DatasetEvaluationError.invalidArguments
        }
        switch arguments[0] {
        case "--evaluate":
            mode = .independentFrames
        case "--recognize-horizontal-swipe":
            mode = .temporalHorizontalSwipe
        default:
            throw DatasetEvaluationError.invalidArguments
        }
        detectorPath = arguments[1]
        detectorChecksum = arguments[2]
        posePath = arguments[3]
        poseChecksum = arguments[4]
        manifestPath = arguments[5]
    }

    func run() async throws -> String {
        let evaluationManifest = try EvaluationManifest.load(at: manifestPath)
        let records = evaluationManifest.records
        guard !records.isEmpty else {
            throw DatasetEvaluationError.emptyManifest
        }

        let manifest = try RTMDetDWPoseBodyPoseManifest.manifest()
        guard let detectorInput = manifest.stage(
            identifiedBy: RTMDetDWPoseBodyPoseManifest.personDetectionStage
        )?.input else {
            throw DatasetEvaluationError.missingDetectorStage
        }
        let preprocessing = try RG10PreprocessingConfiguration(
            sourceWidth: 1_920,
            sourceHeight: 1_080,
            sourceBytesPerRow: 3_840,
            sourceByteCount: 4_147_200,
            wordLayout: .leastSignificantBits,
            outputWidth: detectorInput.width,
            outputHeight: detectorInput.height,
            resizePolicy: detectorInput.resizePolicy,
            tensorLayout: detectorInput.tensorLayout,
            channelOrder: detectorInput.channelOrder,
            blackLevels: RG10BayerValues(
                red: 0,
                greenOnRedRow: 0,
                greenOnBlueRow: 0,
                blue: 0
            ),
            whiteLevel: 1_023,
            gains: RG10BayerValues(
                red: 1,
                greenOnRedRow: 1,
                greenOnBlueRow: 1,
                blue: 1
            ),
            colorMatrix: .identity,
            letterboxColor: RGBTriplet(
                red: detectorInput.letterboxColor.red,
                green: detectorInput.letterboxColor.green,
                blue: detectorInput.letterboxColor.blue
            ),
            normalization: detectorInput.normalization,
            appliesSRGBTransfer: detectorInput.transferFunction == .sRGB
        )
        let providerConfiguration = try OpenVisionTensorRTProviderConfiguration(
            model: manifest,
            detectorPlanPath: detectorPath,
            detectorArtifact: try stageArtifact(
                manifest: manifest,
                checksum: detectorChecksum,
                stage: RTMDetDWPoseBodyPoseManifest.personDetectionStage,
                bindings: [
                    (
                        RTMDetDWPoseBodyPoseManifest.detectionsTensor,
                        "dets",
                        OpenVisionTensorRTProviderConfiguration
                            .detectorExecutionCandidateCount * 5
                    ),
                    (
                        RTMDetDWPoseBodyPoseManifest.classesTensor,
                        "labels",
                        OpenVisionTensorRTProviderConfiguration
                            .detectorExecutionCandidateCount
                    ),
                ]
            ),
            posePlanPath: posePath,
            poseArtifact: try stageArtifact(
                manifest: manifest,
                checksum: poseChecksum,
                stage: RTMDetDWPoseBodyPoseManifest.wholeBodyPoseStage,
                bindings: [
                    (
                        RTMDetDWPoseBodyPoseManifest.simCCXTensor,
                        "simcc_x",
                        nil
                    ),
                    (
                        RTMDetDWPoseBodyPoseManifest.simCCYTensor,
                        "simcc_y",
                        nil
                    ),
                ]
            ),
            detectorPreprocessing: preprocessing
        )
        let provider = try OpenVisionTensorRTProvider(
            configuration: providerConfiguration
        )
        let sessionConfiguration = VisionSessionConfiguration(
            model: manifest,
            transferMode: .stagedHostToDevice(fullFrameCopyCount: 1),
            computeDevices: [
                .main: OpenVisionTensorRTProvider.cudaDevice,
                .postProcessing: OpenVisionTensorRTProvider.cudaDevice,
            ]
        )

        if case .temporalHorizontalSwipe = mode {
            let outcome: TemporalEvaluationOutcome
            do {
                outcome = try await VisionContext.withProvider(
                    provider,
                    configuration: sessionConfiguration
                ) {
                    await evaluateTemporal(records: records)
                }
            } catch {
                throw DatasetEvaluationError.visionContext(
                    String(describing: error)
                )
            }
            switch outcome {
            case .success(let output):
                return output.json
            case .failure(let reason):
                throw DatasetEvaluationError.temporalEvaluation(reason)
            }
        }

        let session = try await provider.makeSession(
            configuration: sessionConfiguration
        )

        let operation: Result<[DatasetFrameEvaluation], any Error>
        do {
            operation = .success(
                try await evaluate(records: records, session: session)
            )
        } catch {
            operation = .failure(error)
        }

        let shutdownFailure: (any Error)?
        do {
            try await session.shutdown()
            shutdownFailure = nil
        } catch {
            shutdownFailure = error
        }

        switch (operation, shutdownFailure) {
        case (.success(let frames), nil):
            let hasGroundTruthJoints = records.contains {
                !$0.groundTruthJoints.isEmpty
            }
            return DatasetEvaluationOutput(
                status: "passed",
                source: evaluationManifest.source
                    ?? (hasGroundTruthJoints
                        ? "Joint-annotated pose dataset evaluation"
                        : "Detection-only dataset evaluation"),
                limitations: evaluationManifest.limitations.isEmpty
                    ? (hasGroundTruthJoints
                        ? [
                            "The manifest does not declare dataset-specific limitations.",
                            "The provider is configured for at most four people per frame.",
                        ]
                        : [
                            "The manifest has no joint ground truth, so pose geometry is not scored.",
                            "The provider is configured for at most four people per frame.",
                        ])
                    : evaluationManifest.limitations,
                summary: DatasetEvaluationSummary(frames: frames),
                frames: frames
            ).json
        case (.failure(let error), nil):
            throw error
        case (.success, .some(let error)):
            throw DatasetEvaluationError.shutdown(String(describing: error))
        case (.failure(let operationError), .some(let shutdownError)):
            throw DatasetEvaluationError.operationAndShutdown(
                operation: String(describing: operationError),
                shutdown: String(describing: shutdownError)
            )
        }
    }

    private func evaluate(
        records: [EvaluationManifestRecord],
        session: any VisionProviderSession
    ) async throws -> [DatasetFrameEvaluation] {
        let dimensions = try CVPixelDimensions(width: 1_920, height: 1_080)
        let pixelFormat = CVPixelFormatType(
            rawValue: RG10PreprocessingConfiguration.pixelFormatRawValue
        )
        let layout = try CVPackedPixelBufferLayout(
            dimensions: dimensions,
            pixelFormat: pixelFormat,
            bytesPerPixel: 2,
            bytesPerRow: 3_840
        )
        let clock = VisionClockDomain(
            id: "dataset-evaluation",
            epoch: 1,
            kind: .deviceMonotonic
        )
        var request = DetectHumanBodyPoseRequest(.revision2)
        request.detectsHands = true
        var evaluations: [DatasetFrameEvaluation] = []
        evaluations.reserveCapacity(records.count)

        for (index, record) in records.enumerated() {
            let storage = try EvaluationRG10Storage(
                filePath: record.path,
                expectedByteCount: 4_147_200,
                alignment: 4_096
            )
            let pixelBuffer = EvaluationRG10PixelBuffer(
                layout: layout,
                storage: storage
            )
            let presentationTime = CMTime(value: Int64(index), timescale: 30)
            let sample = try CMImageSampleBuffer(
                imageBuffer: pixelBuffer,
                formatDescription: CMImmutableVideoFormatDescription(
                    dimensions: dimensions,
                    pixelFormat: pixelFormat
                ),
                timing: CMSampleTimingInfo(
                    duration: CMTime(value: 1, timescale: 30),
                    presentationTimeStamp: presentationTime,
                    decodeTimeStamp: .invalid
                )
            )
            let input = try VisionImageInput(
                sampleBuffer: sample,
                orientation: .up,
                frameID: VisionFrameID(
                    source: "dataset:\(record.scene)",
                    sequence: UInt64(index + 1)
                ),
                clockDomain: clock
            )
            let executionID = VisionExecutionID(
                sessionID: session.descriptor.id,
                sequence: UInt64(index + 1)
            )
            let start = try monotonicNanoseconds()
            let observations = try await session.bodyPoseObservations(
                for: request,
                input: input,
                executionID: executionID
            )
            let end = try monotonicNanoseconds()
            guard input.isReleased else {
                throw DatasetEvaluationError.inputNotReleased(record.id)
            }
            let poses = observations.map(DatasetPoseEvaluation.init)
            let poseAccuracy: PoseAccuracyEvaluation?
            if record.groundTruthJoints.isEmpty {
                poseAccuracy = nil
            } else {
                poseAccuracy = try PoseAccuracyEvaluator.evaluate(
                    groundTruth: record.groundTruthJoints,
                    predictedPoses: poses.map { pose in
                        EvaluationPredictedPose(
                            joints: pose.body.map { joint in
                                EvaluationPredictedJoint(
                                    name: joint.name,
                                    x: joint.x,
                                    y: joint.y
                                )
                            }
                        )
                    }
                )
            }
            evaluations.append(
                DatasetFrameEvaluation(
                    id: record.id,
                    scene: record.scene,
                    groundTruthPersonCount: record.groundTruthPersonCount,
                    predictedPersonCount: observations.count,
                    inferenceMilliseconds: Double(end - start) / 1_000_000.0,
                    poses: poses,
                    poseAccuracy: poseAccuracy
                )
            )
        }
        return evaluations
    }

    private func evaluateTemporal(
        records: [EvaluationManifestRecord]
    ) async -> TemporalEvaluationOutcome {
        let trackingRequest: TrackHumanBodyPoseRequest
        let recognitionSession: RecognitionSession
        do {
            trackingRequest = try TrackHumanBodyPoseRequest(
                trackingSessionID: VisionTrackingSessionID(
                    high: 0x4C_55_4D_45,
                    low: 0x4D_34
                ),
                frameAnalysisSpacing: .zero,
                maximumMissedAnalysisCount: 2,
                maximumTrackCount: 4,
                maximumNormalizedJointDistance: 0.15
            )
            trackingRequest.detectsHands = true
            recognitionSession = RecognitionSession(
                id: RecognitionSessionID(
                    high: 0x4C_55_4D_45,
                    low: 0x4D_34
                ),
                budget: try .lumeProofV1()
            )
        } catch {
            return .failure(String(describing: error))
        }

        var frames: [TemporalFrameEvaluation] = []
        frames.reserveCapacity(records.count)
        var maximumRetainedSamples = 0
        var maximumRetainedFeatureBytes = 0
        var operationFailure: String?

        do {
            let dimensions = try CVPixelDimensions(width: 1_920, height: 1_080)
            let pixelFormat = CVPixelFormatType(
                rawValue: RG10PreprocessingConfiguration.pixelFormatRawValue
            )
            let layout = try CVPackedPixelBufferLayout(
                dimensions: dimensions,
                pixelFormat: pixelFormat,
                bytesPerPixel: 2,
                bytesPerRow: 3_840
            )
            let formatDescription = CMImmutableVideoFormatDescription(
                dimensions: dimensions,
                pixelFormat: pixelFormat
            )
            let clock = VisionClockDomain(
                id: "ipn-temporal-evaluation",
                epoch: 1,
                kind: .deviceMonotonic
            )

            for (index, record) in records.enumerated() {
                let storage = try EvaluationRG10Storage(
                    filePath: record.path,
                    expectedByteCount: 4_147_200,
                    alignment: 4_096
                )
                let pixelBuffer = EvaluationRG10PixelBuffer(
                    layout: layout,
                    storage: storage
                )
                let presentationTime = CMTime(
                    value: Int64(index),
                    timescale: 30
                )
                let sample = try CMImageSampleBuffer(
                    imageBuffer: pixelBuffer,
                    formatDescription: formatDescription,
                    timing: CMSampleTimingInfo(
                        duration: CMTime(value: 1, timescale: 30),
                        presentationTimeStamp: presentationTime,
                        decodeTimeStamp: .invalid
                    )
                )
                let input = try VisionImageInput(
                    sampleBuffer: sample,
                    orientation: .up,
                    frameID: VisionFrameID(
                        source: "ipn-temporal",
                        sequence: UInt64(index + 1)
                    ),
                    clockDomain: clock
                )

                let poseStart = try monotonicNanoseconds()
                let trackingUpdate = try await trackingRequest.perform(on: input)
                let poseEnd = try monotonicNanoseconds()
                guard input.isReleased else {
                    throw DatasetEvaluationError.inputNotReleased(record.id)
                }

                let recognitionStart = poseEnd
                let recognitionUpdate = try await recognitionSession.process(
                    trackingUpdate
                )
                let recognitionEnd = try monotonicNanoseconds()
                let diagnostics = try await recognitionSession.diagnostics()
                maximumRetainedSamples = max(
                    maximumRetainedSamples,
                    diagnostics.retainedSampleCount
                )
                maximumRetainedFeatureBytes = max(
                    maximumRetainedFeatureBytes,
                    diagnostics.estimatedRetainedFeatureBytes
                )
                frames.append(
                    TemporalFrameEvaluation(
                        id: record.id,
                        groundTruthLabel: record.scene,
                        predictedPersonCount: trackingUpdate.observations.count,
                        wasAnalyzed: trackingUpdate.wasAnalyzed,
                        wrists: trackingUpdate.observations.flatMap(
                            TemporalWristEvaluation.values
                        ),
                        associations: recognitionUpdate.associations.map(
                            TemporalActorAssociation.init
                        ),
                        decisions: recognitionUpdate.decisions.map(
                            TemporalDecisionEvaluation.init
                        ),
                        poseMilliseconds: Double(poseEnd - poseStart)
                            / 1_000_000,
                        recognitionMilliseconds: Double(
                            recognitionEnd - recognitionStart
                        ) / 1_000_000,
                        endToEndMilliseconds: Double(
                            recognitionEnd - poseStart
                        ) / 1_000_000
                    )
                )
            }
        } catch {
            operationFailure = String(describing: error)
        }

        var cleanupFailure: String?
        do {
            try await trackingRequest.shutdown()
        } catch {
            cleanupFailure = String(describing: error)
        }
        await recognitionSession.shutdown()

        switch (operationFailure, cleanupFailure) {
        case (.some(let operation), .some(let cleanup)):
            return .failure(
                "operation=\(operation); cleanup=\(cleanup)"
            )
        case (.some(let operation), nil):
            return .failure(operation)
        case (nil, .some(let cleanup)):
            return .failure("cleanup=\(cleanup)")
        case (nil, nil):
            let summary = TemporalEvaluationSummary(
                frames: frames,
                maximumRetainedSamples: maximumRetainedSamples,
                maximumRetainedFeatureBytes: maximumRetainedFeatureBytes
            )
            return .success(
                TemporalEvaluationOutput(
                    status: summary.completedGestureCount > 0
                        ? "passed"
                        : "completedNoGesture",
                    source: "IPN Hand contiguous temporal evaluation",
                    annotationWasRecognitionInput: false,
                    summary: summary,
                    frames: frames
                )
            )
        }
    }

    private func stageArtifact(
        manifest: VisionModelManifest,
        checksum: String,
        stage: VisionModelStageID,
        bindings: [
            (
                semanticID: VisionModelTensorID,
                engineName: String,
                executionElementCapacity: Int?
            )
        ]
    ) throws -> TensorRTStageEngineArtifactDescriptor {
        let artifact = try TensorRTEngineArtifactDescriptor(
            semanticModel: manifest,
            checksum: checksum,
            tensorRTVersion: 101_602,
            cudaRuntimeVersion: 13_020,
            computeCapabilityMajor: 8,
            computeCapabilityMinor: 7,
            precision: .float16
        )
        let outputs = try bindings.map { binding in
            try TensorRTEngineOutputBinding(
                semanticTensorID: binding.semanticID,
                engineTensorName: binding.engineName,
                executionElementCapacity: binding.executionElementCapacity
            )
        }
        return try TensorRTStageEngineArtifactDescriptor(
            artifact: artifact,
            stageID: stage,
            inputTensorName: "input",
            outputBindings: outputs
        )
    }

    private func monotonicNanoseconds() throws -> UInt64 {
        var value = timespec()
        guard clock_gettime(CLOCK_MONOTONIC_RAW, &value) == 0,
              let seconds = UInt64(exactly: value.tv_sec),
              let nanoseconds = UInt64(exactly: value.tv_nsec),
              nanoseconds < 1_000_000_000 else {
            throw DatasetEvaluationError.clock
        }
        let secondComponent = seconds.multipliedReportingOverflow(
            by: 1_000_000_000
        )
        guard !secondComponent.overflow else {
            throw DatasetEvaluationError.clock
        }
        let total = secondComponent.partialValue.addingReportingOverflow(
            nanoseconds
        )
        guard !total.overflow else {
            throw DatasetEvaluationError.clock
        }
        return total.partialValue
    }
}

private struct EvaluationManifestRecord: Sendable {
    let id: String
    let scene: String
    let groundTruthPersonCount: Int
    let path: String
    let groundTruthJoints: [EvaluationGroundTruthJoint]
}

private struct EvaluationManifestData: Sendable {
    let source: String?
    let limitations: [String]
    let records: [EvaluationManifestRecord]
}

private enum EvaluationManifest {
    static func load(at path: String) throws -> EvaluationManifestData {
        let contents = try textFile(at: path)
        var source: String?
        var limitations: [String] = []
        var records: [EvaluationManifestRecord] = []
        for (lineIndex, line) in contents.split(separator: "\n").enumerated() {
            let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
            if fields.first == "#source" {
                guard fields.count == 2, !fields[1].isEmpty, source == nil else {
                    throw DatasetEvaluationError.invalidManifestLine(lineIndex + 1)
                }
                source = String(fields[1])
                continue
            }
            if fields.first == "#limitation" {
                guard fields.count == 2, !fields[1].isEmpty else {
                    throw DatasetEvaluationError.invalidManifestLine(lineIndex + 1)
                }
                limitations.append(String(fields[1]))
                continue
            }
            guard (fields.count == 4 || fields.count == 5),
                  let count = Int(fields[2]),
                  count >= 0,
                  !fields[0].isEmpty,
                  !fields[1].isEmpty,
                  !fields[3].isEmpty else {
                throw DatasetEvaluationError.invalidManifestLine(lineIndex + 1)
            }
            let groundTruthJoints: [EvaluationGroundTruthJoint]
            if fields.count == 5 {
                do {
                    groundTruthJoints = try EvaluationGroundTruthParser.parse(fields[4])
                } catch {
                    throw DatasetEvaluationError.invalidGroundTruth(
                        lineIndex + 1,
                        String(describing: error)
                    )
                }
            } else {
                groundTruthJoints = []
            }
            records.append(
                EvaluationManifestRecord(
                    id: String(fields[0]),
                    scene: String(fields[1]),
                    groundTruthPersonCount: count,
                    path: String(fields[3]),
                    groundTruthJoints: groundTruthJoints
                )
            )
        }
        return EvaluationManifestData(
            source: source,
            limitations: limitations,
            records: records
        )
    }

    private static func textFile(at path: String) throws -> String {
        let descriptor = path.withCString { open($0, O_RDONLY) }
        guard descriptor >= 0 else {
            throw DatasetEvaluationError.manifestRead(path)
        }
        defer { _ = close(descriptor) }
        var status = stat()
        guard fstat(descriptor, &status) == 0,
              let byteCount = Int(exactly: status.st_size),
              byteCount > 0 else {
            throw DatasetEvaluationError.manifestRead(path)
        }
        var bytes = [UInt8](repeating: 0, count: byteCount)
        var offset = 0
        while offset < byteCount {
            let readCount = bytes.withUnsafeMutableBytes { buffer -> Int in
                guard let baseAddress = buffer.baseAddress else { return -1 }
                return read(
                    descriptor,
                    baseAddress.advanced(by: offset),
                    byteCount - offset
                )
            }
            guard readCount > 0 else {
                throw DatasetEvaluationError.manifestRead(path)
            }
            offset += readCount
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}

private final class EvaluationRG10Storage: Sendable {
    // The owner allocates and deallocates this storage exactly once. The stored
    // integer address is protected by Mutex on every target. Pointer borrows
    // are scoped to closures, validate the full byte extent, and never escape.
    // The allocation is byte-bound, 4096-byte aligned, fully initialized by a
    // checked read loop, immutable after initialization, and retained by the
    // CVPixelBuffer owner through provider input consumption.
    private struct State: Sendable {
        var address: UInt?
    }

    let byteCount: Int
    private let state: Mutex<State>

    init(
        filePath: String,
        expectedByteCount: Int,
        alignment: Int
    ) throws {
        guard expectedByteCount > 0, alignment > 0 else {
            throw DatasetEvaluationError.allocation
        }
        let pointer = UnsafeMutableRawPointer.allocate(
            byteCount: expectedByteCount,
            alignment: alignment
        )
        byteCount = expectedByteCount
        state = Mutex(State(address: UInt(bitPattern: pointer)))
        do {
            try initialize(filePath: filePath, pointer: pointer)
        } catch {
            let address = state.withLock { state -> UInt? in
                let address = state.address
                state.address = nil
                return address
            }
            if let address,
               let allocated = UnsafeMutableRawPointer(bitPattern: address) {
                allocated.deallocate()
            }
            throw error
        }
    }

    deinit {
        let address = state.withLock { state -> UInt? in
            let address = state.address
            state.address = nil
            return address
        }
        if let address,
           let pointer = UnsafeMutableRawPointer(bitPattern: address) {
            pointer.deallocate()
        }
    }

    func withReadBytes(
        _ body: (borrowing Span<UInt8>) -> Void
    ) throws(DatasetEvaluationError) {
        let address = state.withLock { $0.address }
        guard let address,
              let pointer = UnsafeRawPointer(bitPattern: address) else {
            throw .storageReleased
        }
        body(
            Span(
                _unsafeStart: pointer.assumingMemoryBound(to: UInt8.self),
                count: byteCount
            )
        )
    }

    private func initialize(
        filePath: String,
        pointer: UnsafeMutableRawPointer
    ) throws {
        let descriptor = filePath.withCString { open($0, O_RDONLY) }
        guard descriptor >= 0 else {
            throw DatasetEvaluationError.fixtureRead(filePath)
        }
        defer { _ = close(descriptor) }
        var status = stat()
        guard fstat(descriptor, &status) == 0,
              status.st_size == byteCount else {
            throw DatasetEvaluationError.fixtureSize(filePath)
        }
        var offset = 0
        while offset < byteCount {
            let readCount = read(
                descriptor,
                pointer.advanced(by: offset),
                byteCount - offset
            )
            guard readCount > 0 else {
                throw DatasetEvaluationError.fixtureRead(filePath)
            }
            offset += readCount
        }
    }
}

private final class EvaluationRG10PixelBuffer: CVPixelBuffer {
    let layout: CVPackedPixelBufferLayout
    let accessCapabilities: CVPixelBufferAccessCapabilities = [.read]
    let attachments = CVBufferAttachments()

    private let storage: EvaluationRG10Storage

    var dimensions: CVPixelDimensions { layout.dimensions }
    var pixelFormat: CVPixelFormatType { layout.pixelFormat }
    var bytesPerRow: Int { layout.bytesPerRow }
    var byteCount: Int { storage.byteCount }

    init(layout: CVPackedPixelBufferLayout, storage: EvaluationRG10Storage) {
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

private struct DatasetJointEvaluation: Sendable {
    let name: String
    let x: Float
    let y: Float
    let confidence: Float

    init(name: String, joint: Joint) {
        self.name = name
        x = joint.location.x
        y = joint.location.y
        confidence = joint.confidence
    }

    var json: String {
        "{\"name\":" + jsonString(name)
            + ",\"x\":" + String(x)
            + ",\"y\":" + String(y)
            + ",\"confidence\":" + String(confidence)
            + "}"
    }
}

private struct DatasetPoseEvaluation: Sendable {
    let confidence: Float
    let body: [DatasetJointEvaluation]
    let leftHand: [DatasetJointEvaluation]
    let rightHand: [DatasetJointEvaluation]

    init(_ observation: HumanBodyPoseObservation) {
        confidence = observation.confidence
        body = observation.allJoints()
            .map { DatasetJointEvaluation(name: $0.key.rawValue, joint: $0.value) }
            .sorted { $0.name < $1.name }
        leftHand = Self.handJoints(observation.leftHand)
        rightHand = Self.handJoints(observation.rightHand)
    }

    private static func handJoints(
        _ observation: HumanHandPoseObservation?
    ) -> [DatasetJointEvaluation] {
        guard let observation else { return [] }
        return observation.allJoints()
            .map { DatasetJointEvaluation(name: $0.key.rawValue, joint: $0.value) }
            .sorted { $0.name < $1.name }
    }

    var json: String {
        "{\"confidence\":" + String(confidence)
            + ",\"body\":" + jsonArray(body.map(\.json))
            + ",\"leftHand\":" + jsonArray(leftHand.map(\.json))
            + ",\"rightHand\":" + jsonArray(rightHand.map(\.json))
            + "}"
    }
}

private struct DatasetFrameEvaluation: Sendable {
    let id: String
    let scene: String
    let groundTruthPersonCount: Int
    let predictedPersonCount: Int
    let inferenceMilliseconds: Double
    let poses: [DatasetPoseEvaluation]
    let poseAccuracy: PoseAccuracyEvaluation?

    var json: String {
        "{\"id\":" + jsonString(id)
            + ",\"scene\":" + jsonString(scene)
            + ",\"groundTruthPersonCount\":"
            + String(groundTruthPersonCount)
            + ",\"predictedPersonCount\":"
            + String(predictedPersonCount)
            + ",\"inferenceMilliseconds\":"
            + String(inferenceMilliseconds)
            + ",\"poses\":" + jsonArray(poses.map(\.json))
            + ",\"poseAccuracy\":" + (poseAccuracy?.json ?? "null")
            + "}"
    }
}

private struct DatasetEvaluationSummary: Sendable {
    let frameCount: Int
    let negativeFrameCount: Int
    let negativeFalsePositiveFrameCount: Int
    let framesWithPose: Int
    let exactCapacityAdjustedCountFrames: Int
    let capacityAdjustedCountRecallProxy: Double
    let capacityAdjustedCountPrecisionProxy: Double
    let meanBodyJointsPerPose: Double
    let meanHandJointsPerPose: Double
    let inferenceP50Milliseconds: Double
    let inferenceP95Milliseconds: Double
    let poseAccuracyFrameCount: Int
    let meanJointRecall: Double
    let pckAtFivePercent: Double
    let pckAtTenPercent: Double
    let normalizedMeanError: Double

    init(frames: [DatasetFrameEvaluation]) {
        frameCount = frames.count
        negativeFrameCount = frames.filter {
            $0.groundTruthPersonCount == 0
        }.count
        negativeFalsePositiveFrameCount = frames.filter {
            $0.groundTruthPersonCount == 0 && $0.predictedPersonCount > 0
        }.count
        framesWithPose = frames.filter { !$0.poses.isEmpty }.count
        exactCapacityAdjustedCountFrames = frames.filter {
            min($0.groundTruthPersonCount, 4) == $0.predictedPersonCount
        }.count
        let matchedCount = frames.reduce(0) {
            $0 + min(min($1.groundTruthPersonCount, 4), $1.predictedPersonCount)
        }
        let capacityAdjustedTruth = frames.reduce(0) {
            $0 + min($1.groundTruthPersonCount, 4)
        }
        let predictedCount = frames.reduce(0) {
            $0 + $1.predictedPersonCount
        }
        capacityAdjustedCountRecallProxy = capacityAdjustedTruth > 0
            ? Double(matchedCount) / Double(capacityAdjustedTruth)
            : 0
        capacityAdjustedCountPrecisionProxy = predictedCount > 0
            ? Double(matchedCount) / Double(predictedCount)
            : 0
        let poses = frames.flatMap(\.poses)
        meanBodyJointsPerPose = poses.isEmpty
            ? 0
            : Double(poses.reduce(0) { $0 + $1.body.count }) / Double(poses.count)
        meanHandJointsPerPose = poses.isEmpty
            ? 0
            : Double(
                poses.reduce(0) { $0 + $1.leftHand.count + $1.rightHand.count }
            ) / Double(poses.count)
        let latencies = frames.map(\.inferenceMilliseconds).sorted()
        inferenceP50Milliseconds = Self.percentile(latencies, 0.50)
        inferenceP95Milliseconds = Self.percentile(latencies, 0.95)
        let poseAccuracy = frames.compactMap(\.poseAccuracy)
        poseAccuracyFrameCount = poseAccuracy.count
        let accuracyDenominator = Double(poseAccuracy.count)
        meanJointRecall = poseAccuracy.isEmpty
            ? 0
            : poseAccuracy.reduce(0) { $0 + $1.jointRecall } / accuracyDenominator
        let groundTruthJointCount = poseAccuracy.reduce(0) {
            $0 + $1.groundTruthJointCount
        }
        pckAtFivePercent = groundTruthJointCount > 0
            ? Double(
                poseAccuracy.reduce(0) {
                    $0 + Int(
                        ($1.pckAtFivePercent * Double($1.groundTruthJointCount))
                            .rounded()
                    )
                }
            ) / Double(groundTruthJointCount)
            : 0
        pckAtTenPercent = groundTruthJointCount > 0
            ? Double(
                poseAccuracy.reduce(0) {
                    $0 + Int(
                        ($1.pckAtTenPercent * Double($1.groundTruthJointCount))
                            .rounded()
                    )
                }
            ) / Double(groundTruthJointCount)
            : 0
        normalizedMeanError = groundTruthJointCount > 0
            ? poseAccuracy.reduce(0) {
                $0 + $1.normalizedMeanError * Double($1.groundTruthJointCount)
            } / Double(groundTruthJointCount)
            : 0
    }

    private static func percentile(_ values: [Double], _ percentile: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let scaledIndex = percentile * Double(values.count - 1)
        return values[Int(scaledIndex.rounded(.up))]
    }

    var json: String {
        "{\"frameCount\":" + String(frameCount)
            + ",\"negativeFrameCount\":" + String(negativeFrameCount)
            + ",\"negativeFalsePositiveFrameCount\":"
            + String(negativeFalsePositiveFrameCount)
            + ",\"framesWithPose\":" + String(framesWithPose)
            + ",\"exactCapacityAdjustedCountFrames\":"
            + String(exactCapacityAdjustedCountFrames)
            + ",\"capacityAdjustedCountRecallProxy\":"
            + String(capacityAdjustedCountRecallProxy)
            + ",\"capacityAdjustedCountPrecisionProxy\":"
            + String(capacityAdjustedCountPrecisionProxy)
            + ",\"meanBodyJointsPerPose\":"
            + String(meanBodyJointsPerPose)
            + ",\"meanHandJointsPerPose\":"
            + String(meanHandJointsPerPose)
            + ",\"inferenceP50Milliseconds\":"
            + String(inferenceP50Milliseconds)
            + ",\"inferenceP95Milliseconds\":"
            + String(inferenceP95Milliseconds)
            + ",\"poseAccuracyFrameCount\":" + String(poseAccuracyFrameCount)
            + ",\"meanJointRecall\":" + String(meanJointRecall)
            + ",\"pckAtFivePercent\":" + String(pckAtFivePercent)
            + ",\"pckAtTenPercent\":" + String(pckAtTenPercent)
            + ",\"normalizedMeanError\":" + String(normalizedMeanError)
            + "}"
    }
}

private struct DatasetEvaluationOutput: Sendable {
    let status: String
    let source: String
    let limitations: [String]
    let summary: DatasetEvaluationSummary
    let frames: [DatasetFrameEvaluation]

    var json: String {
        "{\"status\":" + jsonString(status)
            + ",\"source\":" + jsonString(source)
            + ",\"limitations\":"
            + jsonArray(limitations.map(jsonString))
            + ",\"summary\":" + summary.json
            + ",\"frames\":" + jsonArray(frames.map(\.json))
            + "}"
    }
}

private enum TemporalEvaluationOutcome: Sendable {
    case success(TemporalEvaluationOutput)
    case failure(String)
}

private struct TemporalActorAssociation: Sendable {
    let actorSequence: UInt64
    let trackEpoch: UInt64
    let trackSequence: UInt64
    let kind: String
    let confidence: Float

    init(_ association: ActorAssociation) {
        actorSequence = association.actorID.sequence
        trackEpoch = association.trackID.epoch
        trackSequence = association.trackID.sequence
        switch association.kind {
        case .created:
            kind = "created"
        case .continued:
            kind = "continued"
        case .reassociated:
            kind = "reassociated"
        }
        confidence = association.confidence
    }

    var json: String {
        "{\"actorSequence\":" + String(actorSequence)
            + ",\"trackEpoch\":" + String(trackEpoch)
            + ",\"trackSequence\":" + String(trackSequence)
            + ",\"kind\":" + jsonString(kind)
            + ",\"confidence\":" + String(confidence) + "}"
    }
}

private struct TemporalGestureEvaluation: Sendable {
    let actorSequence: UInt64
    let correlationSequence: UInt64
    let identifier: String
    let phase: String
    let direction: String
    let displacement: Float
    let velocity: Float
    let confidence: Float

    init(_ observation: GestureObservation) {
        actorSequence = observation.actorID.sequence
        correlationSequence = observation.correlationID.sequence
        identifier = observation.identifier.rawValue
        switch observation.phase {
        case .began:
            phase = "began"
        case .changed:
            phase = "changed"
        case .ended:
            phase = "ended"
        case .cancelled:
            phase = "cancelled"
        }
        switch observation.parameters {
        case .horizontalSwipe(let horizontalDirection, let magnitude, let speed):
            direction = horizontalDirection == .left ? "left" : "right"
            displacement = magnitude
            velocity = speed
        }
        confidence = observation.confidence
    }

    var json: String {
        "{\"actorSequence\":" + String(actorSequence)
            + ",\"correlationSequence\":" + String(correlationSequence)
            + ",\"identifier\":" + jsonString(identifier)
            + ",\"phase\":" + jsonString(phase)
            + ",\"direction\":" + jsonString(direction)
            + ",\"displacement\":" + String(displacement)
            + ",\"velocity\":" + String(velocity)
            + ",\"confidence\":" + String(confidence) + "}"
    }
}

private enum TemporalDecisionEvaluation: Sendable {
    case noMatch(actorSequence: UInt64)
    case recognized(TemporalGestureEvaluation)
    case ambiguous(actorSequence: UInt64, candidateCount: Int)

    init(_ decision: RecognitionDecision) {
        switch decision {
        case .noMatch(let actorID):
            self = .noMatch(actorSequence: actorID.sequence)
        case .recognized(let observation):
            self = .recognized(TemporalGestureEvaluation(observation))
        case .ambiguous(let actorID, let candidates):
            self = .ambiguous(
                actorSequence: actorID.sequence,
                candidateCount: candidates.count
            )
        }
    }

    var gesture: TemporalGestureEvaluation? {
        guard case .recognized(let gesture) = self else { return nil }
        return gesture
    }

    var isNoMatch: Bool {
        guard case .noMatch = self else { return false }
        return true
    }

    var json: String {
        switch self {
        case .noMatch(let actorSequence):
            return "{\"kind\":\"noMatch\",\"actorSequence\":"
                + String(actorSequence) + "}"
        case .recognized(let gesture):
            return "{\"kind\":\"recognized\",\"gesture\":"
                + gesture.json + "}"
        case .ambiguous(let actorSequence, let candidateCount):
            return "{\"kind\":\"ambiguous\",\"actorSequence\":"
                + String(actorSequence)
                + ",\"candidateCount\":" + String(candidateCount) + "}"
        }
    }
}

private struct TemporalWristEvaluation: Sendable {
    let trackEpoch: UInt64
    let trackSequence: UInt64
    let side: String
    let x: Float
    let y: Float
    let confidence: Float

    static func values(
        for observation: TrackedHumanBodyPoseObservation
    ) -> [Self] {
        let track = observation.track.id
        var wrists: [Self] = []
        wrists.reserveCapacity(2)
        if let joint = observation.pose.joint(for: .leftWrist) {
            wrists.append(
                Self(
                    trackEpoch: track.epoch,
                    trackSequence: track.sequence,
                    side: "left",
                    x: joint.location.x,
                    y: joint.location.y,
                    confidence: joint.confidence
                )
            )
        }
        if let joint = observation.pose.joint(for: .rightWrist) {
            wrists.append(
                Self(
                    trackEpoch: track.epoch,
                    trackSequence: track.sequence,
                    side: "right",
                    x: joint.location.x,
                    y: joint.location.y,
                    confidence: joint.confidence
                )
            )
        }
        return wrists
    }

    var json: String {
        "{\"trackEpoch\":" + String(trackEpoch)
            + ",\"trackSequence\":" + String(trackSequence)
            + ",\"side\":" + jsonString(side)
            + ",\"x\":" + String(x)
            + ",\"y\":" + String(y)
            + ",\"confidence\":" + String(confidence) + "}"
    }
}

private struct TemporalFrameEvaluation: Sendable {
    let id: String
    let groundTruthLabel: String
    let predictedPersonCount: Int
    let wasAnalyzed: Bool
    let wrists: [TemporalWristEvaluation]
    let associations: [TemporalActorAssociation]
    let decisions: [TemporalDecisionEvaluation]
    let poseMilliseconds: Double
    let recognitionMilliseconds: Double
    let endToEndMilliseconds: Double

    var json: String {
        "{\"id\":" + jsonString(id)
            + ",\"groundTruthLabel\":" + jsonString(groundTruthLabel)
            + ",\"predictedPersonCount\":" + String(predictedPersonCount)
            + ",\"wasAnalyzed\":" + String(wasAnalyzed)
            + ",\"wrists\":" + jsonArray(wrists.map(\.json))
            + ",\"associations\":"
            + jsonArray(associations.map(\.json))
            + ",\"decisions\":" + jsonArray(decisions.map(\.json))
            + ",\"poseMilliseconds\":" + String(poseMilliseconds)
            + ",\"recognitionMilliseconds\":"
            + String(recognitionMilliseconds)
            + ",\"endToEndMilliseconds\":"
            + String(endToEndMilliseconds) + "}"
    }
}

private struct TemporalEvaluationSummary: Sendable {
    let frameCount: Int
    let framesWithPose: Int
    let uniqueActorCount: Int
    let uniqueTrackCount: Int
    let noMatchDecisionCount: Int
    let beganGestureCount: Int
    let changedGestureCount: Int
    let completedGestureCount: Int
    let cancelledGestureCount: Int
    let poseP50Milliseconds: Double
    let poseP95Milliseconds: Double
    let recognitionP50Milliseconds: Double
    let recognitionP95Milliseconds: Double
    let endToEndP50Milliseconds: Double
    let endToEndP95Milliseconds: Double
    let maximumRetainedSamples: Int
    let maximumRetainedFeatureBytes: Int

    init(
        frames: [TemporalFrameEvaluation],
        maximumRetainedSamples: Int,
        maximumRetainedFeatureBytes: Int
    ) {
        frameCount = frames.count
        framesWithPose = frames.filter { $0.predictedPersonCount > 0 }.count
        uniqueActorCount = Set(
            frames.flatMap(\.associations).map(\.actorSequence)
        ).count
        uniqueTrackCount = Set(
            frames.flatMap(\.associations).map {
                "\($0.trackEpoch):\($0.trackSequence)"
            }
        ).count
        let decisions = frames.flatMap(\.decisions)
        noMatchDecisionCount = decisions.filter(\.isNoMatch).count
        let gestures = decisions.compactMap(\.gesture)
        beganGestureCount = gestures.filter { $0.phase == "began" }.count
        changedGestureCount = gestures.filter { $0.phase == "changed" }.count
        completedGestureCount = gestures.filter { $0.phase == "ended" }.count
        cancelledGestureCount = gestures.filter { $0.phase == "cancelled" }.count
        poseP50Milliseconds = Self.percentile(
            frames.map(\.poseMilliseconds),
            0.50
        )
        poseP95Milliseconds = Self.percentile(
            frames.map(\.poseMilliseconds),
            0.95
        )
        recognitionP50Milliseconds = Self.percentile(
            frames.map(\.recognitionMilliseconds),
            0.50
        )
        recognitionP95Milliseconds = Self.percentile(
            frames.map(\.recognitionMilliseconds),
            0.95
        )
        endToEndP50Milliseconds = Self.percentile(
            frames.map(\.endToEndMilliseconds),
            0.50
        )
        endToEndP95Milliseconds = Self.percentile(
            frames.map(\.endToEndMilliseconds),
            0.95
        )
        self.maximumRetainedSamples = maximumRetainedSamples
        self.maximumRetainedFeatureBytes = maximumRetainedFeatureBytes
    }

    private static func percentile(
        _ values: [Double],
        _ percentile: Double
    ) -> Double {
        let ordered = values.sorted()
        guard !ordered.isEmpty else { return 0 }
        let scaledIndex = percentile * Double(ordered.count - 1)
        return ordered[Int(scaledIndex.rounded(.up))]
    }

    var json: String {
        "{\"frameCount\":" + String(frameCount)
            + ",\"framesWithPose\":" + String(framesWithPose)
            + ",\"uniqueActorCount\":" + String(uniqueActorCount)
            + ",\"uniqueTrackCount\":" + String(uniqueTrackCount)
            + ",\"noMatchDecisionCount\":" + String(noMatchDecisionCount)
            + ",\"beganGestureCount\":" + String(beganGestureCount)
            + ",\"changedGestureCount\":" + String(changedGestureCount)
            + ",\"completedGestureCount\":" + String(completedGestureCount)
            + ",\"cancelledGestureCount\":" + String(cancelledGestureCount)
            + ",\"poseP50Milliseconds\":" + String(poseP50Milliseconds)
            + ",\"poseP95Milliseconds\":" + String(poseP95Milliseconds)
            + ",\"recognitionP50Milliseconds\":"
            + String(recognitionP50Milliseconds)
            + ",\"recognitionP95Milliseconds\":"
            + String(recognitionP95Milliseconds)
            + ",\"endToEndP50Milliseconds\":"
            + String(endToEndP50Milliseconds)
            + ",\"endToEndP95Milliseconds\":"
            + String(endToEndP95Milliseconds)
            + ",\"maximumRetainedSamples\":"
            + String(maximumRetainedSamples)
            + ",\"maximumRetainedFeatureBytes\":"
            + String(maximumRetainedFeatureBytes) + "}"
    }
}

private struct TemporalEvaluationOutput: Sendable {
    let status: String
    let source: String
    let annotationWasRecognitionInput: Bool
    let summary: TemporalEvaluationSummary
    let frames: [TemporalFrameEvaluation]

    var json: String {
        "{\"status\":" + jsonString(status)
            + ",\"source\":" + jsonString(source)
            + ",\"annotationWasRecognitionInput\":"
            + String(annotationWasRecognitionInput)
            + ",\"summary\":" + summary.json
            + ",\"frames\":" + jsonArray(frames.map(\.json)) + "}"
    }
}

private struct DatasetEvaluationFailure: Sendable {
    let status: String
    let reason: String

    var json: String {
        "{\"status\":" + jsonString(status)
            + ",\"reason\":" + jsonString(reason) + "}"
    }
}

private func jsonArray(_ values: [String]) -> String {
    "[" + values.joined(separator: ",") + "]"
}

private func jsonString(_ value: String) -> String {
    var result = "\""
    for scalar in value.unicodeScalars {
        switch scalar.value {
        case 0x22:
            result += "\\\""
        case 0x5C:
            result += "\\\\"
        case 0x08:
            result += "\\b"
        case 0x0C:
            result += "\\f"
        case 0x0A:
            result += "\\n"
        case 0x0D:
            result += "\\r"
        case 0x09:
            result += "\\t"
        case 0x00...0x1F:
            let digits = Array("0123456789abcdef".utf8)
            let high = Int((scalar.value >> 4) & 0xF)
            let low = Int(scalar.value & 0xF)
            result += "\\u00"
            result.append(Character(UnicodeScalar(digits[high])))
            result.append(Character(UnicodeScalar(digits[low])))
        default:
            result.unicodeScalars.append(scalar)
        }
    }
    result += "\""
    return result
}

private enum DatasetEvaluationError: Error {
    case invalidArguments
    case emptyManifest
    case missingDetectorStage
    case manifestRead(String)
    case invalidManifestLine(Int)
    case invalidGroundTruth(Int, String)
    case fixtureRead(String)
    case fixtureSize(String)
    case allocation
    case storageReleased
    case inputNotReleased(String)
    case clock
    case shutdown(String)
    case operationAndShutdown(operation: String, shutdown: String)
    case visionContext(String)
    case temporalEvaluation(String)
}
#endif
