import OpenVision
import OpenVisionTensorRT

@main
enum OpenVisionTensorRTEngineProbe {
    static func main() async throws {
        if CommandLine.arguments.count == 4 {
            switch CommandLine.arguments[1] {
            case "--verify-checksum-failure":
                try await verifyChecksumFailure(
                    path: CommandLine.arguments[2],
                    checksum: CommandLine.arguments[3]
                )
                return
            case "--verify-semantic-mismatch":
                try await verifySemanticMismatch(
                    path: CommandLine.arguments[2],
                    checksum: CommandLine.arguments[3]
                )
                return
            default:
                throw EngineProbeError.invalidArguments
            }
        }
        guard CommandLine.arguments.count == 5 else {
            throw EngineProbeError.invalidArguments
        }
        let detector = try await engine(
            path: CommandLine.arguments[1],
            checksum: CommandLine.arguments[2],
            stageID:
                RTMDetDWPoseBodyPoseManifest.personDetectionStage,
            outputs: [
                (
                    RTMDetDWPoseBodyPoseManifest.detectionsTensor,
                    "dets"
                ),
                (
                    RTMDetDWPoseBodyPoseManifest.classesTensor,
                    "labels"
                )
            ]
        )
        let pose = try await engine(
            path: CommandLine.arguments[3],
            checksum: CommandLine.arguments[4],
            stageID:
                RTMDetDWPoseBodyPoseManifest.wholeBodyPoseStage,
            outputs: [
                (
                    RTMDetDWPoseBodyPoseManifest.simCCXTensor,
                    "simcc_x"
                ),
                (
                    RTMDetDWPoseBodyPoseManifest.simCCYTensor,
                    "simcc_y"
                )
            ]
        )

        print(
            "{\"status\":\"available\","
                + "\"detector\":\(description(detector)),"
                + "\"pose\":\(description(pose))}"
        )
    }

    private static func verifyChecksumFailure(
        path: String,
        checksum: String
    ) async throws {
        do {
            let engine = try TensorRTEngine(
                path: path,
                artifact: try detectorArtifact(
                    checksum: checksum,
                    outputs: detectorOutputs
                )
            )
            try await engine.shutdown()
            throw EngineProbeError.expectedChecksumFailure
        } catch let error as TensorRTEngineError {
            guard
                case .loadingFailed(
                    let status,
                    let report
                ) = error,
                status == .engineChecksumMismatch,
                report.failureStage == .checksum,
                !report.checksumVerified
            else {
                throw error
            }
            print(
                "{\"status\":\"passed\","
                    + "\"failure\":\"engineChecksumMismatch\","
                    + "\"stage\":\"checksum\"}"
            )
        }
    }

    private static func verifySemanticMismatch(
        path: String,
        checksum: String
    ) async throws {
        let swappedOutputs = [
            (
                RTMDetDWPoseBodyPoseManifest.detectionsTensor,
                "labels"
            ),
            (
                RTMDetDWPoseBodyPoseManifest.classesTensor,
                "dets"
            )
        ]
        do {
            let engine = try TensorRTEngine(
                path: path,
                artifact: try detectorArtifact(
                    checksum: checksum,
                    outputs: swappedOutputs
                )
            )
            try await engine.shutdown()
            throw EngineProbeError.expectedSemanticMismatch
        } catch let error as TensorRTEngineError {
            guard
                case .incompatibleArtifact(
                    .elementType(let tensor, _, _)
                ) = error,
                tensor == "labels"
            else {
                throw error
            }
            print(
                "{\"status\":\"passed\","
                    + "\"failure\":\"incompatibleArtifact\","
                    + "\"tensor\":\"labels\","
                    + "\"reason\":\"elementType\"}"
            )
        }
    }

    private static func engine(
        path: String,
        checksum: String,
        stageID: VisionModelStageID,
        outputs: [(VisionModelTensorID, String)]
    ) async throws -> [TensorRTEngineTensorDescriptor] {
        let manifest =
            try RTMDetDWPoseBodyPoseManifest.manifest()
        let base = try TensorRTEngineArtifactDescriptor(
            semanticModel: manifest,
            checksum: checksum,
            tensorRTVersion: 101_602,
            cudaRuntimeVersion: 13_020,
            computeCapabilityMajor: 8,
            computeCapabilityMinor: 7,
            precision: .float16
        )
        var bindings: [TensorRTEngineOutputBinding] = []
        bindings.reserveCapacity(outputs.count)
        for output in outputs {
            bindings.append(
                try TensorRTEngineOutputBinding(
                    semanticTensorID: output.0,
                    engineTensorName: output.1
                )
            )
        }
        let stage = try TensorRTStageEngineArtifactDescriptor(
            artifact: base,
            stageID: stageID,
            inputTensorName: "input",
            outputBindings: bindings
        )
        let engine = try TensorRTEngine(
            path: path,
            artifact: stage
        )
        let tensors = await engine.tensors
        try await engine.shutdown()
        return tensors
    }

    private static var detectorOutputs:
        [(VisionModelTensorID, String)]
    {
        [
            (
                RTMDetDWPoseBodyPoseManifest.detectionsTensor,
                "dets"
            ),
            (
                RTMDetDWPoseBodyPoseManifest.classesTensor,
                "labels"
            )
        ]
    }

    private static func detectorArtifact(
        checksum: String,
        outputs: [(VisionModelTensorID, String)]
    ) throws -> TensorRTStageEngineArtifactDescriptor {
        let manifest =
            try RTMDetDWPoseBodyPoseManifest.manifest()
        let base = try TensorRTEngineArtifactDescriptor(
            semanticModel: manifest,
            checksum: checksum,
            tensorRTVersion: 101_602,
            cudaRuntimeVersion: 13_020,
            computeCapabilityMajor: 8,
            computeCapabilityMinor: 7,
            precision: .float16
        )
        var bindings: [TensorRTEngineOutputBinding] = []
        bindings.reserveCapacity(outputs.count)
        for output in outputs {
            bindings.append(
                try TensorRTEngineOutputBinding(
                    semanticTensorID: output.0,
                    engineTensorName: output.1
                )
            )
        }
        return try TensorRTStageEngineArtifactDescriptor(
            artifact: base,
            stageID:
                RTMDetDWPoseBodyPoseManifest.personDetectionStage,
            inputTensorName: "input",
            outputBindings: bindings
        )
    }

    private static func description(
        _ tensors: [TensorRTEngineTensorDescriptor]
    ) -> String {
        let values = tensors.map { tensor in
            let shape = tensor.declaredShape
                .map(String.init)
                .joined(separator: ",")
            let profile: String
            if let value = tensor.profile {
                let minimum = value.minimum
                    .map(String.init)
                    .joined(separator: ",")
                let optimum = value.optimum
                    .map(String.init)
                    .joined(separator: ",")
                let maximum = value.maximum
                    .map(String.init)
                    .joined(separator: ",")
                profile =
                    ",\"profile\":{\"min\":[\(minimum)],"
                    + "\"opt\":[\(optimum)],"
                    + "\"max\":[\(maximum)]}"
            } else {
                profile = ""
            }
            return "{\"name\":\"\(tensor.name)\","
                + "\"shape\":[\(shape)]\(profile)}"
        }
        return "[\(values.joined(separator: ","))]"
    }
}

private enum EngineProbeError: Error {
    case invalidArguments
    case expectedChecksumFailure
    case expectedSemanticMismatch
}
