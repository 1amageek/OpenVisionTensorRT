import OpenVision

public struct OpenVisionTensorRTProviderConfiguration: Sendable {
    /// RTMDet-nano candidate count before the exported graph's final top-k.
    ///
    /// The 320x320 detector has 40x40, 20x20, and 10x10 feature locations.
    /// TensorRT may request callback capacity for all candidates even though
    /// the semantic output accepts at most 100 final detections.
    public static let detectorExecutionCandidateCount = 2_100

    public let model: VisionModelManifest
    public let detectorPlanPath: String
    public let detectorArtifact:
        TensorRTStageEngineArtifactDescriptor
    public let posePlanPath: String
    public let poseArtifact:
        TensorRTStageEngineArtifactDescriptor
    public let detectorPreprocessing:
        RG10PreprocessingConfiguration
    public let posePipeline:
        TensorRTPosePipelineConfiguration

    public init(
        model: VisionModelManifest,
        detectorPlanPath: String,
        detectorArtifact:
            TensorRTStageEngineArtifactDescriptor,
        posePlanPath: String,
        poseArtifact:
            TensorRTStageEngineArtifactDescriptor,
        detectorPreprocessing:
            RG10PreprocessingConfiguration
    ) throws(TensorRTPosePipelineError) {
        guard !detectorPlanPath.isEmpty else {
            throw .invalidConfiguration("detectorPlanPath")
        }
        guard !posePlanPath.isEmpty else {
            throw .invalidConfiguration("posePlanPath")
        }
        guard
            detectorArtifact.artifact.semanticModel == model,
            poseArtifact.artifact.semanticModel == model,
            detectorArtifact.stageID ==
                RTMDetDWPoseBodyPoseManifest.personDetectionStage,
            poseArtifact.stageID ==
                RTMDetDWPoseBodyPoseManifest.wholeBodyPoseStage,
            let detectorStage = model.stage(
                identifiedBy:
                    RTMDetDWPoseBodyPoseManifest.personDetectionStage
            ),
            let poseStage = model.stage(
                identifiedBy:
                    RTMDetDWPoseBodyPoseManifest.wholeBodyPoseStage
            ),
            detectorStage.input.width ==
                detectorPreprocessing.outputWidth,
            detectorStage.input.height ==
                detectorPreprocessing.outputHeight,
            detectorStage.input.tensorLayout ==
                detectorPreprocessing.tensorLayout,
            detectorStage.input.channelOrder ==
                detectorPreprocessing.channelOrder,
            detectorStage.input.normalization ==
                detectorPreprocessing.normalization,
            case .regions(
                _,
                _,
                let minimumConfidence,
                let maximumCount,
                let scale
            ) = poseStage.input.source,
            Self.hasDetectorExecutionCapacity(
                detectorArtifact,
                semanticID:
                    RTMDetDWPoseBodyPoseManifest.detectionsTensor,
                minimumElementCount:
                    Self.detectorExecutionCandidateCount * 5
            ),
            Self.hasDetectorExecutionCapacity(
                detectorArtifact,
                semanticID:
                    RTMDetDWPoseBodyPoseManifest.classesTensor,
                minimumElementCount:
                    Self.detectorExecutionCandidateCount
            )
        else {
            throw .invalidConfiguration("semanticModel")
        }

        self.model = model
        self.detectorPlanPath = detectorPlanPath
        self.detectorArtifact = detectorArtifact
        self.posePlanPath = posePlanPath
        self.poseArtifact = poseArtifact
        self.detectorPreprocessing = detectorPreprocessing
        posePipeline = try TensorRTPosePipelineConfiguration(
            source: detectorPreprocessing,
            detectorInputWidth: detectorStage.input.width,
            detectorInputHeight: detectorStage.input.height,
            poseInputWidth: poseStage.input.width,
            poseInputHeight: poseStage.input.height,
            maximumRegionCount: maximumCount,
            jointCount: 133,
            minimumDetectionConfidence: minimumConfidence,
            maximumDetectionOverlap: 0.5,
            regionScale: scale,
            poseNormalization: poseStage.input.normalization
        )
    }

    private static func hasDetectorExecutionCapacity(
        _ artifact: TensorRTStageEngineArtifactDescriptor,
        semanticID: VisionModelTensorID,
        minimumElementCount: Int
    ) -> Bool {
        artifact.outputBindings.contains {
            $0.semanticTensorID == semanticID &&
                ($0.executionElementCapacity ?? 0) >=
                    minimumElementCount
        }
    }
}
