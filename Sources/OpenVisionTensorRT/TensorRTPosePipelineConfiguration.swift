import OpenVision

public struct TensorRTPosePipelineConfiguration:
    Sendable,
    Hashable
{
    public let source: RG10PreprocessingConfiguration
    public let detectorInputWidth: Int
    public let detectorInputHeight: Int
    public let poseInputWidth: Int
    public let poseInputHeight: Int
    public let maximumRegionCount: Int
    public let jointCount: Int
    public let minimumDetectionConfidence: Float
    public let maximumDetectionOverlap: Float
    public let regionScale: Float
    public let poseNormalization:
        VisionModelInputDescriptor.Normalization

    public init(
        source: RG10PreprocessingConfiguration,
        detectorInputWidth: Int = 320,
        detectorInputHeight: Int = 320,
        poseInputWidth: Int = 192,
        poseInputHeight: Int = 256,
        maximumRegionCount: Int = 4,
        jointCount: Int = 133,
        minimumDetectionConfidence: Float = 0.3,
        maximumDetectionOverlap: Float = 0.5,
        regionScale: Float = 1.25,
        poseNormalization:
            VisionModelInputDescriptor.Normalization
    ) throws(TensorRTPosePipelineError) {
        guard
            detectorInputWidth > 0,
            detectorInputHeight > 0,
            poseInputWidth > 0,
            poseInputHeight > 0,
            detectorInputWidth <= Int(UInt32.max),
            detectorInputHeight <= Int(UInt32.max),
            poseInputWidth <= Int(UInt32.max),
            poseInputHeight <= Int(UInt32.max)
        else {
            throw .invalidConfiguration("inputDimensions")
        }
        guard (1 ... 4).contains(maximumRegionCount) else {
            throw .invalidConfiguration("maximumRegionCount")
        }
        guard (1 ... 256).contains(jointCount) else {
            throw .invalidConfiguration("jointCount")
        }
        guard
            minimumDetectionConfidence.isFinite,
            (0 ... 1).contains(minimumDetectionConfidence)
        else {
            throw .invalidConfiguration(
                "minimumDetectionConfidence"
            )
        }
        guard
            maximumDetectionOverlap.isFinite,
            maximumDetectionOverlap > 0,
            maximumDetectionOverlap <= 1
        else {
            throw .invalidConfiguration(
                "maximumDetectionOverlap"
            )
        }
        guard regionScale.isFinite, regionScale > 0 else {
            throw .invalidConfiguration("regionScale")
        }
        let normalizationValues = [
            poseNormalization.scale.red,
            poseNormalization.scale.green,
            poseNormalization.scale.blue,
            poseNormalization.bias.red,
            poseNormalization.bias.green,
            poseNormalization.bias.blue
        ]
        guard normalizationValues.allSatisfy({
            $0.isFinite
        }) else {
            throw .invalidConfiguration("poseNormalization")
        }

        self.source = source
        self.detectorInputWidth = detectorInputWidth
        self.detectorInputHeight = detectorInputHeight
        self.poseInputWidth = poseInputWidth
        self.poseInputHeight = poseInputHeight
        self.maximumRegionCount = maximumRegionCount
        self.jointCount = jointCount
        self.minimumDetectionConfidence =
            minimumDetectionConfidence
        self.maximumDetectionOverlap = maximumDetectionOverlap
        self.regionScale = regionScale
        self.poseNormalization = poseNormalization
    }
}
