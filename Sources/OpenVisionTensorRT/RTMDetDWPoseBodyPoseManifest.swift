import OpenVision

public enum RTMDetDWPoseBodyPoseManifest {
    public static let personDetectionStage = VisionModelStageID(
        rawValue: "person-detection"
    )
    public static let wholeBodyPoseStage = VisionModelStageID(
        rawValue: "whole-body-pose"
    )
    public static let detectionsTensor = VisionModelTensorID(
        rawValue: "detections"
    )
    public static let classesTensor = VisionModelTensorID(
        rawValue: "classes"
    )
    public static let simCCXTensor = VisionModelTensorID(
        rawValue: "simcc-x"
    )
    public static let simCCYTensor = VisionModelTensorID(
        rawValue: "simcc-y"
    )

    public static func manifest()
        throws(VisionModelManifestError) -> VisionModelManifest
    {
        let detector = try detectorStage()
        let pose = try poseStage()
        return try VisionModelManifest(
            id: "openmmlab.rtmdet-nano-dwpose-m.wholebody",
            revision: "05d8511e+c8b76419",
            request: .detectHumanBodyPoseRequest(.revision2),
            stages: [detector, pose],
            output: try VisionModelOutputDescriptor(
                schemaRevision:
                    "coco-wholebody-133-openvision-body-hand-1",
                stage: wholeBodyPoseStage,
                xDistribution: simCCXTensor,
                yDistribution: simCCYTensor,
                jointMappings: try jointMappings(),
                minimumJointConfidence: 0.3,
                coordinateTransform:
                    .modelInputPixelsUpperLeftToNormalizedImageLowerLeft
            ),
            quality: try VisionModelQualityRequirements(
                permittedPrecisions: [.float32, .float16],
                maximumEndToEndLatencyMilliseconds: 1000.0 / 30.0,
                maximumPersonCount: 4
            )
        )
    }

    private static func detectorStage()
        throws(VisionModelManifestError)
        -> VisionModelStageDescriptor
    {
        try VisionModelStageDescriptor(
            id: personDetectionStage,
            operation: .personDetection,
            input: try VisionModelInputDescriptor(
                width: 320,
                height: 320,
                source: .image,
                resizePolicy: .scaleFit,
                transferFunction: .sRGB,
                tensorLayout: .channelsFirst,
                channelOrder: .bgr,
                elementType: .float32,
                letterboxColor: try .init(
                    red: 114.0 / 255.0,
                    green: 114.0 / 255.0,
                    blue: 114.0 / 255.0
                ),
                normalization: try imageNetNormalization()
            ),
            outputs: [
                try VisionModelTensorDescriptor(
                    id: detectionsTensor,
                    elementType: .float32,
                    shape: [
                        .batch(maximum: 1),
                        .fixed(100),
                        .fixed(5)
                    ],
                    meaning: .personDetections(maximumCount: 100)
                ),
                try VisionModelTensorDescriptor(
                    id: classesTensor,
                    elementType: .int32,
                    shape: [
                        .batch(maximum: 1),
                        .fixed(100)
                    ],
                    meaning: .classIndices(maximumCount: 100)
                )
            ],
            provenance: try VisionModelProvenance(
                publisher: "OpenMMLab",
                architecture: "RTMDet-nano",
                sourceLocation:
                    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
                sourceRevision: "05d8511e",
                sourceDigest: try .init(
                    hexadecimal:
                        "05d8511e7b3fabc62e27d2f624179e004ad14ee63a86ca9d9d22c88f3db0eee1"
                ),
                trainingDatasets: ["COCO", "Objects365"],
                citations: ["https://arxiv.org/abs/2212.07784"],
                licenseIdentifier: nil
            )
        )
    }

    private static func poseStage()
        throws(VisionModelManifestError)
        -> VisionModelStageDescriptor
    {
        try VisionModelStageDescriptor(
            id: wholeBodyPoseStage,
            operation: .humanWholeBodyPose,
            input: try VisionModelInputDescriptor(
                width: 192,
                height: 256,
                source: .regions(
                    stage: personDetectionStage,
                    tensor: detectionsTensor,
                    minimumConfidence: 0.3,
                    maximumCount: 4,
                    scale: 1.25
                ),
                resizePolicy: .regionAffine,
                transferFunction: .sRGB,
                tensorLayout: .channelsFirst,
                channelOrder: .rgb,
                elementType: .float32,
                normalization: try imageNetNormalization()
            ),
            outputs: [
                try VisionModelTensorDescriptor(
                    id: simCCXTensor,
                    elementType: .float32,
                    shape: [
                        .batch(maximum: 4),
                        .fixed(133),
                        .fixed(384)
                    ],
                    meaning: .simCC(
                        axis: .x,
                        jointCount: 133,
                        splitRatio: 2
                    )
                ),
                try VisionModelTensorDescriptor(
                    id: simCCYTensor,
                    elementType: .float32,
                    shape: [
                        .batch(maximum: 4),
                        .fixed(133),
                        .fixed(512)
                    ],
                    meaning: .simCC(
                        axis: .y,
                        jointCount: 133,
                        splitRatio: 2
                    )
                )
            ],
            provenance: try VisionModelProvenance(
                publisher: "OpenMMLab",
                architecture: "DWPose-m",
                sourceLocation:
                    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.pth",
                sourceRevision: "c8b76419",
                sourceDigest: try .init(
                    hexadecimal:
                        "c8b7641988ce785c987b8ad156e411f328b54766d353df19aea5b08433ef1aaf"
                ),
                trainingDatasets: ["COCO-WholeBody", "UBody"],
                citations: [
                    "https://arxiv.org/abs/2303.07399",
                    "https://arxiv.org/abs/2307.15880",
                    "https://arxiv.org/abs/2107.03332"
                ],
                licenseIdentifier: nil
            )
        )
    }

    private static func imageNetNormalization()
        throws(VisionModelManifestError)
        -> VisionModelInputDescriptor.Normalization
    {
        try .channelwiseAffine(
            redScale: 255.0 / 58.395,
            greenScale: 255.0 / 57.12,
            blueScale: 255.0 / 57.375,
            redBias: -123.675 / 58.395,
            greenBias: -116.28 / 57.12,
            blueBias: -103.53 / 57.375
        )
    }

    private static func jointMappings()
        throws(VisionModelManifestError)
        -> [VisionPoseJointMapping]
    {
        var mappings = try bodyJointMappings()
        mappings.append(
            contentsOf: try handJointMappings(
                chirality: .left,
                firstIndex: 91
            )
        )
        mappings.append(
            contentsOf: try handJointMappings(
                chirality: .right,
                firstIndex: 112
            )
        )
        return mappings
    }

    private static func bodyJointMappings()
        throws(VisionModelManifestError)
        -> [VisionPoseJointMapping]
    {
        let direct: [
            (HumanBodyPoseObservation.JointName, Int)
        ] = [
            (.nose, 0),
            (.leftEye, 1),
            (.rightEye, 2),
            (.leftEar, 3),
            (.rightEar, 4),
            (.leftShoulder, 5),
            (.rightShoulder, 6),
            (.leftElbow, 7),
            (.rightElbow, 8),
            (.leftWrist, 9),
            (.rightWrist, 10),
            (.leftHip, 11),
            (.rightHip, 12),
            (.leftKnee, 13),
            (.rightKnee, 14),
            (.leftAnkle, 15),
            (.rightAnkle, 16)
        ]
        var mappings: [VisionPoseJointMapping] = []
        mappings.reserveCapacity(direct.count + 2)
        for (joint, index) in direct {
            mappings.append(
                try VisionPoseJointMapping(
                    target: .body(joint),
                    source: .index(index)
                )
            )
        }
        mappings.append(
            try VisionPoseJointMapping(
                target: .body(.neck),
                source: .midpoint(
                    first: 5,
                    second: 6,
                    confidence: .minimum
                )
            )
        )
        mappings.append(
            try VisionPoseJointMapping(
                target: .body(.root),
                source: .midpoint(
                    first: 11,
                    second: 12,
                    confidence: .minimum
                )
            )
        )
        return mappings
    }

    private static func handJointMappings(
        chirality: HumanHandPoseObservation.Chirality,
        firstIndex: Int
    ) throws(VisionModelManifestError)
        -> [VisionPoseJointMapping]
    {
        let joints: [HumanHandPoseObservation.JointName] = [
            .wrist,
            .thumbCMC,
            .thumbMP,
            .thumbIP,
            .thumbTip,
            .indexMCP,
            .indexPIP,
            .indexDIP,
            .indexTip,
            .middleMCP,
            .middlePIP,
            .middleDIP,
            .middleTip,
            .ringMCP,
            .ringPIP,
            .ringDIP,
            .ringTip,
            .littleMCP,
            .littlePIP,
            .littleDIP,
            .littleTip
        ]
        var mappings: [VisionPoseJointMapping] = []
        mappings.reserveCapacity(joints.count)
        for (offset, joint) in joints.enumerated() {
            mappings.append(
                try VisionPoseJointMapping(
                    target: .hand(
                        chirality: chirality,
                        joint: joint
                    ),
                    source: .index(firstIndex + offset)
                )
            )
        }
        return mappings
    }
}
