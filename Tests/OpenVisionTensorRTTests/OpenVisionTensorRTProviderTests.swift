import OpenCoreMedia
import OpenVision
@testable import OpenVisionTensorRT
import Testing

@Suite("OpenVision TensorRT provider")
struct OpenVisionTensorRTProviderTests {
    @Test("Provider advertises the exact GPU transfer contract")
    func providerCapabilities() throws {
        let configuration = try makeProviderConfiguration()
        let provider = try OpenVisionTensorRTProvider(
            configuration: configuration
        )

        #expect(
            provider.capabilities.requests.contains(
                .detectHumanBodyPoseRequest(.revision2)
            )
        )
        #expect(
            provider.capabilities.transferModes == [
                .stagedHostToDevice(fullFrameCopyCount: 1)
            ]
        )
        #expect(
            provider.capabilities.supports(
                OpenVisionTensorRTProvider.cudaDevice,
                for: .main
            )
        )
        #expect(
            provider.capabilities.supports(
                OpenVisionTensorRTProvider.cudaDevice,
                for: .postProcessing
            )
        )
    }

    @Test("Provider requires detector execution capacities")
    func detectorExecutionCapacityValidation() {
        #expect(
            throws:
                TensorRTPosePipelineError
                    .invalidConfiguration("semanticModel")
        ) {
            _ = try makeProviderConfiguration(
                detectionExecutionElementCapacity: nil
            )
        }
        #expect(
            throws:
                TensorRTPosePipelineError
                    .invalidConfiguration("semanticModel")
        ) {
            _ = try makeProviderConfiguration(
                classExecutionElementCapacity:
                    OpenVisionTensorRTProviderConfiguration
                    .detectorExecutionCandidateCount - 1
            )
        }
    }

    @Test("Pose pipeline reports typed unavailability on macOS")
    func unavailablePosePipeline() throws {
        #if os(macOS)
        let configuration = try makeProviderConfiguration()
        do {
            _ = try TensorRTPosePipeline(
                configuration: configuration.posePipeline
            )
            Issue.record("Expected CUDA pose pipeline unavailability")
        } catch let error {
            guard case .unavailable(let report) = error else {
                Issue.record("Expected typed unavailable evidence")
                return
            }
            #expect(report.failureStage == .libraryOpen)
            #expect(
                report.explicitFrameDeviceAllocationCount == 0
            )
        }
        #endif
    }

    @Test("Compact pose tuples become body and hand observations")
    func observationDecoding() throws {
        var joints = [
            TensorRTDecodedPoseJoint(
                normalizedX: 0.5,
                normalizedY: 0.5,
                confidence: 0.1
            )
        ]
        joints = Array(repeating: joints[0], count: 133)
        for index in 0 ..< 17 {
            joints[index] = TensorRTDecodedPoseJoint(
                normalizedX: Float(index + 1) / 20,
                normalizedY: Float(index + 2) / 21,
                confidence: 0.9
            )
        }
        for index in 91 ..< 133 {
            joints[index] = TensorRTDecodedPoseJoint(
                normalizedX: Float(index - 90) / 44,
                normalizedY: Float(index - 89) / 45,
                confidence: 0.8
            )
        }
        let report = TensorRTPosePipelineReport(
            selectedRegionCount: 1,
            regionSelectionKernelLaunchCount: 1,
            regionAffineKernelLaunchCount: 1,
            simCCDecodeKernelLaunchCount: 1,
            compactDeviceToHostCopyCount: 1,
            explicitFrameDeviceAllocationCount: 0,
            persistentDeviceAllocationByteCount: 2_000_000,
            poseInputByteCount: 589_824,
            compactReadbackByteCount: 1_596
        )
        let batch = TensorRTDecodedPoseBatch(
            joints: joints,
            regionCount: 1,
            jointCount: 133,
            report: report
        )
        let regions = [
            TensorRTPoseRegion(
                centerX: 960,
                centerY: 540,
                width: 400,
                height: 700,
                confidence: 0.75
            )
        ]
        var request = DetectHumanBodyPoseRequest(.revision2)
        request.detectsHands = true
        let sessionID = VisionProviderSessionID(high: 1, low: 2)
        let observations = try TensorRTPoseObservationDecoder(
            minimumConfidence: 0.3
        ).observations(
            batch: batch,
            regions: regions,
            timing: CMSampleTimingInfo(
                duration: CMTime(value: 1, timescale: 30),
                presentationTimeStamp: .zero,
                decodeTimeStamp: .invalid
            ),
            provenance: .unattributedNormalizedImage,
            request: request,
            executionID: VisionExecutionID(
                sessionID: sessionID,
                sequence: 7
            )
        )

        let observation = try #require(observations.first)
        #expect(observations.count == 1)
        #expect(observation.availableJointNames.count == 19)
        #expect(observation.leftHand?.availableJointNames.count == 21)
        #expect(observation.rightHand?.availableJointNames.count == 21)
        #expect(observation.confidence == 0.75)
        #expect(
            observation.provenance ==
                .unattributedNormalizedImage
        )
        #expect(
            observation.joint(for: .neck)?.confidence == 0.9
        )
        #expect(
            observation.joint(for: .root)?.confidence == 0.9
        )
        #expect(
            observation.id ==
                VisionObservationID(high: 7, low: 0)
        )
    }

    @Test("Decoder rejects mismatched compact output shape")
    func decoderShapeFailure() {
        let batch = TensorRTDecodedPoseBatch(
            joints: [],
            regionCount: 0,
            jointCount: 132,
            report: TensorRTPosePipelineReport(
                selectedRegionCount: 0,
                regionSelectionKernelLaunchCount: 1,
                regionAffineKernelLaunchCount: 0,
                simCCDecodeKernelLaunchCount: 0,
                compactDeviceToHostCopyCount: 1,
                explicitFrameDeviceAllocationCount: 0,
                persistentDeviceAllocationByteCount: 1,
                poseInputByteCount: 0,
                compactReadbackByteCount: 0
            )
        )
        let request = DetectHumanBodyPoseRequest(.revision2)
        let sessionID = VisionProviderSessionID(high: 1, low: 2)

        #expect(throws: VisionError.self) {
            _ = try TensorRTPoseObservationDecoder(
                minimumConfidence: 0.3
            ).observations(
                batch: batch,
                regions: [],
                timing: CMSampleTimingInfo(
                    duration: .zero,
                    presentationTimeStamp: .zero,
                    decodeTimeStamp: .invalid
                ),
                provenance: .unattributedNormalizedImage,
                request: request,
                executionID: VisionExecutionID(
                    sessionID: sessionID,
                    sequence: 1
                )
            )
        }
    }

    private func makeProviderConfiguration(
        detectionExecutionElementCapacity: Int? =
            OpenVisionTensorRTProviderConfiguration
            .detectorExecutionCandidateCount * 5,
        classExecutionElementCapacity: Int? =
            OpenVisionTensorRTProviderConfiguration
            .detectorExecutionCandidateCount
    )
        throws -> OpenVisionTensorRTProviderConfiguration
    {
        let manifest =
            try RTMDetDWPoseBodyPoseManifest.manifest()
        let artifact = try TensorRTEngineArtifactDescriptor(
            semanticModel: manifest,
            checksum: String(repeating: "0", count: 64),
            tensorRTVersion: 101_602,
            cudaRuntimeVersion: 13_020,
            computeCapabilityMajor: 8,
            computeCapabilityMinor: 7,
            precision: .float16
        )
        let detector = try TensorRTStageEngineArtifactDescriptor(
            artifact: artifact,
            stageID:
                RTMDetDWPoseBodyPoseManifest.personDetectionStage,
            inputTensorName: "input",
            outputBindings: [
                try TensorRTEngineOutputBinding(
                    semanticTensorID:
                        RTMDetDWPoseBodyPoseManifest
                        .detectionsTensor,
                    engineTensorName: "dets",
                    executionElementCapacity:
                        detectionExecutionElementCapacity
                ),
                try TensorRTEngineOutputBinding(
                    semanticTensorID:
                        RTMDetDWPoseBodyPoseManifest.classesTensor,
                    engineTensorName: "labels",
                    executionElementCapacity:
                        classExecutionElementCapacity
                ),
            ]
        )
        let pose = try TensorRTStageEngineArtifactDescriptor(
            artifact: artifact,
            stageID:
                RTMDetDWPoseBodyPoseManifest.wholeBodyPoseStage,
            inputTensorName: "input",
            outputBindings: [
                try TensorRTEngineOutputBinding(
                    semanticTensorID:
                        RTMDetDWPoseBodyPoseManifest.simCCXTensor,
                    engineTensorName: "simcc_x"
                ),
                try TensorRTEngineOutputBinding(
                    semanticTensorID:
                        RTMDetDWPoseBodyPoseManifest.simCCYTensor,
                    engineTensorName: "simcc_y"
                ),
            ]
        )
        let detectorInput = try #require(
            manifest.stage(
                identifiedBy:
                    RTMDetDWPoseBodyPoseManifest.personDetectionStage
            )?.input
        )
        let preprocessing = try RG10PreprocessingConfiguration(
            sourceWidth: 1920,
            sourceHeight: 1080,
            sourceBytesPerRow: 3840,
            sourceByteCount: 4_147_200,
            wordLayout: .leastSignificantBits,
            outputWidth: detectorInput.width,
            outputHeight: detectorInput.height,
            resizePolicy: detectorInput.resizePolicy,
            tensorLayout: detectorInput.tensorLayout,
            channelOrder: detectorInput.channelOrder,
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
            letterboxColor: .init(
                red: detectorInput.letterboxColor.red,
                green: detectorInput.letterboxColor.green,
                blue: detectorInput.letterboxColor.blue
            ),
            normalization: detectorInput.normalization,
            appliesSRGBTransfer:
                detectorInput.transferFunction == .sRGB
        )
        return try OpenVisionTensorRTProviderConfiguration(
            model: manifest,
            detectorPlanPath: "/tmp/detector.plan",
            detectorArtifact: detector,
            posePlanPath: "/tmp/pose.plan",
            poseArtifact: pose,
            detectorPreprocessing: preprocessing
        )
    }
}
