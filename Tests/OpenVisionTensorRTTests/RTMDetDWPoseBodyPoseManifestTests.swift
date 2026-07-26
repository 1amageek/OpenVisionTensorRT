import OpenVision
@testable import OpenVisionTensorRT
import Testing

@Suite("RTMDet and DWPose semantic manifest")
struct RTMDetDWPoseBodyPoseManifestTests {
    @Test("Manifest fixes both model stages and exact provenance")
    func modelStagesAndProvenance() throws {
        let manifest = try RTMDetDWPoseBodyPoseManifest.manifest()

        #expect(
            manifest.id ==
                "openmmlab.rtmdet-nano-dwpose-m.wholebody"
        )
        #expect(manifest.stages.count == 2)
        let detector = try #require(
            manifest.stage(
                identifiedBy:
                    RTMDetDWPoseBodyPoseManifest
                        .personDetectionStage
            )
        )
        let pose = try #require(
            manifest.stage(
                identifiedBy:
                    RTMDetDWPoseBodyPoseManifest
                        .wholeBodyPoseStage
            )
        )

        #expect(detector.input.width == 320)
        #expect(detector.input.height == 320)
        #expect(detector.input.resizePolicy == .scaleFit)
        #expect(detector.input.channelOrder == .bgr)
        #expect(
            detector.provenance.sourceDigest.hexadecimal ==
                "05d8511e7b3fabc62e27d2f624179e004ad14ee63a86ca9d9d22c88f3db0eee1"
        )
        #expect(pose.input.width == 192)
        #expect(pose.input.height == 256)
        #expect(pose.input.resizePolicy == .regionAffine)
        #expect(pose.input.channelOrder == .rgb)
        #expect(
            pose.provenance.sourceDigest.hexadecimal ==
                "c8b7641988ce785c987b8ad156e411f328b54766d353df19aea5b08433ef1aaf"
        )
        #expect(detector.provenance.licenseIdentifier == nil)
        #expect(pose.provenance.licenseIdentifier == nil)
    }

    @Test("Pose stage consumes bounded detector regions")
    func boundedRegionSource() throws {
        let manifest = try RTMDetDWPoseBodyPoseManifest.manifest()
        let pose = try #require(
            manifest.stage(
                identifiedBy:
                    RTMDetDWPoseBodyPoseManifest
                        .wholeBodyPoseStage
            )
        )

        guard case .regions(
            let stage,
            let tensor,
            let minimumConfidence,
            let maximumCount,
            let scale
        ) = pose.input.source else {
            Issue.record("Expected a detector region source")
            return
        }
        #expect(
            stage ==
                RTMDetDWPoseBodyPoseManifest
                    .personDetectionStage
        )
        #expect(
            tensor ==
                RTMDetDWPoseBodyPoseManifest
                    .detectionsTensor
        )
        #expect(minimumConfidence == 0.3)
        #expect(maximumCount == 4)
        #expect(scale == 1.25)
    }

    @Test("SimCC shapes and channel normalization are exact")
    func tensorsAndNormalization() throws {
        let manifest = try RTMDetDWPoseBodyPoseManifest.manifest()
        let detector = try #require(
            manifest.stage(
                identifiedBy:
                    RTMDetDWPoseBodyPoseManifest
                        .personDetectionStage
            )
        )
        let pose = try #require(
            manifest.stage(
                identifiedBy:
                    RTMDetDWPoseBodyPoseManifest
                        .wholeBodyPoseStage
            )
        )
        let xTensor = try #require(
            pose.output(
                identifiedBy:
                    RTMDetDWPoseBodyPoseManifest.simCCXTensor
            )
        )
        let yTensor = try #require(
            pose.output(
                identifiedBy:
                    RTMDetDWPoseBodyPoseManifest.simCCYTensor
            )
        )

        #expect(
            xTensor.shape == [
                .batch(maximum: 4),
                .fixed(133),
                .fixed(384)
            ]
        )
        #expect(
            yTensor.shape == [
                .batch(maximum: 4),
                .fixed(133),
                .fixed(512)
            ]
        )
        #expect(
            detector.input.normalization ==
                pose.input.normalization
        )
        #expect(
            detector.input.normalization.scale.red ==
                Float(255.0) / Float(58.395)
        )
        #expect(
            detector.input.normalization.bias.blue ==
                -Float(103.53) / Float(57.375)
        )
    }

    @Test("Output covers body and both hands without ambiguity")
    func completeJointMapping() throws {
        let manifest = try RTMDetDWPoseBodyPoseManifest.manifest()
        let mappings = manifest.output.jointMappings

        let bodyCount = mappings.count {
            if case .body = $0.target {
                true
            } else {
                false
            }
        }
        let leftHandCount = mappings.count {
            if case .hand(.left, _) = $0.target {
                true
            } else {
                false
            }
        }
        let rightHandCount = mappings.count {
            if case .hand(.right, _) = $0.target {
                true
            } else {
                false
            }
        }

        #expect(bodyCount == 19)
        #expect(leftHandCount == 21)
        #expect(rightHandCount == 21)
        #expect(mappings.count == 61)
        #expect(
            mappings.contains {
                $0.target == .body(.neck) &&
                $0.source == .midpoint(
                    first: 5,
                    second: 6,
                    confidence: .minimum
                )
            }
        )
        #expect(
            mappings.contains {
                $0.target == .body(.root) &&
                $0.source == .midpoint(
                    first: 11,
                    second: 12,
                    confidence: .minimum
                )
            }
        )
    }
}
