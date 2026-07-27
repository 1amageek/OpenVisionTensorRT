import CTensorRTShim
import OpenVision

/// Owns the reusable CUDA region-affine and SimCC decode allocations.
///
/// The actor permits one prepared frame at a time. Detector outputs and the
/// original RG10 device allocation are borrowed until region preparation has
/// synchronized. Only bounded ROI metadata and compact joint tuples cross the
/// device-to-host boundary.
public actor TensorRTPosePipeline {
    private let configuration: TensorRTPosePipelineConfiguration
    private let owner: TensorRTPosePipelineHandleOwner
    private var regionReadbackStorage: [OVTRTPoseRegion]
    private var jointReadbackStorage: [OVTRTPoseJoint]
    private var activeInputLease: TensorRTPoseInputLease?
    private var decodeIsPending = false
    private var selectedRegionCount = 0

    public init(
        configuration: TensorRTPosePipelineConfiguration
    ) throws(TensorRTPosePipelineError) {
        let deferredCleanup =
            TensorRTPoseDeferredCleanupRegistry.shared.retry()
        guard deferredCleanup.remainingOwnerCount == 0 else {
            throw .deferredCleanupPending(deferredCleanup)
        }

        var rawConfiguration = Self.rawConfiguration(configuration)
        var handle: OpaquePointer?
        var rawResult = OVTRTPosePipelineResult()
        let status = TensorRTRuntimeStatus(
            ovtrt_pose_pipeline_create(
                &rawConfiguration,
                &handle,
                &rawResult
            )
        )
        guard status == .available, let handle else {
            let operation = TensorRTPosePipelineReport(rawResult)
            if let handle {
                let failedOwner =
                    TensorRTPosePipelineHandleOwner(handle: handle)
                var cleanupResult = OVTRTPosePipelineResult()
                let cleanupStatus =
                    failedOwner.destroy(result: &cleanupResult)
                if cleanupStatus != .available {
                    TensorRTPoseDeferredCleanupRegistry.shared
                        .enqueue(failedOwner)
                    throw .creationFailed(
                        status: status,
                        operation: operation,
                        cleanup:
                            TensorRTPosePipelineReport(cleanupResult)
                    )
                }
            }
            if status == .unavailable {
                throw .unavailable(operation)
            }
            throw .failed(status: status, report: operation)
        }
        self.configuration = configuration
        owner = TensorRTPosePipelineHandleOwner(handle: handle)
        regionReadbackStorage = [OVTRTPoseRegion](
            repeating: OVTRTPoseRegion(),
            count: configuration.maximumRegionCount
        )
        jointReadbackStorage = [OVTRTPoseJoint](
            repeating: OVTRTPoseJoint(),
            count:
                configuration.maximumRegionCount *
                configuration.jointCount
        )
    }

    deinit {
        guard owner.isActive else {
            return
        }
        if let activeInputLease, !activeInputLease.isReleased {
            TensorRTPoseDeferredCleanupRegistry.shared.enqueue(
                owner,
                waitingFor: activeInputLease
            )
            return
        }
        var result = OVTRTPosePipelineResult()
        let status = owner.destroy(result: &result)
        if status != .available {
            TensorRTPoseDeferredCleanupRegistry.shared.enqueue(
                owner
            )
        }
    }

    public static func retryDeferredCleanup()
        -> TensorRTPoseDeferredCleanupResult
    {
        TensorRTPoseDeferredCleanupRegistry.shared.retry()
    }

    public var isActive: Bool {
        owner.isActive
    }

    public func prepareInput(
        source: RG10DeviceTensor,
        detections: TensorRTDeviceOutputTensor,
        classes: TensorRTDeviceOutputTensor,
        orientation: VisionImageOrientation
    ) throws(TensorRTPosePipelineError)
        -> TensorRTPosePreparation
    {
        guard owner.isActive else {
            throw .alreadyShutDown
        }
        guard !decodeIsPending else {
            throw .decodePending
        }
        if let activeInputLease {
            guard activeInputLease.isReleased else {
                throw .outputInUse
            }
            self.activeInputLease = nil
        }
        guard
            detections.elementType == .float32,
            classes.elementType == .int64,
            detections.shape.count == 3,
            detections.shape[0] == 1,
            detections.shape[2] == 5,
            classes.shape.count == 2,
            classes.shape[0] == 1,
            classes.shape[1] == detections.shape[1],
            detections.shape[1] > 0,
            detections.shape[1] <= 100
        else {
            throw .invalidDetectorOutput
        }

        let prepared: PreparedInputResult
        do {
            prepared = try preparedInput(
                source: source,
                detections: detections,
                classes: classes,
                orientation: orientation
            )
        } catch {
            throw .invalidDetectorOutput
        }
        guard prepared.status == .available else {
            throw .failed(
                status: prepared.status,
                report: prepared.report
            )
        }
        let report = prepared.report
        let regions = prepared.regions
        selectedRegionCount = regions.count
        guard !regions.isEmpty else {
            return TensorRTPosePreparation(
                input: nil,
                regions: [],
                report: report
            )
        }
        guard
            let deviceAddress = prepared.deviceAddress,
            prepared.byteCount > 0,
            prepared.width == configuration.poseInputWidth,
            prepared.height == configuration.poseInputHeight
        else {
            throw .invalidPoseOutput
        }
        let lease = TensorRTPoseInputLease()
        let input = TensorRTPoseInput(
            address: deviceAddress,
            byteCount: prepared.byteCount,
            batchSize: regions.count,
            width: configuration.poseInputWidth,
            height: configuration.poseInputHeight,
            owner: owner,
            lease: lease
        )
        activeInputLease = lease
        decodeIsPending = true
        return TensorRTPosePreparation(
            input: input,
            regions: regions,
            report: report
        )
    }

    public func decode(
        simCCX: TensorRTDeviceOutputTensor,
        simCCY: TensorRTDeviceOutputTensor
    ) throws(TensorRTPosePipelineError)
        -> TensorRTDecodedPoseBatch
    {
        guard owner.isActive else {
            throw .alreadyShutDown
        }
        guard decodeIsPending, selectedRegionCount > 0 else {
            throw .noDecodePending
        }
        let expectedXShape = [
            selectedRegionCount,
            configuration.jointCount,
            configuration.poseInputWidth * 2
        ]
        let expectedYShape = [
            selectedRegionCount,
            configuration.jointCount,
            configuration.poseInputHeight * 2
        ]
        guard
            simCCX.elementType == .float32,
            simCCY.elementType == .float32,
            simCCX.shape == expectedXShape,
            simCCY.shape == expectedYShape
        else {
            throw .invalidPoseOutput
        }
        let compactJointCount =
            selectedRegionCount * configuration.jointCount
        guard compactJointCount <= jointReadbackStorage.count else {
            throw .invalidPoseOutput
        }
        var rawResult = OVTRTPosePipelineResult()
        var rawStatus = OVTRTStatusInvalidArgument
        do {
            try simCCX.withDeviceAddress { xAddress, _ in
                try simCCY.withDeviceAddress { yAddress, _ in
                    rawStatus =
                        jointReadbackStorage
                        .withUnsafeMutableBufferPointer {
                            jointBuffer in
                            owner.withHandle { handle in
                                ovtrt_pose_pipeline_decode_simcc(
                                    handle,
                                    UnsafeRawPointer(
                                        bitPattern: xAddress
                                    ),
                                    UInt64(simCCX.elementCount),
                                    UnsafeRawPointer(
                                        bitPattern: yAddress
                                    ),
                                    UInt64(simCCY.elementCount),
                                    jointBuffer.baseAddress,
                                    UInt64(compactJointCount),
                                    &rawResult
                                )
                            } ?? OVTRTStatusResourceBusy
                        }
                    }
                }
        } catch {
            throw .invalidPoseOutput
        }
        let status = TensorRTRuntimeStatus(rawStatus)
        let report = TensorRTPosePipelineReport(rawResult)
        guard status == .available else {
            throw .failed(status: status, report: report)
        }
        decodeIsPending = false
        let batch = TensorRTDecodedPoseBatch(
            joints:
                jointReadbackStorage[..<compactJointCount]
                .map(TensorRTDecodedPoseJoint.init),
            regionCount: selectedRegionCount,
            jointCount: configuration.jointCount,
            report: report
        )
        selectedRegionCount = 0
        return batch
    }

    public func discardPreparedFrame()
        throws(TensorRTPosePipelineError)
    {
        guard owner.isActive else {
            throw .alreadyShutDown
        }
        decodeIsPending = false
        selectedRegionCount = 0
    }

    public func shutdown() throws(TensorRTPosePipelineError) {
        guard owner.isActive else {
            throw .alreadyShutDown
        }
        guard !decodeIsPending else {
            throw .decodePending
        }
        if let activeInputLease {
            guard activeInputLease.isReleased else {
                throw .outputInUse
            }
            self.activeInputLease = nil
        }
        var rawResult = OVTRTPosePipelineResult()
        let status = owner.destroy(result: &rawResult)
        guard status == .available else {
            throw .failed(
                status: status,
                report: TensorRTPosePipelineReport(rawResult)
            )
        }
    }

    private struct PreparedInputResult: Sendable {
        let status: TensorRTRuntimeStatus
        let report: TensorRTPosePipelineReport
        let regions: [TensorRTPoseRegion]
        let deviceAddress: UInt?
        let byteCount: Int
        let width: Int
        let height: Int
    }

    private struct PreparedDeviceInputResult: Sendable {
        let status: TensorRTRuntimeStatus
        let report: TensorRTPosePipelineReport
        let deviceAddress: UInt?
        let byteCount: Int
        let width: Int
        let height: Int
    }

    private func preparedInput(
        source: RG10DeviceTensor,
        detections: TensorRTDeviceOutputTensor,
        classes: TensorRTDeviceOutputTensor,
        orientation: VisionImageOrientation
    ) throws -> PreparedInputResult {
        let prepared = try regionReadbackStorage
            .withUnsafeMutableBufferPointer { regionBuffer in
                try Self.preparedDeviceInput(
                    source: source,
                    detections: detections,
                    classes: classes,
                    orientation: orientation,
                    regionBuffer: regionBuffer,
                    owner: owner
                )
            }
        let report = prepared.report
        guard
            report.selectedRegionCount >= 0,
            report.selectedRegionCount <=
                regionReadbackStorage.count
        else {
            throw TensorRTPosePipelineError.invalidDetectorOutput
        }
        return PreparedInputResult(
            status: prepared.status,
            report: report,
            regions:
                regionReadbackStorage[
                    ..<report.selectedRegionCount
                ].map(TensorRTPoseRegion.init),
            deviceAddress: prepared.deviceAddress,
            byteCount: prepared.byteCount,
            width: prepared.width,
            height: prepared.height
        )
    }

    private static func preparedDeviceInput(
        source: RG10DeviceTensor,
        detections: TensorRTDeviceOutputTensor,
        classes: TensorRTDeviceOutputTensor,
        orientation: VisionImageOrientation,
        regionBuffer:
            UnsafeMutableBufferPointer<OVTRTPoseRegion>,
        owner: TensorRTPosePipelineHandleOwner
    ) throws -> PreparedDeviceInputResult {
        var descriptor = OVTRTDeviceTensorView()
        var rawResult = OVTRTPosePipelineResult()
        var rawStatus = OVTRTStatusInvalidArgument
        try source.withRG10SourceDeviceView {
            borrowedSourceView in
            var sourceView = borrowedSourceView
            try detections.withDeviceAddress {
                detectionAddress,
                _ in
                try classes.withDeviceAddress {
                    classAddress,
                    _ in
                    rawStatus = withUnsafePointer(
                        to: &sourceView
                    ) { sourcePointer in
                        owner.withHandle { handle in
                            ovtrt_pose_pipeline_prepare_input(
                                handle,
                                sourcePointer,
                                UnsafeRawPointer(
                                    bitPattern: detectionAddress
                                ),
                                UInt64(detections.shape[1]),
                                UnsafeRawPointer(
                                    bitPattern: classAddress
                                ),
                                UInt64(classes.shape[1]),
                                Self.rawOrientation(orientation),
                                regionBuffer.baseAddress,
                                UInt32(regionBuffer.count),
                                &descriptor,
                                &rawResult
                            )
                        }
                        ?? OVTRTStatusResourceBusy
                    }
                }
            }
        }
        let report = TensorRTPosePipelineReport(rawResult)
        return PreparedDeviceInputResult(
            status: TensorRTRuntimeStatus(rawStatus),
            report: report,
            deviceAddress: descriptor.deviceAddress.map {
                UInt(bitPattern: $0)
            },
            byteCount: Int(exactly: descriptor.byteCount) ?? 0,
            width: Int(descriptor.width),
            height: Int(descriptor.height)
        )
    }

    private static func rawConfiguration(
        _ value: TensorRTPosePipelineConfiguration
    ) -> OVTRTPosePipelineConfiguration {
        let source = value.source
        let normalization = value.poseNormalization
        return OVTRTPosePipelineConfiguration(
            sourceWidth: UInt32(source.sourceWidth),
            sourceHeight: UInt32(source.sourceHeight),
            sourceBytesPerRow: UInt32(source.sourceBytesPerRow),
            sourceWordLayout:
                source.wordLayout == .leastSignificantBits
                ? OVTRTRG10WordLayoutLeastSignificantBits
                : OVTRTRG10WordLayoutMostSignificantBits,
            detectorInputWidth: UInt32(value.detectorInputWidth),
            detectorInputHeight: UInt32(value.detectorInputHeight),
            poseInputWidth: UInt32(value.poseInputWidth),
            poseInputHeight: UInt32(value.poseInputHeight),
            maximumRegionCount: UInt32(value.maximumRegionCount),
            jointCount: UInt32(value.jointCount),
            minimumDetectionConfidence:
                value.minimumDetectionConfidence,
            maximumDetectionOverlap:
                value.maximumDetectionOverlap,
            regionScale: value.regionScale,
            blackLevelR: source.blackLevels.red,
            blackLevelGreenR:
                source.blackLevels.greenOnRedRow,
            blackLevelGreenB:
                source.blackLevels.greenOnBlueRow,
            blackLevelB: source.blackLevels.blue,
            whiteLevel: source.whiteLevel,
            gainR: source.gains.red,
            gainGreenR: source.gains.greenOnRedRow,
            gainGreenB: source.gains.greenOnBlueRow,
            gainB: source.gains.blue,
            colorMatrix00: source.colorMatrix.row0.red,
            colorMatrix01: source.colorMatrix.row0.green,
            colorMatrix02: source.colorMatrix.row0.blue,
            colorMatrix10: source.colorMatrix.row1.red,
            colorMatrix11: source.colorMatrix.row1.green,
            colorMatrix12: source.colorMatrix.row1.blue,
            colorMatrix20: source.colorMatrix.row2.red,
            colorMatrix21: source.colorMatrix.row2.green,
            colorMatrix22: source.colorMatrix.row2.blue,
            normalizationScaleR: normalization.scale.red,
            normalizationScaleG: normalization.scale.green,
            normalizationScaleB: normalization.scale.blue,
            normalizationBiasR: normalization.bias.red,
            normalizationBiasG: normalization.bias.green,
            normalizationBiasB: normalization.bias.blue,
            applySRGBTransfer: source.appliesSRGBTransfer ? 1 : 0
        )
    }

    private static func rawOrientation(
        _ value: VisionImageOrientation
    ) -> OVTRTRG10Orientation {
        switch value {
        case .up:
            OVTRTRG10OrientationUp
        case .upMirrored:
            OVTRTRG10OrientationUpMirrored
        case .down:
            OVTRTRG10OrientationDown
        case .downMirrored:
            OVTRTRG10OrientationDownMirrored
        case .leftMirrored:
            OVTRTRG10OrientationLeftMirrored
        case .right:
            OVTRTRG10OrientationRight
        case .rightMirrored:
            OVTRTRG10OrientationRightMirrored
        case .left:
            OVTRTRG10OrientationLeft
        }
    }
}
