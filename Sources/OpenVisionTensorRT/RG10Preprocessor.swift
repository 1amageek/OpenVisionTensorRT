import CTensorRTShim
import OpenVision

/// Owns one reusable CUDA preprocessing graph.
///
/// `process(_:)` performs its one H2D copy synchronously inside the
/// `VisionImageInput` byte borrow. The returned tensor lease must be released
/// before processing another frame or shutting down.
public actor RG10Preprocessor {
    private let configuration: RG10PreprocessingConfiguration
    private let preparationResult: OVTRTRG10PreprocessingResult
    private let owner: RG10PreprocessorHandleOwner
    private var activeTensorLease: RG10TensorLeaseState?

    public init(
        configuration: RG10PreprocessingConfiguration
    ) throws(RG10PreprocessorError) {
        let deferredCleanup =
            RG10DeferredCleanupRegistry.shared.retry()
        guard deferredCleanup.remainingOwnerCount == 0 else {
            throw .deferredCleanupPending(deferredCleanup)
        }

        var rawConfiguration = Self.rawConfiguration(configuration)
        var handle: OpaquePointer?
        var rawResult = OVTRTRG10PreprocessingResult()
        let rawStatus = ovtrt_rg10_preprocessor_create(
            &rawConfiguration,
            &handle,
            &rawResult
        )
        let status = TensorRTRuntimeStatus(rawStatus)
        guard status == .available, let handle else {
            if let handle {
                let failedOwner =
                    RG10PreprocessorHandleOwner(handle: handle)
                var cleanupResult =
                    OVTRTRG10PreprocessingResult()
                let cleanupStatus =
                    failedOwner.destroy(result: &cleanupResult)
                if cleanupStatus != .available {
                    RG10DeferredCleanupRegistry.shared.enqueue(
                        failedOwner
                    )
                    throw .creationFailed(
                        status: status,
                        operation: RG10PreprocessingReport(rawResult),
                        cleanup:
                            RG10PreprocessingReport(cleanupResult)
                    )
                }
            }
            let report = RG10PreprocessingReport(rawResult)
            if status == .unavailable {
                throw .unavailable(report)
            }
            throw .failed(status: status, report: report)
        }

        self.configuration = configuration
        preparationResult = rawResult
        owner = RG10PreprocessorHandleOwner(handle: handle)
        activeTensorLease = nil
    }

    deinit {
        guard owner.isActive else {
            return
        }
        if let activeTensorLease, !activeTensorLease.isReleased {
            RG10DeferredCleanupRegistry.shared.enqueue(
                owner,
                waitingFor: activeTensorLease
            )
            return
        }
        var rawResult = OVTRTRG10PreprocessingResult()
        let status = owner.destroy(result: &rawResult)
        if status != .available {
            RG10DeferredCleanupRegistry.shared.enqueue(owner)
        }
    }

    public static func retryDeferredCleanup()
        -> RG10DeferredCleanupResult
    {
        RG10DeferredCleanupRegistry.shared.retry()
    }

    public var isActive: Bool {
        owner.isActive
    }

    public func process(
        _ input: VisionImageInput
    ) throws(RG10PreprocessorError) -> RG10PreprocessingOutput {
        guard owner.isActive else {
            throw .alreadyShutDown
        }
        if let activeTensorLease {
            guard activeTensorLease.isReleased else {
                throw .outputInUse
            }
            self.activeTensorLease = nil
        }
        try validate(input)

        var submission = OVTRTRG10PreprocessingResult()
        let rawStatus: OVTRTStatus
        do {
            rawStatus = try input.withReadBytes { bytes in
                bytes.withUnsafeBufferPointer { buffer in
                    guard let baseAddress = buffer.baseAddress else {
                        return OVTRTStatusInvalidArgument
                    }
                    return owner.withHandle { handle in
                        ovtrt_rg10_preprocessor_submit(
                            handle,
                            baseAddress,
                            UInt64(buffer.count),
                            Self.rawOrientation(input.orientation),
                            &submission
                        )
                    } ?? OVTRTStatusResourceBusy
                }
            }
        } catch let error {
            throw .invalidInput(error)
        }

        let submissionStatus = TensorRTRuntimeStatus(rawStatus)
        input.releaseInput()
        guard submissionStatus == .available else {
            throw .failed(
                status: submissionStatus,
                report: RG10PreprocessingReport(
                    preparation: preparationResult,
                    submission: submission,
                    completion: submission
                )
            )
        }

        var completion = OVTRTRG10PreprocessingResult()
        let completionStatus = owner.withHandle { handle in
            TensorRTRuntimeStatus(
                ovtrt_rg10_preprocessor_wait(
                    handle,
                    &completion
                )
            )
        } ?? .resourceBusy
        let report = RG10PreprocessingReport(
            preparation: preparationResult,
            submission: submission,
            completion: completion
        )
        guard completionStatus == .available else {
            throw .failed(status: completionStatus, report: report)
        }
        guard report.satisfiesFrameContract else {
            throw .failed(
                status: .preprocessingFailure,
                report: report
            )
        }

        var descriptor = OVTRTDeviceTensorView()
        let outputStatus = owner.withHandle { handle in
            TensorRTRuntimeStatus(
                ovtrt_rg10_preprocessor_output(
                    handle,
                    &descriptor
                )
            )
        } ?? .resourceBusy
        guard outputStatus == .available else {
            throw .failed(status: outputStatus, report: report)
        }
        let lease = RG10TensorLeaseState()
        let tensor = try RG10DeviceTensor(
            descriptor: descriptor,
            owner: owner,
            lease: lease
        )
        activeTensorLease = lease
        return RG10PreprocessingOutput(
            tensor: tensor,
            report: report
        )
    }

    public func shutdown() throws(RG10PreprocessorError) {
        guard owner.isActive else {
            throw .alreadyShutDown
        }
        if let activeTensorLease {
            guard activeTensorLease.isReleased else {
                throw .outputInUse
            }
            self.activeTensorLease = nil
        }
        var rawResult = OVTRTRG10PreprocessingResult()
        let status = owner.destroy(result: &rawResult)
        guard status == .available else {
            throw .failed(
                status: status,
                report: RG10PreprocessingReport(rawResult)
            )
        }
    }

    private func validate(
        _ input: VisionImageInput
    ) throws(RG10PreprocessorError) {
        guard input.pixelFormat.rawValue ==
                RG10PreprocessingConfiguration.pixelFormatRawValue
        else {
            throw .incompatibleInput(
                .pixelFormat(actual: input.pixelFormat.rawValue)
            )
        }
        guard
            input.dimensions.width == configuration.sourceWidth,
            input.dimensions.height == configuration.sourceHeight
        else {
            throw .incompatibleInput(
                .dimensions(
                    actualWidth: input.dimensions.width,
                    actualHeight: input.dimensions.height,
                    expectedWidth: configuration.sourceWidth,
                    expectedHeight: configuration.sourceHeight
                )
            )
        }
        guard case .packed(let bytesPerRow, let byteCount) =
                input.layout.storage
        else {
            throw .incompatibleInput(.planarStorage)
        }
        guard bytesPerRow == configuration.sourceBytesPerRow else {
            throw .incompatibleInput(
                .bytesPerRow(
                    actual: bytesPerRow,
                    expected: configuration.sourceBytesPerRow
                )
            )
        }
        guard byteCount >= configuration.sourceByteCount else {
            throw .incompatibleInput(
                .byteCount(
                    actual: byteCount,
                    expected: configuration.sourceByteCount
                )
            )
        }
        guard input.storage.transferModes.contains(
            .stagedHostToDevice(fullFrameCopyCount: 1)
        ) else {
            throw .incompatibleInput(.transferMode)
        }
    }

    private static func rawConfiguration(
        _ value: RG10PreprocessingConfiguration
    ) -> OVTRTRG10PreprocessingConfiguration {
        let normalization: (Float, Float)
        switch value.normalization {
        case .zeroToOne:
            normalization = (1, 0)
        case .negativeOneToOne:
            normalization = (2, -1)
        case .affine(let scale, let bias):
            normalization = (scale, bias)
        }
        return OVTRTRG10PreprocessingConfiguration(
            sourceWidth: UInt32(value.sourceWidth),
            sourceHeight: UInt32(value.sourceHeight),
            sourceBytesPerRow: UInt32(value.sourceBytesPerRow),
            sourceByteCount: UInt64(value.sourceByteCount),
            outputWidth: UInt32(value.outputWidth),
            outputHeight: UInt32(value.outputHeight),
            resizePolicy: rawResizePolicy(value.resizePolicy),
            tensorLayout:
                value.tensorLayout == .channelsFirst
                ? OVTRTTensorLayoutNCHW
                : OVTRTTensorLayoutNHWC,
            channelOrder:
                value.channelOrder == .rgb
                ? OVTRTTensorChannelOrderRGB
                : OVTRTTensorChannelOrderBGR,
            blackLevelR: value.blackLevels.red,
            blackLevelGreenR: value.blackLevels.greenOnRedRow,
            blackLevelGreenB: value.blackLevels.greenOnBlueRow,
            blackLevelB: value.blackLevels.blue,
            whiteLevel: value.whiteLevel,
            gainR: value.gains.red,
            gainGreenR: value.gains.greenOnRedRow,
            gainGreenB: value.gains.greenOnBlueRow,
            gainB: value.gains.blue,
            colorMatrix00: value.colorMatrix.row0.red,
            colorMatrix01: value.colorMatrix.row0.green,
            colorMatrix02: value.colorMatrix.row0.blue,
            colorMatrix10: value.colorMatrix.row1.red,
            colorMatrix11: value.colorMatrix.row1.green,
            colorMatrix12: value.colorMatrix.row1.blue,
            colorMatrix20: value.colorMatrix.row2.red,
            colorMatrix21: value.colorMatrix.row2.green,
            colorMatrix22: value.colorMatrix.row2.blue,
            letterboxR: value.letterboxColor.red,
            letterboxG: value.letterboxColor.green,
            letterboxB: value.letterboxColor.blue,
            normalizationScale: normalization.0,
            normalizationBias: normalization.1,
            applySRGBTransfer: value.appliesSRGBTransfer ? 1 : 0
        )
    }

    private static func rawResizePolicy(
        _ value: VisionModelInputDescriptor.ResizePolicy
    ) -> OVTRTRG10ResizePolicy {
        switch value {
        case .scaleFill:
            OVTRTRG10ResizePolicyScaleFill
        case .scaleFit:
            OVTRTRG10ResizePolicyScaleFit
        case .centerCrop:
            OVTRTRG10ResizePolicyCenterCrop
        }
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
