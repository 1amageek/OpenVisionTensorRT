import OpenVision

public struct RG10PreprocessingConfiguration: Sendable, Hashable {
    enum SupportedResizePolicy: Sendable, Hashable {
        case scaleFill
        case scaleFit
        case centerCrop
    }

    public static let pixelFormatRawValue: UInt32 = 0x3031_4752
    public static let maximumSourceByteCount = 512 * 1024 * 1024
    public static let maximumOutputElementCount =
        3 * 4096 * 4096

    public let sourceWidth: Int
    public let sourceHeight: Int
    public let sourceBytesPerRow: Int
    public let sourceByteCount: Int
    public let outputWidth: Int
    public let outputHeight: Int
    public let resizePolicy: VisionModelInputDescriptor.ResizePolicy
    public let tensorLayout: VisionTensorLayout
    public let channelOrder: VisionTensorChannelOrder
    public let blackLevels: RG10BayerValues
    public let whiteLevel: Float
    public let gains: RG10BayerValues
    public let colorMatrix: RGBColorMatrix
    public let letterboxColor: RGBTriplet
    public let normalization:
        VisionModelInputDescriptor.Normalization
    public let appliesSRGBTransfer: Bool
    let supportedResizePolicy: SupportedResizePolicy

    public init(
        sourceWidth: Int,
        sourceHeight: Int,
        sourceBytesPerRow: Int,
        sourceByteCount: Int,
        outputWidth: Int,
        outputHeight: Int,
        resizePolicy: VisionModelInputDescriptor.ResizePolicy,
        tensorLayout: VisionTensorLayout,
        channelOrder: VisionTensorChannelOrder,
        blackLevels: RG10BayerValues,
        whiteLevel: Float,
        gains: RG10BayerValues,
        colorMatrix: RGBColorMatrix,
        letterboxColor: RGBTriplet = .zero,
        normalization: VisionModelInputDescriptor.Normalization,
        appliesSRGBTransfer: Bool
    ) throws(RG10PreprocessingConfigurationError) {
        guard
            sourceWidth >= 2,
            sourceHeight >= 2,
            sourceWidth <= Int(UInt32.max),
            sourceHeight <= Int(UInt32.max)
        else {
            throw .invalidSourceDimensions(
                width: sourceWidth,
                height: sourceHeight
            )
        }
        let minimumRow = sourceWidth.multipliedReportingOverflow(by: 2)
        guard
            !minimumRow.overflow,
            sourceBytesPerRow >= minimumRow.partialValue,
            sourceBytesPerRow <= Int(UInt32.max)
        else {
            throw .invalidSourceBytesPerRow(sourceBytesPerRow)
        }
        let minimumBytes = sourceBytesPerRow
            .multipliedReportingOverflow(by: sourceHeight)
        guard
            !minimumBytes.overflow,
            sourceByteCount >= minimumBytes.partialValue,
            sourceByteCount <= Self.maximumSourceByteCount
        else {
            throw .invalidSourceByteCount(sourceByteCount)
        }
        let outputPixelCount =
            outputWidth.multipliedReportingOverflow(by: outputHeight)
        let outputElementCount = outputPixelCount.partialValue
            .multipliedReportingOverflow(by: 3)
        guard
            outputWidth > 0,
            outputHeight > 0,
            outputWidth <= Int(UInt32.max),
            outputHeight <= Int(UInt32.max),
            !outputPixelCount.overflow,
            !outputElementCount.overflow,
            outputElementCount.partialValue <=
                Self.maximumOutputElementCount
        else {
            throw .invalidOutputDimensions(
                width: outputWidth,
                height: outputHeight
            )
        }
        guard
            Self.isFinite(whiteLevel),
            whiteLevel > 0,
            whiteLevel <= 1023
        else {
            throw .invalidWhiteLevel(whiteLevel)
        }
        let blackValues = Self.values(blackLevels)
        guard blackValues.allSatisfy({
            Self.isFinite($0) && $0 >= 0 && $0 < whiteLevel
        }) else {
            throw .invalidBlackLevels
        }
        guard Self.values(gains).allSatisfy({
            Self.isFinite($0) && $0 > 0
        }) else {
            throw .invalidGains
        }
        guard Self.values(colorMatrix).allSatisfy(Self.isFinite) else {
            throw .invalidColorMatrix
        }
        guard Self.values(letterboxColor).allSatisfy({
            Self.isFinite($0) && $0 >= 0 && $0 <= 1
        }) else {
            throw .invalidLetterboxColor
        }
        let supportedResizePolicy: SupportedResizePolicy
        switch resizePolicy {
        case .scaleFill:
            supportedResizePolicy = .scaleFill
        case .scaleFit:
            supportedResizePolicy = .scaleFit
        case .centerCrop:
            supportedResizePolicy = .centerCrop
        case .regionAffine:
            throw .unsupportedResizePolicy(resizePolicy)
        }
        let normalizationValues =
            Self.values(normalization.scale) +
            Self.values(normalization.bias)
        guard
            normalizationValues.allSatisfy(Self.isFinite)
        else {
            throw .invalidNormalization
        }

        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.sourceBytesPerRow = sourceBytesPerRow
        self.sourceByteCount = sourceByteCount
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        self.resizePolicy = resizePolicy
        self.tensorLayout = tensorLayout
        self.channelOrder = channelOrder
        self.blackLevels = blackLevels
        self.whiteLevel = whiteLevel
        self.gains = gains
        self.colorMatrix = colorMatrix
        self.letterboxColor = letterboxColor
        self.normalization = normalization
        self.appliesSRGBTransfer = appliesSRGBTransfer
        self.supportedResizePolicy = supportedResizePolicy
    }

    private static func isFinite(_ value: Float) -> Bool {
        value.isFinite
    }

    private static func values(
        _ value: RG10BayerValues
    ) -> [Float] {
        [
            value.red,
            value.greenOnRedRow,
            value.greenOnBlueRow,
            value.blue
        ]
    }

    private static func values(_ value: RGBTriplet) -> [Float] {
        [value.red, value.green, value.blue]
    }

    private static func values(
        _ value: VisionModelInputDescriptor.RGBValues
    ) -> [Float] {
        [value.red, value.green, value.blue]
    }

    private static func values(
        _ value: RGBColorMatrix
    ) -> [Float] {
        values(value.row0) + values(value.row1) + values(value.row2)
    }
}
