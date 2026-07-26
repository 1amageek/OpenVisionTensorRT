public enum RG10PreprocessingConfigurationError:
    Error,
    Sendable,
    Equatable
{
    case invalidSourceDimensions(width: Int, height: Int)
    case invalidSourceBytesPerRow(Int)
    case invalidSourceByteCount(Int)
    case invalidOutputDimensions(width: Int, height: Int)
    case invalidBlackLevels
    case invalidWhiteLevel(Float)
    case invalidGains
    case invalidColorMatrix
    case invalidLetterboxColor
    case invalidNormalization
}
