public enum RG10WordLayout: Sendable, Hashable {
    /// The 10-bit sample occupies bits 0...9 of each little-endian word.
    case leastSignificantBits

    /// The 10-bit sample occupies bits 6...15 of each little-endian word.
    ///
    /// NVIDIA Jetson Xavier and Orin VI expose RAW10 through the T_R16
    /// memory layout. The lower six bits may contain replicated sample bits
    /// and must not be interpreted as additional image data.
    case mostSignificantBits

    public var sampleBitShift: UInt8 {
        switch self {
        case .leastSignificantBits:
            0
        case .mostSignificantBits:
            6
        }
    }
}
