public struct RG10LibraryOpenFailures:
    OptionSet,
    Sendable,
    Hashable
{
    public let rawValue: UInt8

    public static let cudaRuntime = Self(rawValue: 1 << 0)
    public static let cudaDriver = Self(rawValue: 1 << 1)
    public static let nvrtc = Self(rawValue: 1 << 2)

    public init(rawValue: UInt8) {
        self.rawValue = rawValue
    }
}
