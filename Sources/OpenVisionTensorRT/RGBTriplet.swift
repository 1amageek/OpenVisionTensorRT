public struct RGBTriplet: Sendable, Hashable {
    public let red: Float
    public let green: Float
    public let blue: Float

    public init(red: Float, green: Float, blue: Float) {
        self.red = red
        self.green = green
        self.blue = blue
    }

    public static let zero = Self(red: 0, green: 0, blue: 0)
}
