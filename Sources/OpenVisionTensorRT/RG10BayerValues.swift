public struct RG10BayerValues: Sendable, Hashable {
    public let red: Float
    public let greenOnRedRow: Float
    public let greenOnBlueRow: Float
    public let blue: Float

    public init(
        red: Float,
        greenOnRedRow: Float,
        greenOnBlueRow: Float,
        blue: Float
    ) {
        self.red = red
        self.greenOnRedRow = greenOnRedRow
        self.greenOnBlueRow = greenOnBlueRow
        self.blue = blue
    }
}
