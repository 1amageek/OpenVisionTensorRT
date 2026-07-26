public struct RGBColorMatrix: Sendable, Hashable {
    public let row0: RGBTriplet
    public let row1: RGBTriplet
    public let row2: RGBTriplet

    public init(
        row0: RGBTriplet,
        row1: RGBTriplet,
        row2: RGBTriplet
    ) {
        self.row0 = row0
        self.row1 = row1
        self.row2 = row2
    }

    public static let identity = Self(
        row0: RGBTriplet(red: 1, green: 0, blue: 0),
        row1: RGBTriplet(red: 0, green: 1, blue: 0),
        row2: RGBTriplet(red: 0, green: 0, blue: 1)
    )
}
