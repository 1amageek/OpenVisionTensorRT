public enum RG10InputMismatch: Sendable, Equatable {
    case pixelFormat(actual: UInt32)
    case dimensions(
        actualWidth: Int,
        actualHeight: Int,
        expectedWidth: Int,
        expectedHeight: Int
    )
    case planarStorage
    case bytesPerRow(actual: Int, expected: Int)
    case byteCount(actual: Int, expected: Int)
    case transferMode
}
