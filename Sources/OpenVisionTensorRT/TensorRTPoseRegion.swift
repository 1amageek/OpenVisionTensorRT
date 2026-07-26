import CTensorRTShim

public struct TensorRTPoseRegion: Sendable, Hashable {
    public let centerX: Float
    public let centerY: Float
    public let width: Float
    public let height: Float
    public let confidence: Float

    init(_ region: OVTRTPoseRegion) {
        centerX = region.centerX
        centerY = region.centerY
        width = region.width
        height = region.height
        confidence = region.confidence
    }

    init(
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float,
        confidence: Float
    ) {
        self.centerX = centerX
        self.centerY = centerY
        self.width = width
        self.height = height
        self.confidence = confidence
    }
}
