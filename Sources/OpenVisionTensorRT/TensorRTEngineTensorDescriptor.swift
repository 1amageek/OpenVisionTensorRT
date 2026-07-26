import OpenVision

public struct TensorRTEngineTensorDescriptor:
    Sendable,
    Hashable
{
    public enum IOMode: Sendable, Hashable {
        case input
        case output
    }

    public struct Profile: Sendable, Hashable {
        public let minimum: [Int]
        public let optimum: [Int]
        public let maximum: [Int]

        public init(
            minimum: [Int],
            optimum: [Int],
            maximum: [Int]
        ) {
            self.minimum = minimum
            self.optimum = optimum
            self.maximum = maximum
        }
    }

    public let name: String
    public let ioMode: IOMode
    public let elementType:
        VisionModelInputDescriptor.ElementType
    public let declaredShape: [Int]
    public let profile: Profile?

    public init(
        name: String,
        ioMode: IOMode,
        elementType: VisionModelInputDescriptor.ElementType,
        declaredShape: [Int],
        profile: Profile?
    ) {
        self.name = name
        self.ioMode = ioMode
        self.elementType = elementType
        self.declaredShape = declaredShape
        self.profile = profile
    }
}
