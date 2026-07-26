import CTensorRTShim

public enum TensorRTEngineLoadStage:
    Int32,
    Sendable,
    Hashable
{
    case none = 0
    case configuration = 1
    case libraryOpen = 2
    case symbolLoad = 3
    case fileOpen = 4
    case fileStat = 5
    case fileMapping = 6
    case checksum = 7
    case runtimeCreation = 8
    case deserialization = 9
    case tensorInspection = 10
    case unknown = -1

    init(_ stage: OVTRTEngineLoadStage) {
        self = Self(rawValue: Int32(stage.rawValue)) ?? .unknown
    }
}
