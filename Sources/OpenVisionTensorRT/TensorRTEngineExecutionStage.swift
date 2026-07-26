import CTensorRTShim

public enum TensorRTEngineExecutionStage:
    Int32,
    Sendable,
    Hashable
{
    case none = 0
    case configuration = 1
    case contextCreation = 2
    case streamCreation = 3
    case eventCreation = 4
    case shapeConfiguration = 5
    case outputAllocation = 6
    case tensorBinding = 7
    case enqueue = 8
    case synchronization = 9
    case outputInspection = 10
    case cleanup = 11
    case unknown = -1

    init(_ value: OVTRTEngineExecutionStage) {
        self = Self(rawValue: Int32(value.rawValue)) ?? .unknown
    }
}
