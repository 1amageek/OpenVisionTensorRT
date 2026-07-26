import CTensorRTShim

public enum CUDATransferStage:
    Int32,
    Sendable,
    Hashable
{
    case none = 0
    case libraryOpen = 1
    case symbolLoad = 2
    case hostAllocation = 3
    case hostRegistration = 4
    case streamCreation = 5
    case eventCreation = 6
    case deviceAllocation = 7
    case hostToDevice = 8
    case eventRecord = 9
    case eventSynchronization = 10
    case eventTiming = 11
    case deviceToHostVerification = 12
    case contentVerification = 13
    case eventDestruction = 14
    case deviceDeallocation = 15
    case streamDestruction = 16
    case hostUnregistration = 17
    case streamSynchronization = 18
    case unknown = -1

    init(_ stage: OVTRTCUDATransferStage) {
        self = CUDATransferStage(
            rawValue: Int32(stage.rawValue)
        ) ?? .unknown
    }
}
