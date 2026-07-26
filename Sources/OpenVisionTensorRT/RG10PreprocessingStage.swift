import CTensorRTShim

public enum RG10PreprocessingStage:
    Int32,
    Sendable,
    Hashable
{
    case none = 0
    case configuration = 1
    case libraryOpen = 2
    case symbolLoad = 3
    case streamCreation = 4
    case eventCreation = 5
    case inputAllocation = 6
    case outputAllocation = 7
    case configurationAllocation = 8
    case configurationTransfer = 9
    case nvrtcProgramCreation = 10
    case nvrtcCompilation = 11
    case ptxAccess = 12
    case driverInitialization = 13
    case moduleLoad = 14
    case kernelLookup = 15
    case hostToDevice = 16
    case kernelLaunch = 17
    case outputEvent = 18
    case outputSynchronization = 19
    case eventTiming = 20
    case outputReadback = 21
    case contentVerification = 22
    case streamSynchronization = 23
    case moduleUnload = 24
    case deviceDeallocation = 25
    case eventDestruction = 26
    case streamDestruction = 27
    case nvrtcProgramDestruction = 28
    case libraryClose = 29
    case unknown = -1

    init(_ stage: OVTRTRG10PreprocessingStage) {
        self = Self(rawValue: Int32(stage.rawValue)) ?? .unknown
    }
}
