public enum TensorRTPosePipelineStage:
    Int32,
    Sendable,
    Hashable
{
    case none = 0
    case configuration = 1
    case libraryOpen = 2
    case symbolLoad = 3
    case streamCreation = 4
    case deviceAllocation = 5
    case configurationTransfer = 6
    case nvrtcProgramCreation = 7
    case nvrtcCompilation = 8
    case ptxAccess = 9
    case driverInitialization = 10
    case moduleLoad = 11
    case kernelLookup = 12
    case regionSelection = 13
    case regionReadback = 14
    case regionAffine = 15
    case poseSynchronization = 16
    case simCCDecode = 17
    case jointReadback = 18
    case streamSynchronization = 19
    case moduleUnload = 20
    case deviceDeallocation = 21
    case streamDestruction = 22
    case libraryClose = 23
    case unknown = -1
}
