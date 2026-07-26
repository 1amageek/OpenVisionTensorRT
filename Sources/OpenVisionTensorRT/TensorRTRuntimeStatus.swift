public enum TensorRTRuntimeStatus: Int32, Sendable, Hashable {
    case available = 0
    case unavailable = 1
    case invalidArgument = 2
    case allocationFailed = 3
    case cudaRuntimeFailure = 4
    case tensorRTRuntimeFailure = 5
    case transferVerificationFailure = 6
    case preprocessingFailure = 7
    case nvrtcFailure = 8
    case cudaDriverFailure = 9
    case resourceBusy = 10
    case unknown = -1
}
