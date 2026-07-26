public enum TensorRTRuntimeStatus: Int32, Sendable, Hashable {
    case available = 0
    case unavailable = 1
    case invalidArgument = 2
    case allocationFailed = 3
    case cudaRuntimeFailure = 4
    case tensorRTRuntimeFailure = 5
    case transferVerificationFailure = 6
    case unknown = -1
}
