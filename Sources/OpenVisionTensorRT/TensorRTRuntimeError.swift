public enum TensorRTRuntimeError: Error, Sendable, Equatable {
    case unavailable(TensorRTRuntimeProbe)
    case creationFailed(TensorRTRuntimeStatus)
    case alreadyShutDown
}
