public enum TensorRTInferenceOutputError:
    Error,
    Sendable,
    Equatable
{
    case released
    case borrowInProgress
}
