public enum TensorRTDeviceInputError:
    Error,
    Sendable,
    Equatable
{
    case released
    case inaccessible
    case invalidByteCount(Int)
    case byteCountMismatch(declared: Int, borrowed: Int)
}
