import OpenVision

public enum TensorRTEngineCompatibilityError:
    Error,
    Sendable,
    Equatable
{
    case tensorRTVersion(expected: Int32, actual: Int32)
    case cudaRuntimeVersion(expected: Int32, actual: Int32)
    case computeCapability(
        expectedMajor: Int32,
        expectedMinor: Int32,
        actualMajor: Int32,
        actualMinor: Int32
    )
    case checksumNotVerified
    case tensorCount(expected: Int, actual: Int)
    case missingTensor(String)
    case unexpectedTensor(String)
    case ioMode(
        tensor: String,
        expected: TensorRTEngineTensorDescriptor.IOMode,
        actual: TensorRTEngineTensorDescriptor.IOMode
    )
    case elementType(
        tensor: String,
        expected: VisionModelInputDescriptor.ElementType,
        actual: VisionModelInputDescriptor.ElementType
    )
    case shape(
        tensor: String,
        axis: Int,
        expected: VisionModelTensorDescriptor.Dimension,
        actual: Int
    )
    case missingProfile(String)
    case invalidProfile(
        tensor: String,
        axis: Int,
        minimum: Int,
        optimum: Int,
        maximum: Int,
        expectedMaximum: Int
    )
}
