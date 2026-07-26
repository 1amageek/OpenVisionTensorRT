import CTensorRTShim

public struct TensorRTEngineExecutionReport:
    Sendable,
    Hashable
{
    public let failureStage: TensorRTEngineExecutionStage
    public let outputTensorCount: Int
    public let persistentDeviceAllocationCount: Int
    /// Device allocations explicitly issued by this package per submission.
    ///
    /// TensorRT-owned workspace allocations are not observable through this
    /// counter and are prepared by the static execution context.
    public let explicitFrameDeviceAllocationCount: Int
    public let batchSize: Int
    public let persistentDeviceAllocationByteCount: UInt64
    public let inputByteCount: UInt64
    public let outputByteCount: UInt64
    public let submissionCount: UInt64
    public let inferenceMilliseconds: Float

    init(_ value: OVTRTEngineExecutionResult) {
        failureStage =
            TensorRTEngineExecutionStage(value.failureStage)
        outputTensorCount = Int(value.outputTensorCount)
        persistentDeviceAllocationCount =
            Int(value.persistentDeviceAllocationCount)
        explicitFrameDeviceAllocationCount =
            Int(value.frameDeviceAllocationCount)
        batchSize = Int(value.batchSize)
        persistentDeviceAllocationByteCount =
            value.persistentDeviceAllocationByteCount
        inputByteCount = value.inputByteCount
        outputByteCount = value.outputByteCount
        submissionCount = value.submissionCount
        inferenceMilliseconds = value.inferenceMilliseconds
    }
}
