public final class TensorRTInferenceOutput: Sendable {
    public let tensors: [TensorRTDeviceOutputTensor]
    public let report: TensorRTEngineExecutionReport

    private let lease: TensorRTOutputLeaseState

    init(
        tensors: [TensorRTDeviceOutputTensor],
        report: TensorRTEngineExecutionReport,
        lease: TensorRTOutputLeaseState
    ) {
        self.tensors = tensors
        self.report = report
        self.lease = lease
        lease.retainHolder()
    }

    deinit {
        lease.releaseHolder()
    }

    public var isReleased: Bool {
        lease.isReleased
    }

    public func release()
        throws(TensorRTInferenceOutputError)
    {
        try lease.release()
    }
}
