import CTensorRTShim

public actor TensorRTRuntime {
    private let owner: TensorRTRuntimeHandleOwner

    public init() throws(TensorRTRuntimeError) {
        let probe = TensorRTRuntimeProbe.current()
        guard probe.isAvailable else {
            throw .unavailable(probe)
        }

        var handle: OpaquePointer?
        let status = ovtrt_runtime_create(&handle)
        let runtimeStatus = TensorRTRuntimeStatus(status)
        guard
            runtimeStatus == .available,
            let handle
        else {
            throw .creationFailed(runtimeStatus)
        }
        owner = TensorRTRuntimeHandleOwner(handle: handle)
    }

    public var isActive: Bool {
        owner.isActive
    }

    public func shutdown() throws(TensorRTRuntimeError) {
        guard let handle = owner.consume() else {
            throw .alreadyShutDown
        }
        ovtrt_runtime_destroy(handle)
    }
}
