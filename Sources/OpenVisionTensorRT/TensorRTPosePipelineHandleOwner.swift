import CTensorRTShim
import Synchronization

final class TensorRTPosePipelineHandleOwner: Sendable {
    private struct State: Sendable {
        var address: UInt?
        var operationInProgress: Bool
        var destructionInProgress: Bool
    }

    private let state: Mutex<State>

    init(handle: OpaquePointer) {
        state = Mutex(
            State(
                address: UInt(bitPattern: handle),
                operationInProgress: false,
                destructionInProgress: false
            )
        )
    }

    deinit {
        var result = OVTRTPosePipelineResult()
        _ = destroy(result: &result)
    }

    var isActive: Bool {
        state.withLock { $0.address != nil }
    }

    func withHandle<Result>(
        _ body: (OpaquePointer) -> Result
    ) -> Result? {
        let address = state.withLock { state -> UInt? in
            guard
                let address = state.address,
                !state.operationInProgress,
                !state.destructionInProgress
            else {
                return nil
            }
            state.operationInProgress = true
            return address
        }
        guard
            let address,
            let handle = OpaquePointer(bitPattern: address)
        else {
            return nil
        }
        defer {
            state.withLock { state in
                state.operationInProgress = false
            }
        }
        return body(handle)
    }

    func destroy(
        result: inout OVTRTPosePipelineResult
    ) -> TensorRTRuntimeStatus {
        let address = state.withLock { state -> UInt? in
            guard
                let address = state.address,
                !state.operationInProgress,
                !state.destructionInProgress
            else {
                return nil
            }
            state.destructionInProgress = true
            return address
        }
        guard let address else {
            return .resourceBusy
        }
        var handle = OpaquePointer(bitPattern: address)
        let status = TensorRTRuntimeStatus(
            ovtrt_pose_pipeline_destroy(&handle, &result)
        )
        state.withLock { state in
            state.destructionInProgress = false
            if let handle {
                state.address = UInt(bitPattern: handle)
            } else {
                state.address = nil
            }
        }
        return status
    }
}
