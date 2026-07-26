import CTensorRTShim
import Synchronization

final class RG10PreprocessorHandleOwner: Sendable {
    // The mutex protects only address and lease state. CUDA submission,
    // synchronization, and destruction run after the lock is released.
    // An operation lease prevents destruction from invalidating the opaque
    // pointer until the synchronous C call returns.
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
        var result = OVTRTRG10PreprocessingResult()
        let status = destroy(result: &result)
        guard
            status != .available,
            let handle = detachForDeferredCleanup()
        else {
            return
        }
        RG10DeferredCleanupRegistry.shared.enqueue(
            RG10PreprocessorHandleOwner(handle: handle)
        )
    }

    var isActive: Bool {
        state.withLock { $0.address != nil }
    }

    func withHandle<Result>(
        _ body: (OpaquePointer) -> Result
    ) -> Result? {
        let bits = state.withLock { state -> UInt? in
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
        guard let bits, let handle = OpaquePointer(bitPattern: bits) else {
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
        result: inout OVTRTRG10PreprocessingResult
    ) -> TensorRTRuntimeStatus {
        let bits = state.withLock { state -> UInt? in
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
        guard let bits else {
            return .resourceBusy
        }

        var handle = OpaquePointer(bitPattern: bits)
        let status = TensorRTRuntimeStatus(
            ovtrt_rg10_preprocessor_destroy(
                &handle,
                &result
            )
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

    private func detachForDeferredCleanup() -> OpaquePointer? {
        let bits = state.withLock { state -> UInt? in
            guard
                let address = state.address,
                !state.operationInProgress,
                !state.destructionInProgress
            else {
                return nil
            }
            state.address = nil
            return address
        }
        guard let bits else {
            return nil
        }
        return OpaquePointer(bitPattern: bits)
    }
}
