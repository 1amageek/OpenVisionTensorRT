import Synchronization

final class TensorRTOutputLeaseState: Sendable {
    private struct State: Sendable {
        var isReleased: Bool
        var borrowCount: Int
        var holderCount: Int
    }

    private let state = Mutex(
        State(
            isReleased: false,
            borrowCount: 0,
            holderCount: 0
        )
    )

    var isReleased: Bool {
        state.withLock { $0.isReleased }
    }

    func withBorrow<Result>(
        _ body: () throws -> Result
    ) throws -> Result {
        let acquired = state.withLock { state in
            guard !state.isReleased else {
                return false
            }
            state.borrowCount += 1
            return true
        }
        guard acquired else {
            throw TensorRTInferenceOutputError.released
        }
        defer {
            state.withLock { state in
                state.borrowCount -= 1
            }
        }
        return try body()
    }

    func release() throws(TensorRTInferenceOutputError) {
        let failure = state.withLock {
            state -> TensorRTInferenceOutputError? in
            guard !state.isReleased else {
                return .released
            }
            guard state.borrowCount == 0 else {
                return .borrowInProgress
            }
            state.isReleased = true
            return nil
        }
        if let failure {
            throw failure
        }
    }

    func retainHolder() {
        state.withLock { state in
            precondition(!state.isReleased)
            state.holderCount += 1
        }
    }

    func releaseHolder() {
        state.withLock { state in
            precondition(state.holderCount > 0)
            state.holderCount -= 1
            if state.holderCount == 0 {
                state.isReleased = true
            }
        }
    }
}
