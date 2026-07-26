import Synchronization

final class TensorRTPoseInputLease: Sendable {
    private struct State: Sendable {
        var isReleased = false
        var borrowCount = 0
    }

    private let state = Mutex(State())

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
            throw TensorRTPosePipelineError.inputReleased
        }
        defer {
            state.withLock { state in
                state.borrowCount -= 1
            }
        }
        return try body()
    }

    func release() throws(TensorRTPosePipelineError) {
        let failure = state.withLock {
            state -> TensorRTPosePipelineError? in
            guard !state.isReleased else {
                return .inputReleased
            }
            guard state.borrowCount == 0 else {
                return .outputInUse
            }
            state.isReleased = true
            return nil
        }
        if let failure {
            throw failure
        }
    }

    func releaseForDeinitialization() {
        state.withLock { state in
            if state.borrowCount == 0 {
                state.isReleased = true
            }
        }
    }
}
