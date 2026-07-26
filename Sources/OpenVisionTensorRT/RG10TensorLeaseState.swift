import Synchronization

final class RG10TensorLeaseState: Sendable {
    private struct State: Sendable {
        var isReleased: Bool
        var borrowCount: Int
    }

    private let state = Mutex(
        State(isReleased: false, borrowCount: 0)
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
            throw RG10DeviceTensorError.released
        }
        defer {
            state.withLock { state in
                state.borrowCount -= 1
            }
        }
        return try body()
    }

    func release() throws(RG10DeviceTensorError) {
        let failure = state.withLock {
            state -> RG10DeviceTensorError? in
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

    func releaseForDeinitialization() {
        state.withLock { state in
            guard !state.isReleased else {
                return
            }
            state.isReleased = true
        }
    }
}
