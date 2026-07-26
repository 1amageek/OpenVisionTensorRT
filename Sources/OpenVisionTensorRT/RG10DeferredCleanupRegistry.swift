import CTensorRTShim
import Synchronization

final class RG10DeferredCleanupRegistry: Sendable {
    private struct Entry: Sendable {
        let owner: RG10PreprocessorHandleOwner
        let blockingTensorLease: RG10TensorLeaseState?
    }

    static let shared = RG10DeferredCleanupRegistry()

    private let entries = Mutex<[Entry]>([])

    private init() {}

    func enqueue(
        _ owner: RG10PreprocessorHandleOwner,
        waitingFor blockingTensorLease: RG10TensorLeaseState? = nil
    ) {
        entries.withLock { entries in
            entries.append(
                Entry(
                    owner: owner,
                    blockingTensorLease: blockingTensorLease
                )
            )
        }
    }

    func retry() -> RG10DeferredCleanupResult {
        let pending = entries.withLock { entries in
            let pending = entries
            entries.removeAll(keepingCapacity: true)
            return pending
        }
        var survivors: [Entry] = []
        survivors.reserveCapacity(pending.count)
        var failures: [RG10DeferredCleanupFailure] = []
        failures.reserveCapacity(pending.count)
        var attemptedOwnerCount = 0

        for entry in pending {
            if let lease = entry.blockingTensorLease,
               !lease.isReleased
            {
                survivors.append(entry)
                continue
            }
            attemptedOwnerCount += 1
            var rawResult = OVTRTRG10PreprocessingResult()
            let status = entry.owner.destroy(result: &rawResult)
            guard status == .available else {
                survivors.append(entry)
                failures.append(
                    RG10DeferredCleanupFailure(
                        status: status,
                        report: RG10PreprocessingReport(rawResult)
                    )
                )
                continue
            }
        }
        let remainingOwnerCount = entries.withLock { entries in
            if !survivors.isEmpty {
                entries.append(contentsOf: survivors)
            }
            return entries.count
        }
        return RG10DeferredCleanupResult(
            attemptedOwnerCount: attemptedOwnerCount,
            remainingOwnerCount: remainingOwnerCount,
            failures: failures
        )
    }
}
