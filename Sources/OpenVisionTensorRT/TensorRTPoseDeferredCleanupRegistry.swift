import CTensorRTShim
import Synchronization

final class TensorRTPoseDeferredCleanupRegistry: Sendable {
    private struct Entry: Sendable {
        let owner: TensorRTPosePipelineHandleOwner
        let blockingInputLease: TensorRTPoseInputLease?
    }

    static let shared = TensorRTPoseDeferredCleanupRegistry()

    private let entries = Mutex<[Entry]>([])

    private init() {}

    func enqueue(
        _ owner: TensorRTPosePipelineHandleOwner,
        waitingFor blockingInputLease: TensorRTPoseInputLease? = nil
    ) {
        entries.withLock { entries in
            entries.append(
                Entry(
                    owner: owner,
                    blockingInputLease: blockingInputLease
                )
            )
        }
    }

    func retry() -> TensorRTPoseDeferredCleanupResult {
        let pending = entries.withLock { entries in
            let pending = entries
            entries.removeAll(keepingCapacity: true)
            return pending
        }
        var survivors: [Entry] = []
        survivors.reserveCapacity(pending.count)
        var failures: [TensorRTPoseDeferredCleanupFailure] = []
        failures.reserveCapacity(pending.count)
        var attemptedOwnerCount = 0

        for entry in pending {
            if let lease = entry.blockingInputLease,
               !lease.isReleased
            {
                survivors.append(entry)
                continue
            }
            attemptedOwnerCount += 1
            var rawResult = OVTRTPosePipelineResult()
            let status = entry.owner.destroy(result: &rawResult)
            guard status == .available else {
                survivors.append(entry)
                failures.append(
                    TensorRTPoseDeferredCleanupFailure(
                        status: status,
                        report: TensorRTPosePipelineReport(rawResult)
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
        return TensorRTPoseDeferredCleanupResult(
            attemptedOwnerCount: attemptedOwnerCount,
            remainingOwnerCount: remainingOwnerCount,
            failures: failures
        )
    }
}
