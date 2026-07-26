public struct TensorRTPoseDeferredCleanupResult:
    Sendable,
    Equatable
{
    public let attemptedOwnerCount: Int
    public let remainingOwnerCount: Int
    public let failures: [TensorRTPoseDeferredCleanupFailure]

    init(
        attemptedOwnerCount: Int,
        remainingOwnerCount: Int,
        failures: [TensorRTPoseDeferredCleanupFailure]
    ) {
        self.attemptedOwnerCount = attemptedOwnerCount
        self.remainingOwnerCount = remainingOwnerCount
        self.failures = failures
    }
}
