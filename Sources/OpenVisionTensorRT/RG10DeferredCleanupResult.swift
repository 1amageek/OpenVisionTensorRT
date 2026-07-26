public struct RG10DeferredCleanupResult:
    Sendable,
    Equatable
{
    public let attemptedOwnerCount: Int
    public let remainingOwnerCount: Int
    public let failures: [RG10DeferredCleanupFailure]

    init(
        attemptedOwnerCount: Int,
        remainingOwnerCount: Int,
        failures: [RG10DeferredCleanupFailure]
    ) {
        self.attemptedOwnerCount = attemptedOwnerCount
        self.remainingOwnerCount = remainingOwnerCount
        self.failures = failures
    }
}
