public protocol TensorRTDeviceInput: Sendable {
    var byteCount: Int { get }

    /// Borrows the CUDA device address for one synchronous operation.
    ///
    /// The address must not escape `body`. Work enqueued from `body` must
    /// complete before the input lease is released.
    func withTensorRTDeviceAddress(
        _ body: (UInt, Int) -> Void
    ) throws(TensorRTDeviceInputError)
}
