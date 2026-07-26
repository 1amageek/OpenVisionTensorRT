import OpenVision

/// A scoped view over one engine-owned CUDA output allocation.
///
/// The address remains valid until the parent `TensorRTInferenceOutput` is
/// released. The address cannot escape an active borrow. A downstream CUDA
/// operation may outlive the borrow only when the parent output remains
/// retained and is released after that operation's completion fence.
public final class TensorRTDeviceOutputTensor: Sendable {
    public let name: String
    public let byteCount: Int
    public let elementCount: Int
    public let shape: [Int]
    public let elementType: VisionModelInputDescriptor.ElementType

    private let address: UInt
    private let owner: TensorRTEngineHandleOwner
    private let lease: TensorRTOutputLeaseState

    init(
        name: String,
        address: UInt,
        byteCount: Int,
        elementCount: Int,
        shape: [Int],
        elementType: VisionModelInputDescriptor.ElementType,
        owner: TensorRTEngineHandleOwner,
        lease: TensorRTOutputLeaseState
    ) {
        self.name = name
        self.address = address
        self.byteCount = byteCount
        self.elementCount = elementCount
        self.shape = shape
        self.elementType = elementType
        self.owner = owner
        self.lease = lease
        lease.retainHolder()
    }

    deinit {
        lease.releaseHolder()
    }

    public func withDeviceAddress<Result>(
        _ body: (UInt, Int) throws -> Result
    ) throws -> Result {
        _ = owner
        return try lease.withBorrow {
            try body(address, byteCount)
        }
    }
}

extension TensorRTDeviceOutputTensor: TensorRTDeviceInput {
    public func withTensorRTDeviceAddress(
        _ body: (UInt, Int) -> Void
    ) throws(TensorRTDeviceInputError) {
        do {
            try withDeviceAddress(body)
        } catch TensorRTInferenceOutputError.released {
            throw .released
        } catch {
            throw .inaccessible
        }
    }
}
