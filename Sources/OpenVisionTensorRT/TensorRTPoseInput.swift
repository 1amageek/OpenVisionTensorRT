/// A completion-fenced lease over the reusable DWPose CUDA input tensor.
///
/// The device address is borrowed only for a synchronous TensorRT submission.
/// TensorRT must finish reading it before `release()` permits the pipeline to
/// overwrite the allocation for another frame.
public final class TensorRTPoseInput:
    Sendable,
    TensorRTDeviceInput
{
    public let byteCount: Int
    public let batchSize: Int
    public let width: Int
    public let height: Int

    private let address: UInt
    private let owner: TensorRTPosePipelineHandleOwner
    private let lease: TensorRTPoseInputLease

    init(
        address: UInt,
        byteCount: Int,
        batchSize: Int,
        width: Int,
        height: Int,
        owner: TensorRTPosePipelineHandleOwner,
        lease: TensorRTPoseInputLease
    ) {
        self.address = address
        self.byteCount = byteCount
        self.batchSize = batchSize
        self.width = width
        self.height = height
        self.owner = owner
        self.lease = lease
    }

    deinit {
        lease.releaseForDeinitialization()
    }

    public var isReleased: Bool {
        lease.isReleased
    }

    public func release() throws(TensorRTPosePipelineError) {
        try lease.release()
    }

    public func withTensorRTDeviceAddress(
        _ body: (UInt, Int) -> Void
    ) throws(TensorRTDeviceInputError) {
        _ = owner
        do {
            try lease.withBorrow {
                body(address, byteCount)
            }
        } catch TensorRTPosePipelineError.inputReleased {
            throw .released
        } catch {
            throw .inaccessible
        }
    }
}
