import CTensorRTShim
import OpenVision

/// A lease over the provider-owned CUDA tensor produced by RG10 preprocessing.
///
/// The device address remains valid until `release()` succeeds. A device
/// operation enqueued inside `withDeviceAddress(_:)` may outlive the closure
/// only while this tensor remains retained, and `release()` may be called only
/// after the operation's completion fence has passed. The address must not be
/// stored or returned as an independently owned pointer. Releasing this tensor
/// allows its preprocessor to overwrite the reusable output allocation.
public final class RG10DeviceTensor:
    Sendable,
    TensorRTDeviceInput
{
    public let byteCount: Int
    public let elementCount: Int
    public let width: Int
    public let height: Int
    public let channelCount: Int
    public let layout: VisionTensorLayout
    public let channelOrder: VisionTensorChannelOrder

    private let address: UInt
    private let sourceAddress: UInt
    private let sourceByteCount: UInt64
    private let sourceWidth: UInt32
    private let sourceHeight: UInt32
    private let sourceBytesPerRow: UInt32
    private let owner: RG10PreprocessorHandleOwner
    private let lease: RG10TensorLeaseState

    init(
        descriptor: OVTRTDeviceTensorView,
        sourceDescriptor: OVTRTRG10SourceView,
        owner: RG10PreprocessorHandleOwner,
        lease: RG10TensorLeaseState
    ) throws(RG10PreprocessorError) {
        guard
            let deviceAddress = descriptor.deviceAddress,
            let byteCount = Int(exactly: descriptor.byteCount),
            let elementCount = Int(exactly: descriptor.elementCount),
            descriptor.width > 0,
            descriptor.height > 0,
            descriptor.channelCount > 0
        else {
            throw .invalidTensorDescriptor
        }
        guard
            let sourceDeviceAddress =
                sourceDescriptor.deviceAddress,
            sourceDescriptor.byteCount > 0,
            sourceDescriptor.width > 0,
            sourceDescriptor.height > 0,
            sourceDescriptor.bytesPerRow > 0
        else {
            throw .invalidTensorDescriptor
        }
        address = UInt(bitPattern: deviceAddress)
        sourceAddress = UInt(bitPattern: sourceDeviceAddress)
        sourceByteCount = sourceDescriptor.byteCount
        sourceWidth = sourceDescriptor.width
        sourceHeight = sourceDescriptor.height
        sourceBytesPerRow = sourceDescriptor.bytesPerRow
        self.byteCount = byteCount
        self.elementCount = elementCount
        width = Int(descriptor.width)
        height = Int(descriptor.height)
        channelCount = Int(descriptor.channelCount)
        layout =
            descriptor.layout == OVTRTTensorLayoutNCHW
            ? .channelsFirst
            : .channelsLast
        channelOrder =
            descriptor.channelOrder == OVTRTTensorChannelOrderRGB
            ? .rgb
            : .bgr
        self.owner = owner
        self.lease = lease
    }

    deinit {
        lease.releaseForDeinitialization()
    }

    public var isReleased: Bool {
        lease.isReleased
    }

    public func withDeviceAddress<Result>(
        _ body: (UInt, Int) throws -> Result
    ) throws -> Result {
        _ = owner
        return try lease.withBorrow {
            try body(address, byteCount)
        }
    }

    public func release() throws(RG10DeviceTensorError) {
        try lease.release()
    }

    public func withTensorRTDeviceAddress(
        _ body: (UInt, Int) -> Void
    ) throws(TensorRTDeviceInputError) {
        do {
            try withDeviceAddress(body)
        } catch RG10DeviceTensorError.released {
            throw .released
        } catch {
            throw .inaccessible
        }
    }

    func withRG10SourceDeviceView<Result>(
        _ body: (OVTRTRG10SourceView) throws -> Result
    ) throws -> Result {
        _ = owner
        return try lease.withBorrow {
            try body(
                OVTRTRG10SourceView(
                    deviceAddress:
                        UnsafeRawPointer(bitPattern: sourceAddress),
                    byteCount: sourceByteCount,
                    width: sourceWidth,
                    height: sourceHeight,
                    bytesPerRow: sourceBytesPerRow
                )
            )
        }
    }
}
