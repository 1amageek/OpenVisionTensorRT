import OpenVision
import OpenVisionTensorRT
import Synchronization

@main
enum OpenVisionTensorRTSwiftProbe {
    static func main() async throws {
        #if os(Linux)
        try await run()
        #else
        print("{\"status\":\"unavailable\"}")
        #endif
    }

    #if os(Linux)
    private static func run() async throws {
        let width = 1920
        let height = 1080
        let bytesPerRow = width * 2
        let byteCount = bytesPerRow * height
        let storage = try RG10ProbeStorage(
            byteCount: byteCount,
            alignment: 4096
        )
        try storage.initializeRG10(
            width: width,
            height: height,
            bytesPerRow: bytesPerRow
        )
        let dimensions = try CVPixelDimensions(
            width: width,
            height: height
        )
        let pixelFormat = CVPixelFormatType(
            rawValue:
                RG10PreprocessingConfiguration.pixelFormatRawValue
        )
        let layout = try CVPackedPixelBufferLayout(
            dimensions: dimensions,
            pixelFormat: pixelFormat,
            bytesPerPixel: 2,
            bytesPerRow: bytesPerRow
        )
        let pixelBuffer = PortableRG10PixelBuffer(
            layout: layout,
            storage: storage
        )
        let sample = try CMImageSampleBuffer(
            imageBuffer: pixelBuffer,
            formatDescription: CMImmutableVideoFormatDescription(
                dimensions: dimensions,
                pixelFormat: pixelFormat
            ),
            timing: CMSampleTimingInfo(
                duration: CMTime(value: 1, timescale: 30),
                presentationTimeStamp: .zero,
                decodeTimeStamp: .invalid
            )
        )
        let clock = VisionClockDomain(
            id: "jetson-swift-probe",
            epoch: 1,
            kind: .deviceMonotonic
        )
        let validity = try VisionTimeRange(
            range: CMTimeRange(
                start: .zero,
                duration: CMTime(value: 1, timescale: 1)
            ),
            clockDomain: clock
        )
        let calibrationReference = VisionCalibrationReference(
            id: "jetson-swift-probe",
            revision: 1
        )
        let calibration = try VisionCameraCalibration(
            reference: calibrationReference,
            source: "jetson-swift-probe",
            calibratedAt: try VisionTimestamp(
                time: .zero,
                clockDomain: clock
            ),
            validity: validity,
            intrinsics: VisionCameraIntrinsics(
                matrix: .identity,
                referenceDimensions: dimensions
            )
        )
        let input = try VisionImageInput(
            sampleBuffer: sample,
            frameID: VisionFrameID(
                source: "jetson-swift-probe",
                sequence: 1
            ),
            clockDomain: clock,
            calibration: calibration
        )
        let configuration = try RG10PreprocessingConfiguration(
            sourceWidth: width,
            sourceHeight: height,
            sourceBytesPerRow: bytesPerRow,
            sourceByteCount: byteCount,
            outputWidth: 256,
            outputHeight: 256,
            resizePolicy: .scaleFit,
            tensorLayout: .channelsFirst,
            channelOrder: .rgb,
            blackLevels: RG10BayerValues(
                red: 0,
                greenOnRedRow: 0,
                greenOnBlueRow: 0,
                blue: 0
            ),
            whiteLevel: 1023,
            gains: RG10BayerValues(
                red: 1,
                greenOnRedRow: 1,
                greenOnBlueRow: 1,
                blue: 1
            ),
            colorMatrix: .identity,
            normalization: .zeroToOne,
            appliesSRGBTransfer: false
        )
        let preprocessor = try RG10Preprocessor(
            configuration: configuration
        )
        let output = try await preprocessor.process(input)
        guard
            output.report.satisfiesFrameContract,
            output.tensor.width == 256,
            output.tensor.height == 256,
            output.tensor.channelCount == 3,
            output.tensor.elementCount == 256 * 256 * 3,
            input.isReleased
        else {
            throw SwiftProbeError.contractViolation
        }
        let addressIsNonzero = try output.tensor.withDeviceAddress {
            address,
            byteCount in
            guard address != 0, byteCount == 256 * 256 * 3 * 4 else {
                throw SwiftProbeError.contractViolation
            }
            return true
        }
        try output.tensor.release()
        try await preprocessor.shutdown()
        print(
            "{\"status\":\"available\","
                + "\"swiftPublicPath\":\"passed\","
                + "\"deviceAddressNonzero\":"
                + (addressIsNonzero ? "true" : "false")
                + ",\"inputReleased\":true,"
                + "\"h2dCopies\":"
                + String(
                    output.report.fullFrameHostToDeviceCopyCount
                )
                + ",\"kernelLaunches\":"
                + String(output.report.kernelLaunchCount)
                + "}"
        )
    }
    #endif
}

private enum SwiftProbeError: Error {
    case allocationFailed
    case storageReleased
    case contractViolation
}

private final class RG10ProbeStorage: Sendable {
    // This object is the exactly-once allocation owner. State stores the
    // address as an integer so no pointer crosses a Sendable boundary.
    // Pointer reconstruction is scoped to the borrow closures, which validate
    // byteCount before constructing a Span. Deinitialization clears the
    // address under the same Mutex before deallocating it once.
    private struct State: Sendable {
        var address: UInt?
    }

    let byteCount: Int
    private let state: Mutex<State>

    init(
        byteCount: Int,
        alignment: Int
    ) throws(SwiftProbeError) {
        guard byteCount > 0, alignment > 0 else {
            throw .allocationFailed
        }
        let pointer = UnsafeMutableRawPointer.allocate(
            byteCount: byteCount,
            alignment: alignment
        )
        self.byteCount = byteCount
        state = Mutex(State(address: UInt(bitPattern: pointer)))
    }

    deinit {
        let address = state.withLock { state -> UInt? in
            let address = state.address
            state.address = nil
            return address
        }
        if let address,
           let pointer = UnsafeMutableRawPointer(bitPattern: address)
        {
            pointer.deallocate()
        }
    }

    func initializeRG10(
        width: Int,
        height: Int,
        bytesPerRow: Int
    ) throws(SwiftProbeError) {
        try withMutableBytes { bytes in
            for y in 0..<height {
                for x in 0..<width {
                    let value = UInt16(
                        (x * 17 + y * 31 + (x ^ y)) & 1023
                    )
                    let offset = y * bytesPerRow + x * 2
                    bytes[offset] = UInt8(truncatingIfNeeded: value)
                    bytes[offset + 1] = UInt8(
                        truncatingIfNeeded: value >> 8
                    )
                }
            }
        }
    }

    func withReadBytes(
        _ body: (borrowing Span<UInt8>) -> Void
    ) throws(SwiftProbeError) {
        let address = state.withLock { $0.address }
        guard
            let address,
            let pointer = UnsafeRawPointer(bitPattern: address)
        else {
            throw .storageReleased
        }
        body(
            Span(
                _unsafeStart:
                    pointer.assumingMemoryBound(to: UInt8.self),
                count: byteCount
            )
        )
    }

    private func withMutableBytes(
        _ body: (inout MutableSpan<UInt8>) -> Void
    ) throws(SwiftProbeError) {
        let address = state.withLock { $0.address }
        guard
            let address,
            let pointer =
                UnsafeMutableRawPointer(bitPattern: address)
        else {
            throw .storageReleased
        }
        var bytes = MutableSpan(
            _unsafeStart:
                pointer.assumingMemoryBound(to: UInt8.self),
            count: byteCount
        )
        body(&bytes)
    }
}

private final class PortableRG10PixelBuffer: CVPixelBuffer {
    let layout: CVPackedPixelBufferLayout
    let accessCapabilities: CVPixelBufferAccessCapabilities = [.read]
    let attachments = CVBufferAttachments()

    private let storage: RG10ProbeStorage

    var dimensions: CVPixelDimensions {
        layout.dimensions
    }

    var pixelFormat: CVPixelFormatType {
        layout.pixelFormat
    }

    var bytesPerRow: Int {
        layout.bytesPerRow
    }

    var byteCount: Int {
        storage.byteCount
    }

    init(
        layout: CVPackedPixelBufferLayout,
        storage: RG10ProbeStorage
    ) {
        self.layout = layout
        self.storage = storage
    }

    func withReadBytes(
        _ body: (borrowing Span<UInt8>) -> Void
    ) throws(CVPixelBufferError) {
        do {
            try storage.withReadBytes(body)
        } catch {
            throw .storageReleased
        }
    }

    func withWriteBytes(
        _ body: (inout MutableSpan<UInt8>) -> Void
    ) throws(CVPixelBufferError) {
        throw .unsupportedAccess(.write)
    }
}
