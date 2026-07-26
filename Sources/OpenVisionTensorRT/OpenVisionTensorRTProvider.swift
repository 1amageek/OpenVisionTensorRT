import OpenCoreVideo
import OpenVision
import Synchronization

public final class OpenVisionTensorRTProvider:
    VisionProvider,
    Sendable
{
    public static let cudaDevice = VisionComputeDeviceID(
        rawValue: "nvidia.cuda"
    )

    public let descriptor = VisionProviderDescriptor(
        id: VisionProviderID(rawValue: "openvision.tensorrt"),
        revision: "rtmdet-dwpose-1"
    )
    public let capabilities: VisionProviderCapabilities

    private let configuration:
        OpenVisionTensorRTProviderConfiguration
    private let sessionSequence = Mutex<UInt64>(0)

    public init(
        configuration: OpenVisionTensorRTProviderConfiguration
    ) throws(VisionError) {
        self.configuration = configuration
        capabilities = try VisionProviderCapabilities(
            requests: [
                .detectHumanBodyPoseRequest(.revision2)
            ],
            pixelFormats: [
                CVPixelFormatType(
                    rawValue:
                        RG10PreprocessingConfiguration
                        .pixelFormatRawValue
                )
            ],
            memoryDomains: [.host],
            inputOwnershipModes: [.retained],
            transferModes: [
                .stagedHostToDevice(fullFrameCopyCount: 1)
            ],
            computeDevices: [
                .main: [Self.cudaDevice],
                .postProcessing: [Self.cudaDevice]
            ],
            maximumInFlightRequestCount: 1
        )
    }

    public func makeSession(
        configuration requested:
            VisionSessionConfiguration
    ) async throws(VisionError) -> any VisionProviderSession {
        guard requested.model == configuration.model else {
            throw .modelIncompatible(requested.model)
        }
        guard
            requested.transferMode ==
                .stagedHostToDevice(fullFrameCopyCount: 1)
        else {
            throw .unsupportedTransferMode(requested.transferMode)
        }
        for (stage, device) in requested.computeDevices {
            guard capabilities.supports(device, for: stage) else {
                throw .unsupportedComputeDevice(device)
            }
        }
        let sessionID = try nextSessionID()
        let session = try OpenVisionTensorRTProviderSession(
            descriptor: VisionProviderSessionDescriptor(
                id: sessionID,
                provider: descriptor,
                model: configuration.model
            ),
            configuration: configuration
        )
        try await session.prepare()
        return session
    }

    private func nextSessionID()
        throws(VisionError) -> VisionProviderSessionID
    {
        try sessionSequence.withLock { value throws(VisionError) in
            let next = value.addingReportingOverflow(1)
            guard !next.overflow else {
                throw .resourceExhausted(
                    resource: "tensorRTSessionSequence"
                )
            }
            value = next.partialValue
            return VisionProviderSessionID(
                high: 0x4F56_5452_5400_0001,
                low: value
            )
        }
    }

    static func backendFailure(
        operation: String,
        code: Int32
    ) -> VisionError {
        .backend(
            VisionBackendFailure(
                providerID:
                    VisionProviderID(
                        rawValue: "openvision.tensorrt"
                    ),
                operation: operation,
                code: code
            )
        )
    }
}
