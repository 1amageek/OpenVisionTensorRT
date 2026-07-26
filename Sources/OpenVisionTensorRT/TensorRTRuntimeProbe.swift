import CTensorRTShim

public struct TensorRTRuntimeProbe: Sendable, Hashable {
    public let status: TensorRTRuntimeStatus
    public let tensorRTVersion: Int32
    public let cudaRuntimeVersion: Int32
    public let cudaDriverVersion: Int32
    public let cudaDeviceCount: Int32

    public var isAvailable: Bool {
        status == .available && cudaDeviceCount > 0
    }

    public static func current() -> TensorRTRuntimeProbe {
        var result = OVTRTProbeResult()
        let status = ovtrt_probe(&result)
        return TensorRTRuntimeProbe(
            status: TensorRTRuntimeStatus(status),
            tensorRTVersion: result.tensorRTVersion,
            cudaRuntimeVersion: result.cudaRuntimeVersion,
            cudaDriverVersion: result.cudaDriverVersion,
            cudaDeviceCount: result.cudaDeviceCount
        )
    }

    init(
        status: TensorRTRuntimeStatus,
        tensorRTVersion: Int32,
        cudaRuntimeVersion: Int32,
        cudaDriverVersion: Int32,
        cudaDeviceCount: Int32
    ) {
        self.status = status
        self.tensorRTVersion = tensorRTVersion
        self.cudaRuntimeVersion = cudaRuntimeVersion
        self.cudaDriverVersion = cudaDriverVersion
        self.cudaDeviceCount = cudaDeviceCount
    }
}

extension TensorRTRuntimeStatus {
    init(_ status: OVTRTStatus) {
        self = TensorRTRuntimeStatus(
            rawValue: Int32(status.rawValue)
        ) ?? .unknown
    }
}
