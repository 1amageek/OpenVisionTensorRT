import CTensorRTShim

public enum CUDATransferProbe {
    public static func run(
        configuration: CUDATransferProbeConfiguration
    ) throws(CUDATransferProbeError) -> CUDATransferProbeResult {
        var rawConfiguration =
            OVTRTCUDATransferProbeConfiguration(
                byteCount: configuration.byteCount,
                warmupIterationCount:
                    configuration.warmupIterationCount,
                measuredIterationCount:
                    configuration.measuredIterationCount
            )
        var rawResult = OVTRTCUDATransferProbeResult()
        let rawStatus = ovtrt_cuda_transfer_probe(
            &rawConfiguration,
            &rawResult
        )
        let status = TensorRTRuntimeStatus(rawStatus)
        let result = CUDATransferProbeResult(rawResult)

        switch status {
        case .available:
            guard result.isTransferContractSatisfied else {
                throw .failed(
                    status: .transferVerificationFailure,
                    result: result
                )
            }
            return result
        case .unavailable:
            throw .unavailable(result)
        default:
            throw .failed(status: status, result: result)
        }
    }
}
