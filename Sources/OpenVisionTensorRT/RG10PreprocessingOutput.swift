public struct RG10PreprocessingOutput: Sendable {
    public let tensor: RG10DeviceTensor
    public let report: RG10PreprocessingReport

    init(
        tensor: RG10DeviceTensor,
        report: RG10PreprocessingReport
    ) {
        self.tensor = tensor
        self.report = report
    }
}
