public struct TensorRTPosePreparation: Sendable {
    public let input: TensorRTPoseInput?
    public let regions: [TensorRTPoseRegion]
    public let report: TensorRTPosePipelineReport

    public var selectedRegionCount: Int {
        regions.count
    }
}
