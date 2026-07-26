import CTensorRTShim
import OpenVision

public actor TensorRTEngine {
    public let artifact: TensorRTStageEngineArtifactDescriptor
    public let loadReport: TensorRTEngineLoadReport
    public let tensors: [TensorRTEngineTensorDescriptor]

    private let owner: TensorRTEngineHandleOwner

    public init(
        path: String,
        artifact: TensorRTStageEngineArtifactDescriptor
    ) throws(TensorRTEngineError) {
        var handle: OpaquePointer?
        var rawReport = OVTRTEngineLoadResult()
        let rawStatus = path.withCString { pathCString in
            artifact.artifact.checksum.withCString {
                checksumCString in
                ovtrt_engine_create(
                    pathCString,
                    checksumCString,
                    &handle,
                    &rawReport
                )
            }
        }
        let status = TensorRTRuntimeStatus(rawStatus)
        let report = TensorRTEngineLoadReport(rawReport)
        guard status == .available, let handle else {
            if let handle {
                ovtrt_engine_destroy(handle)
            }
            if status == .unavailable {
                throw .unavailable(report)
            }
            throw .loadingFailed(status: status, report: report)
        }

        do {
            let inspected = try Self.inspectedTensors(
                handle: handle,
                count: report.ioTensorCount
            )
            try Self.validateRuntime(
                report: report,
                artifact: artifact.artifact
            )
            try Self.validateTensors(
                inspected,
                artifact: artifact
            )
            self.artifact = artifact
            loadReport = report
            tensors = inspected
            owner = TensorRTEngineHandleOwner(handle: handle)
        } catch {
            ovtrt_engine_destroy(handle)
            throw error
        }
    }

    public var isActive: Bool {
        owner.isActive
    }

    public func shutdown() throws(TensorRTEngineError) {
        guard let handle = owner.consume() else {
            throw .alreadyShutDown
        }
        ovtrt_engine_destroy(handle)
    }

    private static func inspectedTensors(
        handle: OpaquePointer,
        count: Int
    ) throws(TensorRTEngineError)
        -> [TensorRTEngineTensorDescriptor]
    {
        guard count > 0 else {
            throw .inspectionFailed(
                status: .engineArtifactFailure,
                tensorIndex: 0
            )
        }
        var tensors: [TensorRTEngineTensorDescriptor] = []
        tensors.reserveCapacity(count)
        for index in 0 ..< count {
            tensors.append(
                try inspectedTensor(
                    handle: handle,
                    index: index
                )
            )
        }
        return tensors
    }

    private static func inspectedTensor(
        handle: OpaquePointer,
        index: Int
    ) throws(TensorRTEngineError)
        -> TensorRTEngineTensorDescriptor
    {
        let rawIndex = UInt32(index)
        var requiredCapacity: UInt32 = 0
        var status = TensorRTRuntimeStatus(
            ovtrt_engine_tensor_name(
                handle,
                rawIndex,
                nil,
                0,
                &requiredCapacity
            )
        )
        guard
            status == .available,
            requiredCapacity > 1,
            let capacity = Int(exactly: requiredCapacity)
        else {
            throw .inspectionFailed(
                status: status,
                tensorIndex: index
            )
        }
        var nameBytes = [CChar](
            repeating: 0,
            count: capacity
        )
        status = nameBytes.withUnsafeMutableBufferPointer {
            buffer in
            TensorRTRuntimeStatus(
                ovtrt_engine_tensor_name(
                    handle,
                    rawIndex,
                    buffer.baseAddress,
                    requiredCapacity,
                    &requiredCapacity
                )
            )
        }
        guard status == .available else {
            throw .inspectionFailed(
                status: status,
                tensorIndex: index
            )
        }
        guard let name = nameBytes.withUnsafeBufferPointer({ buffer in
            buffer.baseAddress.map { String(cString: $0) }
        }) else {
            throw .inspectionFailed(
                status: .engineArtifactFailure,
                tensorIndex: index
            )
        }

        var rawInfo = OVTRTEngineTensorInfo()
        status = TensorRTRuntimeStatus(
            ovtrt_engine_tensor_info(
                handle,
                rawIndex,
                &rawInfo
            )
        )
        guard
            status == .available,
            rawInfo.rank > 0,
            let rank = Int(exactly: rawInfo.rank)
        else {
            throw .inspectionFailed(
                status: status,
                tensorIndex: index
            )
        }
        let ioMode: TensorRTEngineTensorDescriptor.IOMode
        if rawInfo.ioMode == OVTRTTensorIOModeInput {
            ioMode = .input
        } else if rawInfo.ioMode == OVTRTTensorIOModeOutput {
            ioMode = .output
        } else {
            throw .inspectionFailed(
                status: .engineArtifactFailure,
                tensorIndex: index
            )
        }
        guard
            let elementType = elementType(rawInfo.elementType)
        else {
            throw .unsupportedTensorElementType(
                tensorIndex: index
            )
        }
        let declaredShape = try shape(
            handle: handle,
            tensorIndex: index,
            rank: rank,
            selector: OVTRTShapeSelectorDeclared
        )
        let profile: TensorRTEngineTensorDescriptor.Profile?
        if ioMode == .input, declaredShape.contains(-1) {
            profile = try .init(
                minimum: shape(
                    handle: handle,
                    tensorIndex: index,
                    rank: rank,
                    selector: OVTRTShapeSelectorMinimum
                ),
                optimum: shape(
                    handle: handle,
                    tensorIndex: index,
                    rank: rank,
                    selector: OVTRTShapeSelectorOptimum
                ),
                maximum: shape(
                    handle: handle,
                    tensorIndex: index,
                    rank: rank,
                    selector: OVTRTShapeSelectorMaximum
                )
            )
        } else {
            profile = nil
        }
        return TensorRTEngineTensorDescriptor(
            name: name,
            ioMode: ioMode,
            elementType: elementType,
            declaredShape: declaredShape,
            profile: profile
        )
    }

    private static func shape(
        handle: OpaquePointer,
        tensorIndex: Int,
        rank: Int,
        selector: OVTRTShapeSelector
    ) throws(TensorRTEngineError) -> [Int] {
        var result: [Int] = []
        result.reserveCapacity(rank)
        for axis in 0 ..< rank {
            var dimension: Int64 = 0
            let status = TensorRTRuntimeStatus(
                ovtrt_engine_tensor_dimension(
                    handle,
                    UInt32(tensorIndex),
                    UInt32(axis),
                    selector,
                    &dimension
                )
            )
            guard
                status == .available,
                let value = Int(exactly: dimension)
            else {
                throw .inspectionFailed(
                    status: status,
                    tensorIndex: tensorIndex
                )
            }
            result.append(value)
        }
        return result
    }

    private static func elementType(
        _ raw: OVTRTTensorElementType
    ) -> VisionModelInputDescriptor.ElementType? {
        switch raw {
        case OVTRTTensorElementTypeFloat32:
            .float32
        case OVTRTTensorElementTypeFloat16:
            .float16
        case OVTRTTensorElementTypeInt8:
            .int8
        case OVTRTTensorElementTypeInt32:
            .int32
        case OVTRTTensorElementTypeInt64:
            .int64
        default:
            nil
        }
    }

    private static func validateRuntime(
        report: TensorRTEngineLoadReport,
        artifact: TensorRTEngineArtifactDescriptor
    ) throws(TensorRTEngineError) {
        guard report.checksumVerified else {
            throw .incompatibleArtifact(.checksumNotVerified)
        }
        guard
            report.tensorRTVersion == artifact.tensorRTVersion
        else {
            throw .incompatibleArtifact(
                .tensorRTVersion(
                    expected: artifact.tensorRTVersion,
                    actual: report.tensorRTVersion
                )
            )
        }
        guard
            report.cudaRuntimeVersion ==
                artifact.cudaRuntimeVersion
        else {
            throw .incompatibleArtifact(
                .cudaRuntimeVersion(
                    expected: artifact.cudaRuntimeVersion,
                    actual: report.cudaRuntimeVersion
                )
            )
        }
        guard
            report.computeCapabilityMajor ==
                artifact.computeCapabilityMajor,
            report.computeCapabilityMinor ==
                artifact.computeCapabilityMinor
        else {
            throw .incompatibleArtifact(
                .computeCapability(
                    expectedMajor:
                        artifact.computeCapabilityMajor,
                    expectedMinor:
                        artifact.computeCapabilityMinor,
                    actualMajor:
                        report.computeCapabilityMajor,
                    actualMinor:
                        report.computeCapabilityMinor
                )
            )
        }
    }

    private static func validateTensors(
        _ tensors: [TensorRTEngineTensorDescriptor],
        artifact: TensorRTStageEngineArtifactDescriptor
    ) throws(TensorRTEngineError) {
        guard
            let stage = artifact.artifact.semanticModel.stage(
                identifiedBy: artifact.stageID
            )
        else {
            throw .incompatibleArtifact(
                .missingTensor(artifact.inputTensorName)
            )
        }
        let expectedCount = 1 + artifact.outputBindings.count
        guard tensors.count == expectedCount else {
            throw .incompatibleArtifact(
                .tensorCount(
                    expected: expectedCount,
                    actual: tensors.count
                )
            )
        }
        let expectedNames = Set(
            [artifact.inputTensorName] +
            artifact.outputBindings.map {
                $0.engineTensorName
            }
        )
        for tensor in tensors where !expectedNames.contains(tensor.name) {
            throw .incompatibleArtifact(
                .unexpectedTensor(tensor.name)
            )
        }
        guard
            let input = tensors.first(
                where: { $0.name == artifact.inputTensorName }
            )
        else {
            throw .incompatibleArtifact(
                .missingTensor(artifact.inputTensorName)
            )
        }
        try validateInput(input, stage: stage)
        for binding in artifact.outputBindings {
            guard
                let actual = tensors.first(
                    where: {
                        $0.name == binding.engineTensorName
                    }
                ),
                let expected = stage.output(
                    identifiedBy: binding.semanticTensorID
                )
            else {
                throw .incompatibleArtifact(
                    .missingTensor(binding.engineTensorName)
                )
            }
            try validateTensor(
                actual,
                expectedMode: .output,
                expectedElementType: expected.elementType,
                expectedShape: expected.shape
            )
        }
    }

    private static func validateInput(
        _ tensor: TensorRTEngineTensorDescriptor,
        stage: VisionModelStageDescriptor
    ) throws(TensorRTEngineError) {
        let batch: VisionModelTensorDescriptor.Dimension
        switch stage.input.source {
        case .image:
            batch = .fixed(1)
        case .regions(_, _, _, let maximumCount, _):
            batch = .batch(maximum: maximumCount)
        }
        let expectedShape:
            [VisionModelTensorDescriptor.Dimension]
        switch stage.input.tensorLayout {
        case .channelsFirst:
            expectedShape = [
                batch,
                .fixed(3),
                .fixed(stage.input.height),
                .fixed(stage.input.width)
            ]
        case .channelsLast:
            expectedShape = [
                batch,
                .fixed(stage.input.height),
                .fixed(stage.input.width),
                .fixed(3)
            ]
        }
        try validateTensor(
            tensor,
            expectedMode: .input,
            expectedElementType: stage.input.elementType,
            expectedShape: expectedShape
        )
    }

    private static func validateTensor(
        _ tensor: TensorRTEngineTensorDescriptor,
        expectedMode: TensorRTEngineTensorDescriptor.IOMode,
        expectedElementType:
            VisionModelInputDescriptor.ElementType,
        expectedShape: [VisionModelTensorDescriptor.Dimension]
    ) throws(TensorRTEngineError) {
        guard tensor.ioMode == expectedMode else {
            throw .incompatibleArtifact(
                .ioMode(
                    tensor: tensor.name,
                    expected: expectedMode,
                    actual: tensor.ioMode
                )
            )
        }
        guard tensor.elementType == expectedElementType else {
            throw .incompatibleArtifact(
                .elementType(
                    tensor: tensor.name,
                    expected: expectedElementType,
                    actual: tensor.elementType
                )
            )
        }
        guard
            tensor.declaredShape.count == expectedShape.count
        else {
            throw .incompatibleArtifact(
                .tensorCount(
                    expected: expectedShape.count,
                    actual: tensor.declaredShape.count
                )
            )
        }
        for axis in expectedShape.indices {
            let expected = expectedShape[axis]
            let actual = tensor.declaredShape[axis]
            switch expected {
            case .fixed(let value):
                guard actual == value else {
                    throw .incompatibleArtifact(
                        .shape(
                            tensor: tensor.name,
                            axis: axis,
                            expected: expected,
                            actual: actual
                        )
                    )
                }
            case .batch(let maximum):
                if maximum == 1 {
                    guard actual == 1 else {
                        throw .incompatibleArtifact(
                            .shape(
                                tensor: tensor.name,
                                axis: axis,
                                expected: expected,
                                actual: actual
                            )
                        )
                    }
                    continue
                }
                guard actual == -1 else {
                    throw .incompatibleArtifact(
                        .shape(
                            tensor: tensor.name,
                            axis: axis,
                            expected: expected,
                            actual: actual
                        )
                    )
                }
                if expectedMode == .input {
                    guard let profile = tensor.profile else {
                        throw .incompatibleArtifact(
                            .missingProfile(tensor.name)
                        )
                    }
                    let minimum = profile.minimum[axis]
                    let optimum = profile.optimum[axis]
                    let actualMaximum = profile.maximum[axis]
                    guard
                        minimum >= 1,
                        minimum <= optimum,
                        optimum <= actualMaximum,
                        actualMaximum == maximum
                    else {
                        throw .incompatibleArtifact(
                            .invalidProfile(
                                tensor: tensor.name,
                                axis: axis,
                                minimum: minimum,
                                optimum: optimum,
                                maximum: actualMaximum,
                                expectedMaximum: maximum
                            )
                        )
                    }
                }
            case .variable:
                guard actual == -1 else {
                    throw .incompatibleArtifact(
                        .shape(
                            tensor: tensor.name,
                            axis: axis,
                            expected: expected,
                            actual: actual
                        )
                    )
                }
            }
        }
    }
}
