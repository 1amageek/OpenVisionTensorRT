import CTensorRTShim
import OpenVision

public actor TensorRTEngine {
    public let artifact: TensorRTStageEngineArtifactDescriptor
    public let loadReport: TensorRTEngineLoadReport
    public let tensors: [TensorRTEngineTensorDescriptor]

    private let owner: TensorRTEngineHandleOwner
    private var executionPrepared: Bool
    private var activeOutputLease: TensorRTOutputLeaseState?

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
            executionPrepared = false
            activeOutputLease = nil
        } catch {
            ovtrt_engine_destroy(handle)
            throw error
        }
    }

    public var isActive: Bool {
        owner.isActive
    }

    public var isExecutionPrepared: Bool {
        executionPrepared
    }

    public func prepareExecution()
        throws(TensorRTEngineError)
        -> TensorRTEngineExecutionReport
    {
        guard owner.isActive else {
            throw .alreadyShutDown
        }
        guard !executionPrepared else {
            throw .executionAlreadyPrepared
        }
        let outputCapacityByteCounts =
            try maximumOutputCapacityByteCounts()
        guard
            let rawOutputCapacityCount =
                UInt32(exactly: outputCapacityByteCounts.count),
            rawOutputCapacityCount > 0
        else {
            throw .invalidOutputCount(
                outputCapacityByteCounts.count
            )
        }
        var rawReport = OVTRTEngineExecutionResult()
        let status =
            outputCapacityByteCounts
            .withUnsafeBufferPointer { capacities in
                guard let baseAddress = capacities.baseAddress else {
                    return TensorRTRuntimeStatus.invalidArgument
                }
                return owner.withHandle { handle in
                    TensorRTRuntimeStatus(
                        ovtrt_engine_prepare_execution(
                            handle,
                            baseAddress,
                            rawOutputCapacityCount,
                            &rawReport
                        )
                    )
                } ?? .resourceBusy
            }
        let report = TensorRTEngineExecutionReport(rawReport)
        guard status == .available else {
            if report.failureStage == .cleanup {
                executionPrepared = true
                throw .executionCleanupFailed(
                    status: status,
                    report: report
                )
            }
            throw .executionPreparationFailed(
                status: status,
                report: report
            )
        }
        executionPrepared = true
        guard
            report.outputTensorCount == artifact.outputBindings.count,
            report.persistentDeviceAllocationCount == report.outputTensorCount,
            report.persistentDeviceAllocationByteCount > 0,
            report.explicitFrameDeviceAllocationCount == 0
        else {
            var rawCleanupReport = OVTRTEngineExecutionResult()
            let cleanupStatus = owner.withHandle { handle in
                TensorRTRuntimeStatus(
                    ovtrt_engine_release_execution(
                        handle,
                        &rawCleanupReport
                    )
                )
            } ?? .resourceBusy
            guard cleanupStatus == .available else {
                throw .executionCleanupFailed(
                    status: cleanupStatus,
                    report:
                        TensorRTEngineExecutionReport(
                            rawCleanupReport
                        )
                )
            }
            executionPrepared = false
            throw .executionPreparationFailed(
                status: .engineExecutionSetupFailure,
                report: report
            )
        }
        return report
    }

    public func execute(
        _ input: any TensorRTDeviceInput,
        batchSize: Int = 1
    ) throws(TensorRTEngineError) -> TensorRTInferenceOutput {
        guard owner.isActive else {
            throw .alreadyShutDown
        }
        guard executionPrepared else {
            throw .executionNotPrepared
        }
        if let activeOutputLease {
            guard activeOutputLease.isReleased else {
                throw .outputInUse
            }
            self.activeOutputLease = nil
        }
        let inputShape = try expectedInputShape(
            batchSize: batchSize
        )
        let declaredInputByteCount = input.byteCount
        guard
            declaredInputByteCount > 0,
            let rawInputByteCount =
                UInt64(exactly: declaredInputByteCount)
        else {
            throw .invalidInput(
                .invalidByteCount(declaredInputByteCount)
            )
        }
        var rawReport = OVTRTEngineExecutionResult()
        var executionStatus: TensorRTRuntimeStatus?
        var inputFailure: TensorRTDeviceInputError?
        do {
            try input.withTensorRTDeviceAddress {
                address,
                byteCount in
                guard address != 0 else {
                    inputFailure = .inaccessible
                    return
                }
                guard byteCount == declaredInputByteCount else {
                    inputFailure = .byteCountMismatch(
                        declared: declaredInputByteCount,
                        borrowed: byteCount
                    )
                    return
                }
                executionStatus =
                    inputShape.withUnsafeBufferPointer { dimensions in
                        guard
                            let baseAddress = dimensions.baseAddress,
                            let rawRank =
                                UInt32(exactly: dimensions.count)
                        else {
                            return .invalidArgument
                        }
                        return owner.withHandle { handle in
                            TensorRTRuntimeStatus(
                                ovtrt_engine_execute(
                                    handle,
                                    UnsafeRawPointer(
                                        bitPattern: address
                                    ),
                                    rawInputByteCount,
                                    baseAddress,
                                    rawRank,
                                    &rawReport
                                )
                            )
                        } ?? .resourceBusy
                    }
            }
        } catch let error {
            throw .invalidInput(error)
        }
        if let inputFailure {
            throw .invalidInput(inputFailure)
        }
        guard let status = executionStatus else {
            throw .invalidInput(.inaccessible)
        }
        let report = TensorRTEngineExecutionReport(rawReport)
        guard status == .available else {
            throw .executionFailed(
                status: status,
                report: report
            )
        }
        guard
            report.batchSize == batchSize,
            report.inputByteCount == rawInputByteCount,
            report.outputTensorCount == artifact.outputBindings.count,
            report.submissionCount > 0,
            report.explicitFrameDeviceAllocationCount == 0,
            report.inferenceMilliseconds.isFinite,
            report.inferenceMilliseconds >= 0
        else {
            throw .executionFailed(
                status: .engineExecutionFailure,
                report: report
            )
        }

        let lease = TensorRTOutputLeaseState()
        let outputDescriptors = tensors.filter {
            $0.ioMode == .output
        }
        var outputs: [TensorRTDeviceOutputTensor] = []
        outputs.reserveCapacity(outputDescriptors.count)
        for index in outputDescriptors.indices {
            outputs.append(
                try inspectedOutput(
                    index: index,
                    descriptor: outputDescriptors[index],
                    lease: lease
                )
            )
        }
        activeOutputLease = lease
        return TensorRTInferenceOutput(
            tensors: outputs,
            report: report,
            lease: lease
        )
    }

    private func maximumOutputCapacityByteCounts()
        throws(TensorRTEngineError) -> [UInt64]
    {
        guard
            let stage = artifact.artifact.semanticModel
                .stage(identifiedBy: artifact.stageID)
        else {
            throw .incompatibleArtifact(
                .missingStage(artifact.stageID)
            )
        }
        let outputDescriptors = tensors.filter {
            $0.ioMode == .output
        }
        var capacities: [UInt64] = []
        capacities.reserveCapacity(outputDescriptors.count)
        for descriptor in outputDescriptors {
            guard
                let binding = artifact.outputBindings.first(
                    where: {
                        $0.engineTensorName == descriptor.name
                    }
                ),
                let semanticOutput = stage.output(
                    identifiedBy: binding.semanticTensorID
                )
            else {
                throw .incompatibleArtifact(
                    .missingTensor(descriptor.name)
                )
            }
            var elementCount: UInt64 = 1
            for dimension in semanticOutput.shape {
                let maximum: Int
                switch dimension {
                case .fixed(let value):
                    maximum = value
                case .batch(let value):
                    maximum = value
                case .variable(let value):
                    maximum = value
                }
                guard
                    let converted = UInt64(exactly: maximum)
                else {
                    throw .invalidOutputCapacity(descriptor.name)
                }
                let product =
                    elementCount
                    .multipliedReportingOverflow(by: converted)
                guard !product.overflow else {
                    throw .invalidOutputCapacity(descriptor.name)
                }
                elementCount = product.partialValue
            }
            guard
                let elementByteCount = Self.elementByteCount(
                    semanticOutput.elementType
                )
            else {
                throw .invalidOutputCapacity(descriptor.name)
            }
            let byteCount =
                elementCount
                .multipliedReportingOverflow(
                    by: elementByteCount
                )
            guard !byteCount.overflow, byteCount.partialValue > 0 else {
                throw .invalidOutputCapacity(descriptor.name)
            }
            capacities.append(byteCount.partialValue)
        }
        return capacities
    }

    public func shutdown() throws(TensorRTEngineError) {
        if let activeOutputLease {
            guard activeOutputLease.isReleased else {
                throw .outputInUse
            }
            self.activeOutputLease = nil
        }
        if executionPrepared {
            var rawReport = OVTRTEngineExecutionResult()
            let status =
                owner.withHandle { handle in
                    TensorRTRuntimeStatus(
                        ovtrt_engine_release_execution(
                            handle,
                            &rawReport
                        )
                    )
                } ?? .resourceBusy
            let report =
                TensorRTEngineExecutionReport(rawReport)
            guard status == .available else {
                throw .executionCleanupFailed(
                    status: status,
                    report: report
                )
            }
            executionPrepared = false
        }
        guard let handle = owner.consume() else {
            throw .alreadyShutDown
        }
        ovtrt_engine_destroy(handle)
    }

    private func expectedInputShape(
        batchSize: Int
    ) throws(TensorRTEngineError) -> [Int64] {
        guard
            let stage = artifact.artifact.semanticModel
                .stage(identifiedBy: artifact.stageID)
        else {
            throw .incompatibleArtifact(
                .missingStage(artifact.stageID)
            )
        }
        let input = stage.input
        let maximumBatchSize: Int
        switch input.source {
        case .image:
            maximumBatchSize = 1
        case .regions(_, _, _, let maximumCount, _):
            maximumBatchSize = maximumCount
        }
        guard
            batchSize > 0,
            batchSize <= maximumBatchSize
        else {
            throw .invalidBatchSize(batchSize)
        }
        let values: [Int]
        switch input.tensorLayout {
        case .channelsFirst:
            values = [
                batchSize,
                3,
                input.height,
                input.width,
            ]
        case .channelsLast:
            values = [
                batchSize,
                input.height,
                input.width,
                3,
            ]
        }
        var shape: [Int64] = []
        shape.reserveCapacity(values.count)
        for value in values {
            guard let dimension = Int64(exactly: value) else {
                throw .invalidBatchSize(batchSize)
            }
            shape.append(dimension)
        }
        return shape
    }

    private func inspectedOutput(
        index: Int,
        descriptor: TensorRTEngineTensorDescriptor,
        lease: TensorRTOutputLeaseState
    ) throws(TensorRTEngineError)
        -> TensorRTDeviceOutputTensor
    {
        var rawView = OVTRTEngineOutputView()
        let status =
            owner.withHandle { handle in
                TensorRTRuntimeStatus(
                    ovtrt_engine_output(
                        handle,
                        UInt32(index),
                        &rawView
                    )
                )
            } ?? .resourceBusy
        guard
            status == .available,
            let address = rawView.deviceAddress,
            let byteCount = Int(exactly: rawView.byteCount),
            let elementCount =
                Int(exactly: rawView.elementCount),
            rawView.rank > 0,
            let rank = Int(exactly: rawView.rank),
            let actualElementType =
                Self.elementType(rawView.elementType),
            actualElementType == descriptor.elementType
        else {
            throw .outputInspectionFailed(
                status: status,
                outputIndex: index
            )
        }
        var shape: [Int] = []
        shape.reserveCapacity(rank)
        for axis in 0..<rank {
            var rawDimension: Int64 = 0
            let dimensionStatus =
                owner.withHandle { handle in
                    TensorRTRuntimeStatus(
                        ovtrt_engine_output_dimension(
                            handle,
                            UInt32(index),
                            UInt32(axis),
                            &rawDimension
                        )
                    )
                } ?? .resourceBusy
            guard
                dimensionStatus == .available,
                let dimension = Int(exactly: rawDimension),
                dimension >= 0
            else {
                throw .outputInspectionFailed(
                    status: dimensionStatus,
                    outputIndex: index
                )
            }
            shape.append(dimension)
        }
        return TensorRTDeviceOutputTensor(
            name: descriptor.name,
            address: UInt(bitPattern: address),
            byteCount: byteCount,
            elementCount: elementCount,
            shape: shape,
            elementType: actualElementType,
            owner: owner,
            lease: lease
        )
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
        guard
            let name = nameBytes.withUnsafeBufferPointer({ buffer in
            buffer.baseAddress.map { String(cString: $0) }
            })
        else {
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

    private static func elementByteCount(
        _ elementType: VisionModelInputDescriptor.ElementType
    ) -> UInt64? {
        switch elementType {
        case .float32, .int32:
            4
        case .float16:
            2
        case .int8:
            1
        case .int64:
            8
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
            report.cudaRuntimeVersion == artifact.cudaRuntimeVersion
        else {
            throw .incompatibleArtifact(
                .cudaRuntimeVersion(
                    expected: artifact.cudaRuntimeVersion,
                    actual: report.cudaRuntimeVersion
                )
            )
        }
        guard
            report.computeCapabilityMajor == artifact.computeCapabilityMajor,
            report.computeCapabilityMinor == artifact.computeCapabilityMinor
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
            [artifact.inputTensorName]
                + artifact.outputBindings.map {
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
        let expectedShape: [VisionModelTensorDescriptor.Dimension]
        switch stage.input.tensorLayout {
        case .channelsFirst:
            expectedShape = [
                batch,
                .fixed(3),
                .fixed(stage.input.height),
                .fixed(stage.input.width),
            ]
        case .channelsLast:
            expectedShape = [
                batch,
                .fixed(stage.input.height),
                .fixed(stage.input.width),
                .fixed(3),
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
