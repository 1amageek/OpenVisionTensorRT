import OpenVision

public actor OpenVisionTensorRTProviderSession:
    VisionProviderSession
{
    public nonisolated let descriptor:
        VisionProviderSessionDescriptor

    private let configuration:
        OpenVisionTensorRTProviderConfiguration
    private let preprocessor: RG10Preprocessor
    private let detectorEngine: TensorRTEngine
    private let posePipeline: TensorRTPosePipeline
    private let poseEngine: TensorRTEngine
    private let observationDecoder:
        TensorRTPoseObservationDecoder
    private var cancelledExecutions: Set<VisionExecutionID> = []
    private var isProcessing = false
    private var enginesArePrepared = false
    private var isShutDown = false

    init(
        descriptor: VisionProviderSessionDescriptor,
        configuration:
            OpenVisionTensorRTProviderConfiguration
    ) throws(VisionError) {
        self.descriptor = descriptor
        self.configuration = configuration
        do {
            preprocessor = try RG10Preprocessor(
                configuration:
                    configuration.detectorPreprocessing
            )
        } catch {
            throw Self.backendFailure(
                operation: "createPreprocessor",
                code: 10
            )
        }
        do {
            detectorEngine = try TensorRTEngine(
                path: configuration.detectorPlanPath,
                artifact: configuration.detectorArtifact
            )
        } catch {
            throw Self.backendFailure(
                operation: "createDetectorEngine",
                code: 11
            )
        }
        do {
            posePipeline = try TensorRTPosePipeline(
                configuration: configuration.posePipeline
            )
        } catch {
            throw Self.backendFailure(
                operation: "createPosePipeline",
                code: 12
            )
        }
        do {
            poseEngine = try TensorRTEngine(
                path: configuration.posePlanPath,
                artifact: configuration.poseArtifact
            )
        } catch {
            throw Self.backendFailure(
                operation: "createPoseEngine",
                code: 13
            )
        }
        observationDecoder = TensorRTPoseObservationDecoder(
            minimumConfidence:
                configuration.model.output.minimumJointConfidence
        )
    }

    func prepare() async throws(VisionError) {
        guard !isShutDown else {
            throw .sessionShutDown(descriptor.id)
        }
        guard !enginesArePrepared else {
            return
        }
        do {
            _ = try await detectorEngine.prepareExecution()
        } catch let error {
            throw Self.engineFailure(
                operation: "prepareDetectorEngine",
                error: error
            )
        }
        do {
            _ = try await poseEngine.prepareExecution()
        } catch let error {
            let preparationFailure = Self.engineFailure(
                operation: "preparePoseEngine",
                error: error
            )
            do {
                try await shutdown()
            } catch {
                throw Self.backendFailure(
                    operation: "cleanupPreparedEngines",
                    code: 18
                )
            }
            throw preparationFailure
        }
        enginesArePrepared = true
    }

    public func bodyPoseObservations(
        for request: DetectHumanBodyPoseRequest,
        input: VisionImageInput,
        executionID: VisionExecutionID
    ) async throws(VisionError) -> [HumanBodyPoseObservation] {
        guard !isShutDown else {
            throw .sessionShutDown(descriptor.id)
        }
        guard !isProcessing else {
            throw .providerBusy(descriptor.provider.id)
        }
        guard enginesArePrepared else {
            throw Self.backendFailure(
                operation: "enginesNotPrepared",
                code: 19
            )
        }
        guard request.descriptor == configuration.model.request else {
            throw .unsupportedRequest(request.descriptor)
        }
        guard request.regionOfInterest == .fullImage else {
            throw Self.backendFailure(
                operation: "regionOfInterest",
                code: 14
            )
        }
        guard executionID.sessionID == descriptor.id else {
            throw Self.backendFailure(
                operation: "executionSession",
                code: 15
            )
        }

        isProcessing = true
        defer {
            isProcessing = false
            cancelledExecutions.remove(executionID)
        }
        try cancellationFailure(executionID)

        let preprocessing: RG10PreprocessingOutput
        do {
            preprocessing = try await preprocessor.process(input)
        } catch {
            throw Self.backendFailure(
                operation: "preprocess",
                code: 20
            )
        }
        do {
            try cancellationFailure(executionID)
        } catch {
            if let cleanupFailure = await cleanup(
                rg10Tensor: preprocessing.tensor
            ) {
                throw cleanupFailure
            }
            throw error
        }

        let detectorOutput: TensorRTInferenceOutput
        do {
            detectorOutput = try await detectorEngine.execute(
                preprocessing.tensor,
                batchSize: 1
            )
        } catch let error {
            if let cleanupFailure = await cleanup(
                rg10Tensor: preprocessing.tensor
            ) {
                throw cleanupFailure
            }
            throw Self.engineFailure(
                operation: "detector",
                error: error
            )
        }
        let detectorTensors: (
            detections: TensorRTDeviceOutputTensor,
            classes: TensorRTDeviceOutputTensor
        )
        do {
            detectorTensors = try self.detectorTensors(
                detectorOutput
            )
        } catch {
            if let cleanupFailure = await cleanup(
                engineOutput: detectorOutput,
                rg10Tensor: preprocessing.tensor
            ) {
                throw cleanupFailure
            }
            throw error
        }

        let preparation: TensorRTPosePreparation
        do {
            preparation = try await posePipeline.prepareInput(
                source: preprocessing.tensor,
                detections: detectorTensors.detections,
                classes: detectorTensors.classes,
                orientation: input.orientation
            )
        } catch {
            if let cleanupFailure = await cleanup(
                engineOutput: detectorOutput,
                rg10Tensor: preprocessing.tensor
            ) {
                throw cleanupFailure
            }
            throw Self.backendFailure(
                operation: "preparePoseInput",
                code: 22
            )
        }
        if let cleanupFailure = await cleanup(
            engineOutput: detectorOutput,
            rg10Tensor: preprocessing.tensor
        ) {
            throw cleanupFailure
        }

        guard let poseInput = preparation.input else {
            return []
        }
        do {
            try cancellationFailure(executionID)
        } catch {
            if let cleanupFailure = await cleanup(
                poseInput: poseInput,
                discardsPreparedPoseFrame: true
            ) {
                throw cleanupFailure
            }
            throw error
        }

        let poseOutput: TensorRTInferenceOutput
        do {
            poseOutput = try await poseEngine.execute(
                poseInput,
                batchSize: poseInput.batchSize
            )
        } catch let error {
            if let cleanupFailure = await cleanup(
                poseInput: poseInput,
                discardsPreparedPoseFrame: true
            ) {
                throw cleanupFailure
            }
            throw Self.engineFailure(
                operation: "pose",
                error: error
            )
        }
        if let cleanupFailure = await cleanup(
            poseInput: poseInput
        ) {
            if let remainingFailure = await cleanup(
                engineOutput: poseOutput,
                discardsPreparedPoseFrame: true
            ) {
                throw remainingFailure
            }
            throw cleanupFailure
        }
        let poseTensors: (
            x: TensorRTDeviceOutputTensor,
            y: TensorRTDeviceOutputTensor
        )
        do {
            poseTensors = try self.poseTensors(poseOutput)
        } catch {
            if let cleanupFailure = await cleanup(
                engineOutput: poseOutput,
                discardsPreparedPoseFrame: true
            ) {
                throw cleanupFailure
            }
            throw error
        }

        let decoded: TensorRTDecodedPoseBatch
        do {
            decoded = try await posePipeline.decode(
                simCCX: poseTensors.x,
                simCCY: poseTensors.y
            )
        } catch {
            if let cleanupFailure = await cleanup(
                engineOutput: poseOutput,
                discardsPreparedPoseFrame: true
            ) {
                throw cleanupFailure
            }
            throw Self.backendFailure(
                operation: "decodeSimCC",
                code: 24
            )
        }
        if let cleanupFailure = await cleanup(
            engineOutput: poseOutput
        ) {
            throw cleanupFailure
        }
        try cancellationFailure(executionID)
        return try observationDecoder.observations(
            batch: decoded,
            regions: preparation.regions,
            timing: input.timing,
            provenance: input.observationProvenance,
            request: request,
            executionID: executionID
        )
    }

    public func cancel(
        _ executionID: VisionExecutionID
    ) async {
        guard executionID.sessionID == descriptor.id else {
            return
        }
        cancelledExecutions.insert(executionID)
    }

    public func shutdown() async throws(VisionError) {
        guard !isShutDown else {
            throw .sessionShutDown(descriptor.id)
        }
        guard !isProcessing else {
            throw .providerBusy(descriptor.provider.id)
        }
        var firstFailure: VisionError?
        do {
            try await posePipeline.shutdown()
        } catch {
            firstFailure = Self.backendFailure(
                operation: "shutdownPosePipeline",
                code: 30
            )
        }
        do {
            try await poseEngine.shutdown()
        } catch {
            if firstFailure == nil {
                firstFailure = Self.backendFailure(
                    operation: "shutdownPoseEngine",
                    code: 31
                )
            }
        }
        do {
            try await detectorEngine.shutdown()
        } catch {
            if firstFailure == nil {
                firstFailure = Self.backendFailure(
                    operation: "shutdownDetectorEngine",
                    code: 32
                )
            }
        }
        do {
            try await preprocessor.shutdown()
        } catch {
            if firstFailure == nil {
                firstFailure = Self.backendFailure(
                    operation: "shutdownPreprocessor",
                    code: 33
                )
            }
        }
        if let firstFailure {
            throw firstFailure
        }
        enginesArePrepared = false
        isShutDown = true
    }

    private func cancellationFailure(
        _ executionID: VisionExecutionID
    ) throws(VisionError) {
        if
            Task.isCancelled ||
            cancelledExecutions.contains(executionID)
        {
            throw .cancelled(executionID)
        }
    }

    private func detectorTensors(
        _ output: TensorRTInferenceOutput
    ) throws(VisionError) -> (
        detections: TensorRTDeviceOutputTensor,
        classes: TensorRTDeviceOutputTensor
    ) {
        guard
            let detections = tensor(
                RTMDetDWPoseBodyPoseManifest.detectionsTensor,
                output: output,
                artifact: configuration.detectorArtifact
            ),
            let classes = tensor(
                RTMDetDWPoseBodyPoseManifest.classesTensor,
                output: output,
                artifact: configuration.detectorArtifact
            )
        else {
            throw Self.backendFailure(
                operation: "detectorBindings",
                code: 42
            )
        }
        return (detections, classes)
    }

    private func poseTensors(
        _ output: TensorRTInferenceOutput
    ) throws(VisionError) -> (
        x: TensorRTDeviceOutputTensor,
        y: TensorRTDeviceOutputTensor
    ) {
        guard
            let x = tensor(
                RTMDetDWPoseBodyPoseManifest.simCCXTensor,
                output: output,
                artifact: configuration.poseArtifact
            ),
            let y = tensor(
                RTMDetDWPoseBodyPoseManifest.simCCYTensor,
                output: output,
                artifact: configuration.poseArtifact
            )
        else {
            throw Self.backendFailure(
                operation: "poseBindings",
                code: 43
            )
        }
        return (x, y)
    }

    private func tensor(
        _ semanticID: VisionModelTensorID,
        output: TensorRTInferenceOutput,
        artifact: TensorRTStageEngineArtifactDescriptor
    ) -> TensorRTDeviceOutputTensor? {
        guard
            let binding = artifact.outputBindings.first(
                where: { $0.semanticTensorID == semanticID }
            )
        else {
            return nil
        }
        return output.tensors.first {
            $0.name == binding.engineTensorName
        }
    }

    private func release(
        _ tensor: RG10DeviceTensor
    ) throws(VisionError) {
        do {
            try tensor.release()
        } catch {
            throw Self.backendFailure(
                operation: "releaseRG10Tensor",
                code: 50
            )
        }
    }

    private func release(
        _ input: TensorRTPoseInput
    ) throws(VisionError) {
        do {
            try input.release()
        } catch {
            throw Self.backendFailure(
                operation: "releasePoseInput",
                code: 51
            )
        }
    }

    private func release(
        _ output: TensorRTInferenceOutput
    ) throws(VisionError) {
        do {
            try output.release()
        } catch {
            throw Self.backendFailure(
                operation: "releaseEngineOutput",
                code: 52
            )
        }
    }

    private func discardPreparedFrame()
        async throws(VisionError)
    {
        do {
            try await posePipeline.discardPreparedFrame()
        } catch {
            throw Self.backendFailure(
                operation: "discardPreparedFrame",
                code: 53
            )
        }
    }

    private func cleanup(
        engineOutput: TensorRTInferenceOutput? = nil,
        poseInput: TensorRTPoseInput? = nil,
        rg10Tensor: RG10DeviceTensor? = nil,
        discardsPreparedPoseFrame: Bool = false
    ) async -> VisionError? {
        var firstFailure: VisionError?
        if let engineOutput {
            do {
                try release(engineOutput)
            } catch let error {
                firstFailure = error
            }
        }
        if let poseInput {
            do {
                try release(poseInput)
            } catch let error {
                if firstFailure == nil {
                    firstFailure = error
                }
            }
        }
        if let rg10Tensor {
            do {
                try release(rg10Tensor)
            } catch let error {
                if firstFailure == nil {
                    firstFailure = error
                }
            }
        }
        if discardsPreparedPoseFrame {
            do {
                try await discardPreparedFrame()
            } catch let error {
                if firstFailure == nil {
                    firstFailure = error
                }
            }
        }
        return firstFailure
    }

    private static func backendFailure(
        operation: String,
        code: Int32
    ) -> VisionError {
        OpenVisionTensorRTProvider.backendFailure(
            operation: operation,
            code: code
        )
    }

    private static func engineFailure(
        operation: String,
        error: TensorRTEngineError
    ) -> VisionError {
        switch error {
        case .executionPreparationFailed(let status, let report),
            .executionFailed(let status, let report),
            .executionCleanupFailed(let status, let report):
            return backendFailure(
                operation:
                    operation + "."
                    + engineStageName(report.failureStage),
                code: status.rawValue
            )
        case .invalidInput(let inputError):
            return backendFailure(
                operation:
                    operation + "."
                    + inputFailureName(inputError),
                code: 90
            )
        case .executionAlreadyPrepared:
            return backendFailure(
                operation: operation + ".alreadyPrepared",
                code: 91
            )
        case .executionNotPrepared:
            return backendFailure(
                operation: operation + ".notPrepared",
                code: 92
            )
        case .outputInUse:
            return backendFailure(
                operation: operation + ".outputInUse",
                code: 93
            )
        default:
            return backendFailure(
                operation: operation + ".engineContract",
                code: 99
            )
        }
    }

    private static func engineStageName(
        _ stage: TensorRTEngineExecutionStage
    ) -> String {
        switch stage {
        case .none:
            "none"
        case .configuration:
            "configuration"
        case .contextCreation:
            "contextCreation"
        case .streamCreation:
            "streamCreation"
        case .eventCreation:
            "eventCreation"
        case .shapeConfiguration:
            "shapeConfiguration"
        case .outputAllocation:
            "outputAllocation"
        case .tensorBinding:
            "tensorBinding"
        case .enqueue:
            "enqueue"
        case .synchronization:
            "synchronization"
        case .outputInspection:
            "outputInspection"
        case .cleanup:
            "cleanup"
        case .unknown:
            "unknown"
        }
    }

    private static func inputFailureName(
        _ error: TensorRTDeviceInputError
    ) -> String {
        switch error {
        case .released:
            "inputReleased"
        case .inaccessible:
            "inputInaccessible"
        case .invalidByteCount:
            "invalidInputByteCount"
        case .byteCountMismatch:
            "inputByteCount"
        }
    }
}
