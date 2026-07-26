import OpenVision

public struct TensorRTStageEngineArtifactDescriptor:
    Sendable,
    Hashable
{
    public let artifact: TensorRTEngineArtifactDescriptor
    public let stageID: VisionModelStageID
    public let inputTensorName: String
    public let outputBindings: [TensorRTEngineOutputBinding]

    public init(
        artifact: TensorRTEngineArtifactDescriptor,
        stageID: VisionModelStageID,
        inputTensorName: String,
        outputBindings: [TensorRTEngineOutputBinding]
    ) throws(TensorRTStageEngineArtifactError) {
        guard !stageID.rawValue.isEmpty else {
            throw .emptyStageID
        }
        guard
            let stage = artifact.semanticModel.stage(
                identifiedBy: stageID
            )
        else {
            throw .missingStage(stageID)
        }
        guard !inputTensorName.isEmpty else {
            throw .emptyInputTensorName
        }

        var semanticTensorIDs: Set<VisionModelTensorID> = []
        var engineTensorNames: Set<String> = [inputTensorName]
        for binding in outputBindings {
            guard
                semanticTensorIDs.insert(
                    binding.semanticTensorID
                ).inserted
            else {
                throw .duplicateSemanticTensorID(
                    binding.semanticTensorID
                )
            }
            guard
                engineTensorNames.insert(
                    binding.engineTensorName
                ).inserted
            else {
                if binding.engineTensorName == inputTensorName {
                    throw .inputOutputNameCollision(
                        binding.engineTensorName
                    )
                }
                throw .duplicateEngineTensorName(
                    binding.engineTensorName
                )
            }
            guard
                stage.output(
                    identifiedBy: binding.semanticTensorID
                ) != nil
            else {
                throw .unknownSemanticTensorID(
                    binding.semanticTensorID
                )
            }
        }
        for output in stage.outputs {
            guard semanticTensorIDs.contains(output.id) else {
                throw .missingSemanticTensorID(output.id)
            }
        }

        self.artifact = artifact
        self.stageID = stageID
        self.inputTensorName = inputTensorName
        self.outputBindings = outputBindings
    }
}
