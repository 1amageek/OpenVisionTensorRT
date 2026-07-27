package enum EvaluationGroundTruthParserError: Error, Equatable {
    case emptyField
    case invalidJoint(Int)
    case invalidCoordinate(index: Int, value: String)
    case outOfBounds(index: Int)
}

package enum EvaluationGroundTruthParser {
    package static func parse(
        _ field: Substring
    ) throws(EvaluationGroundTruthParserError) -> [EvaluationGroundTruthJoint] {
        guard !field.isEmpty else {
            throw .emptyField
        }
        var joints: [EvaluationGroundTruthJoint] = []
        let encodedJoints = field.split(separator: ";")
        joints.reserveCapacity(encodedJoints.count)
        for (index, encoded) in encodedJoints.enumerated() {
            let components = encoded.split(
                separator: ",",
                omittingEmptySubsequences: false
            )
            guard components.count == 3, !components[0].isEmpty else {
                throw .invalidJoint(index)
            }
            guard let x = Float(components[1]) else {
                throw .invalidCoordinate(index: index, value: String(components[1]))
            }
            guard let y = Float(components[2]) else {
                throw .invalidCoordinate(index: index, value: String(components[2]))
            }
            guard x.isFinite, y.isFinite,
                  (0...1).contains(x), (0...1).contains(y) else {
                throw .outOfBounds(index: index)
            }
            joints.append(
                EvaluationGroundTruthJoint(
                    name: String(components[0]),
                    x: x,
                    y: y
                )
            )
        }
        return joints
    }
}
