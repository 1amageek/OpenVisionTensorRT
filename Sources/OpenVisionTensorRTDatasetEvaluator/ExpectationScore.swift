/// Scores what the recognizer committed to against what the dataset says
/// happened.
///
/// Each expectation is claimed by at most one commitment and each commitment
/// claims at most one expectation, so a single gesture cannot be counted as
/// both a hit and a miss, and a burst of commitments inside one window cannot
/// inflate the score.
struct ExpectationScore: Sendable {
    /// What became of one expectation. The failures are kept apart because they
    /// are different defects: a label error means the wrong recognizer won the
    /// motion, a direction error means the right one read it backwards, and a
    /// temporal error means the evidence was gathered somewhere else.
    enum Outcome: Sendable, Hashable {
        case truePositive
        case labelError
        case directionError
        case temporalError
        case falseNegative

        var rawValue: String {
            switch self {
            case .truePositive: "truePositive"
            case .labelError: "labelError"
            case .directionError: "directionError"
            case .temporalError: "temporalError"
            case .falseNegative: "falseNegative"
            }
        }
    }

    struct Match: Sendable {
        let expectation: GestureExpectation.Resolved
        let outcome: Outcome
        /// `nil` exactly when the outcome is `.falseNegative`.
        let commitment: GestureCommitment?
        /// Measured overlap, reported whenever a commitment claimed the window
        /// so a threshold can be chosen from data instead of guessed.
        let intersectionOverUnion: Double?

        var json: String {
            "{\"expectation\":" + expectation.expectation.json
                + ",\"firstFrameIndex\":"
                + String(expectation.firstFrameIndex)
                + ",\"lastFrameIndex\":" + String(expectation.lastFrameIndex)
                + ",\"outcome\":" + jsonString(outcome.rawValue)
                + ",\"commitment\":" + (commitment?.json ?? "null")
                + ",\"intersectionOverUnion\":"
                + (intersectionOverUnion.map { String($0) } ?? "null") + "}"
        }
    }

    let matches: [Match]
    /// Commitments that fell inside no annotated window.
    let falsePositives: [GestureCommitment]
    /// Commitments the session declined to decide between. Not scored either
    /// way, but reported so they cannot pass unnoticed.
    let ambiguousCommitmentCount: Int
    /// Present only when the manifest declared one. Absent means overlap is
    /// measured and reported but not used to reject a match.
    let minimumIntersectionOverUnion: Double?

    init(
        expectations: [GestureExpectation.Resolved],
        commitments: [GestureCommitment],
        ambiguousCommitmentCount: Int,
        minimumIntersectionOverUnion: Double?
    ) {
        var unclaimed = Array(expectations.indices)
        var matches: [Int: Match] = [:]
        var falsePositives: [GestureCommitment] = []
        for commitment in commitments.sorted(by: {
            $0.commitmentFrameIndex < $1.commitmentFrameIndex
        }) {
            guard let position = unclaimed.firstIndex(where: {
                expectations[$0].contains(
                    frameIndex: commitment.commitmentFrameIndex
                )
            }) else {
                falsePositives.append(commitment)
                continue
            }
            let index = unclaimed.remove(at: position)
            let expectation = expectations[index]
            let overlap = commitment.intersectionOverUnion(with: expectation)
            let outcome: Outcome
            if commitment.identifier != expectation.identifier {
                outcome = .labelError
            } else if commitment.direction != expectation.direction {
                outcome = .directionError
            } else if let minimum = minimumIntersectionOverUnion,
                      overlap < minimum {
                outcome = .temporalError
            } else {
                outcome = .truePositive
            }
            matches[index] = Match(
                expectation: expectation,
                outcome: outcome,
                commitment: commitment,
                intersectionOverUnion: overlap
            )
        }
        for index in unclaimed {
            matches[index] = Match(
                expectation: expectations[index],
                outcome: .falseNegative,
                commitment: nil,
                intersectionOverUnion: nil
            )
        }
        self.matches = expectations.indices.compactMap { matches[$0] }
        self.falsePositives = falsePositives
        self.ambiguousCommitmentCount = ambiguousCommitmentCount
        self.minimumIntersectionOverUnion = minimumIntersectionOverUnion
    }

    func count(of outcome: Outcome) -> Int {
        matches.filter { $0.outcome == outcome }.count
    }

    /// A pass requires every expectation to be met and nothing to be committed
    /// outside one. A run that declares no expectations is not passing — it is
    /// unscored, and reports itself as such.
    var isPassing: Bool {
        !matches.isEmpty
            && matches.allSatisfy { $0.outcome == .truePositive }
            && falsePositives.isEmpty
    }

    var status: String {
        if matches.isEmpty { return "completedNoExpectation" }
        return isPassing ? "passed" : "failedExpectations"
    }

    var json: String {
        "{\"status\":" + jsonString(status)
            + ",\"expectationCount\":" + String(matches.count)
            + ",\"truePositiveCount\":" + String(count(of: .truePositive))
            + ",\"labelErrorCount\":" + String(count(of: .labelError))
            + ",\"directionErrorCount\":" + String(count(of: .directionError))
            + ",\"temporalErrorCount\":" + String(count(of: .temporalError))
            + ",\"falseNegativeCount\":" + String(count(of: .falseNegative))
            + ",\"falsePositiveCount\":" + String(falsePositives.count)
            + ",\"ambiguousCommitmentCount\":"
            + String(ambiguousCommitmentCount)
            + ",\"minimumIntersectionOverUnion\":"
            + (minimumIntersectionOverUnion.map { String($0) } ?? "null")
            + ",\"matches\":" + jsonArray(matches.map(\.json))
            + ",\"falsePositives\":"
            + jsonArray(falsePositives.map(\.json)) + "}"
    }
}
