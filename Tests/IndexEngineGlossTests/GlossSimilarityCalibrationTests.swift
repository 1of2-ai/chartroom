import Foundation
import IndexEngine
import Testing
@testable import IndexEngineGloss

/// Guards the one number that decides whether a vector hit counts as an answer.
///
/// `GlossSimilarity.weakThreshold` was derived from the distributions below rather than assumed, so
/// the constant is only defensible while the distributions still hold. This re-derives them against
/// the live model: if a model revision moves either population across the threshold, this fails here
/// rather than silently changing what every client treats as a result.
///
/// Gated on the model bundle like the other model-backed tests — a pointer-only checkout skips.
@Suite("Gloss similarity calibration")
struct GlossSimilarityCalibrationTests {
    static let bundle: URL? = GlossModelBundleLocator.locate()

    /// Disjoint topics with no shared content words: the noise floor the threshold must clear.
    static let unrelatedPairs: [(String, String)] = [
        ("How do I renew a passport in the United States?", "The sourdough starter should double in size before baking."),
        ("Quarterly revenue rose twelve percent year over year.", "Glacial moraines record the extent of past ice sheets."),
        ("The cat refuses to use the new litter box.", "TLS certificate pinning prevents interception by a proxy."),
        ("Recommend a hotel near the Barcelona waterfront.", "Mitochondria generate ATP through oxidative phosphorylation."),
        ("My knee hurts after running downhill.", "Compile the kernel module against the running headers."),
        ("What time does the museum close on Sundays?", "Reinforced concrete requires steel in the tension zone."),
        ("She inherited a farmhouse in rural Portugal.", "The compiler eliminates the bounds check in this loop."),
        ("Bake the salmon at two hundred degrees.", "Municipal bonds are exempt from federal income tax."),
        ("The election results were certified on Tuesday.", "Reduce shutter speed to capture motion blur in water."),
        ("Install the dishwasher drain hose above the trap.", "Ancient Greek tragedies were performed at civic festivals."),
        ("Our flight to Reykjavik was delayed six hours.", "The enzyme denatures irreversibly above sixty degrees."),
        ("He plays bass in a jazz quartet on weekends.", "Distributed consensus requires a quorum of replicas."),
    ]

    /// Restatements carrying the same meaning in different words: the signal the threshold must keep.
    static let relatedPairs: [(String, String)] = [
        ("How do I reset my account password?", "What are the steps to change my login credentials?"),
        ("The server returns a 500 error under load.", "Under heavy traffic the backend responds with an internal server error."),
        ("Bake the bread until the crust is golden brown.", "Leave the loaf in the oven until it develops a deep golden crust."),
        ("Revenue grew substantially last quarter.", "Sales increased sharply over the previous three months."),
        ("The dog needs a walk twice a day.", "Take the dog out for exercise morning and evening."),
        ("Encrypt the database backups before uploading them.", "Backups should be encrypted prior to transfer to remote storage."),
    ]

    @Test(
        "the weak threshold separates unrelated text from restatements on the live model",
        .enabled(if: GlossSimilarityCalibrationTests.bundle != nil)
    )
    func thresholdSeparatesPopulations() async throws {
        let bundleURL = try #require(Self.bundle)
        let provider = try await GlossTextEmbeddingProvider.load(bundleURL: bundleURL)
        let threshold = GlossSimilarity.weakThreshold

        func similarity(_ query: String, _ document: String) async throws -> Float {
            let q = try await provider.embed(query, kind: .query)
            let d = try await provider.embed(document, kind: .document)
            return zip(q, d).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        }

        for (query, document) in Self.unrelatedPairs {
            let score = try await similarity(query, document)
            #expect(
                score < threshold,
                "unrelated pair scored \(score), at or above the \(threshold) weak threshold: \(query)"
            )
        }

        for (query, document) in Self.relatedPairs {
            let score = try await similarity(query, document)
            #expect(
                score > threshold,
                "restatement scored \(score), at or below the \(threshold) weak threshold: \(query)"
            )
        }
    }

    /// The threshold is only meaningful if it is stated. This is the regression guard for the
    /// default that used to be inherited from the protocol.
    @Test("both providers report the calibrated threshold rather than inheriting one")
    func providersStateTheirOwnThreshold() {
        #expect(GlossSimilarity.weakThreshold > HashingEmbedder().weakSimilarityThreshold)
    }
}
