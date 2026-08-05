import Foundation
import Glossematics
import IndexEngine

/// `Embedder` backed by the Glossematics text tower.
///
/// This is the concrete provider `IndexEngine` uses for semantic vectors once a
/// model bundle is available. It maps the engine's `EmbedKind` onto the bundle's
/// retrieval prompts (`manifest.prompts`) and reports the model id and dimension
/// from the manifest so the engine derives a distinct `embeddingSpaceID` and never
/// mixes model vectors with mock vectors.
///
/// `GlossTextEmbedder.embed` is synchronous, CPU/ANE-heavy, and internally
/// thread-safe. The engine calls `embed` from inside the `IndexStore` actor, so
/// running inference inline would stall ingest and search. The provider therefore
/// hops every call onto a dedicated serial queue: inference runs off the actor,
/// and the serial queue keeps Core ML bucket compilation from stampeding.
/// `GlossTextEmbedder` is not `Sendable` (it caches lazily-built Core ML encoders). The
/// providers hop every call onto one serial `DispatchQueue` and the embedder never escapes
/// it, so crossing the `@Sendable` closure boundary inside this box is sound: confinement
/// is the synchronization.
struct QueueConfinedGlossTextEmbedder: @unchecked Sendable {
    let value: GlossTextEmbedder
    init(_ value: GlossTextEmbedder) { self.value = value }
}

public final class GlossTextEmbeddingProvider: Embedder, @unchecked Sendable {
    public let modelID: String
    public let dimension: Int
    public let embeddingSpaceID: String
    public let isModelBacked = true
    public let weakSimilarityThreshold: Float

    private let textEmbedder: QueueConfinedGlossTextEmbedder
    private let queue = DispatchQueue(label: "indexengine.gloss.text", qos: .userInitiated)

    init(
        textEmbedder: GlossTextEmbedder,
        modelID: String,
        dimension: Int,
        embeddingSpaceID: String,
        weakSimilarityThreshold: Float
    ) {
        self.textEmbedder = QueueConfinedGlossTextEmbedder(textEmbedder)
        self.modelID = modelID
        self.dimension = dimension
        self.embeddingSpaceID = embeddingSpaceID
        self.weakSimilarityThreshold = weakSimilarityThreshold
    }

    /// Load the text tower from a converted model bundle directory.
    ///
    /// Reads `manifest.json` (migrating v1 in memory), resolves the multi-function text
    /// model and tokenizer folder, and compiles the Core ML model. Throws if the bundle
    /// is malformed or the model cannot be compiled/loaded.
    public static func load(
        bundleURL: URL,
        compute: TextComputePreference = .efficiency,
        weakSimilarityThresholdOverride: Float? = nil
    ) async throws -> GlossTextEmbeddingProvider {
        let loaded = try GlossBundleLoader.load(url: bundleURL)
        let manifest = loaded.bundle.manifest
        let modelURL = loaded.bundle.resolve(manifest.text.model)
        let tokenizerFolder = try GlossTokenizerStaging.resolvedFolder(for: loaded.bundle)
        let textEmbedder = try await GlossTextEmbedder(
            multiFunctionModelURL: modelURL,
            tokenizerFolder: tokenizerFolder,
            buckets: manifest.text.buckets,
            computeUnits: compute.computeUnits,
            prompts: .init(query: manifest.prompts.query, document: manifest.prompts.document),
            padTokenID: manifest.tokens.padID,
            batchSize: manifest.text.batch?.size,
            batchBuckets: manifest.text.batch?.buckets ?? []
        )
        return GlossTextEmbeddingProvider(
            textEmbedder: textEmbedder,
            modelID: manifest.modelID,
            dimension: manifest.embeddingDimension,
            embeddingSpaceID: loaded.embeddingSpaceID,
            weakSimilarityThreshold: GlossWeakSimilarityThreshold.threshold(
                forModelID: manifest.modelID,
                override: weakSimilarityThresholdOverride
            )
        )
    }

    public func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        // Capture only Sendable values in the @Sendable closure; build the
        // (non-Sendable) prompt inside it.
        let isQuery = kind == .query
        let embedder = textEmbedder
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let prompt: GlossTextEmbedder.Prompt = isQuery ? .query : .document
                    continuation.resume(returning: try embedder.value.embed(text, prompt: prompt))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    public func embed(_ texts: [String], kind: EmbedKind) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let isQuery = kind == .query
        let embedder = textEmbedder
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let prompt: GlossTextEmbedder.Prompt = isQuery ? .query : .document
                    continuation.resume(returning: try embedder.value.embed(texts: texts, prompt: prompt))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
