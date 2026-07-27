import Foundation
import IndexEngine
import JinaEmbeddings

/// `Embedder` backed by the real `jina-embeddings-v5-omni-small` text tower.
///
/// This is the concrete provider `IndexEngine` uses for semantic vectors once a
/// model bundle is available. It maps the engine's `EmbedKind` onto Jina's
/// retrieval prompts (`Query: ` / `Document: `) and reports the model id and
/// dimension from the bundle manifest so the engine derives a distinct
/// `embeddingSpaceID` and never mixes Jina vectors with mock vectors.
///
/// `JinaTextEmbedder.embed` is synchronous, CPU/ANE-heavy, and internally
/// thread-safe. The engine calls `embed` from inside the `IndexStore` actor, so
/// running inference inline would stall ingest and search. The provider therefore
/// hops every call onto a dedicated serial queue: inference runs off the actor,
/// and the serial queue keeps Core ML bucket compilation from stampeding.
public final class JinaTextEmbeddingProvider: Embedder, @unchecked Sendable {
    public let modelID: String
    public let dimension: Int
    public let embeddingSpaceID: String
    public let isModelBacked = true
    public var weakSimilarityThreshold: Float { JinaSimilarity.weakThreshold }

    private let textEmbedder: JinaTextEmbedder
    private let queue = DispatchQueue(label: "indexengine.jina.text", qos: .userInitiated)

    init(textEmbedder: JinaTextEmbedder, modelID: String, dimension: Int, embeddingSpaceID: String) {
        self.textEmbedder = textEmbedder
        self.modelID = modelID
        self.dimension = dimension
        self.embeddingSpaceID = embeddingSpaceID
    }

    /// Load the text tower from a converted model bundle directory.
    ///
    /// Reads `manifest.json`, resolves the multi-function text model and tokenizer
    /// folder, and compiles the Core ML model. Throws if the bundle is malformed or
    /// the model cannot be compiled/loaded.
    public static func load(
        bundleURL: URL,
        compute: TextComputePreference = .efficiency
    ) async throws -> JinaTextEmbeddingProvider {
        let bundle = try JinaModelBundle(url: bundleURL)
        let modelURL = bundle.resolve(bundle.manifest.text.model)
        let tokenizerFolder = try JinaTokenizerStaging.resolvedFolder(for: bundle)
        let textEmbedder = try await JinaTextEmbedder(
            multiFunctionModelURL: modelURL,
            tokenizerFolder: tokenizerFolder,
            buckets: bundle.manifest.text.buckets,
            computeUnits: compute.computeUnits
        )
        return JinaTextEmbeddingProvider(
            textEmbedder: textEmbedder,
            modelID: bundle.manifest.modelID,
            dimension: bundle.manifest.embeddingDimension,
            embeddingSpaceID: bundle.embeddingSpaceID
        )
    }

    public func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        // Capture only Sendable values in the @Sendable closure; build the
        // (non-Sendable) Jina prompt inside it.
        let isQuery = kind == .query
        let embedder = textEmbedder
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let prompt: JinaTextEmbedder.Prompt = isQuery ? .query : .document
                    continuation.resume(returning: try embedder.embed(text, prompt: prompt))
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
                    let prompt: JinaTextEmbedder.Prompt = isQuery ? .query : .document
                    continuation.resume(returning: try embedder.embed(batch: texts, prompt: prompt))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
