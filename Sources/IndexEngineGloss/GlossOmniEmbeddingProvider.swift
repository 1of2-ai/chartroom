import Foundation
import Glossematics
import IndexEngine

/// Multimodal `Embedder` backed by a Glossematics model bundle
/// (`jina-embeddings-v5-omni-small` in the shipped artifact).
///
/// Text and images run through the *same* model into one shared space, so a text
/// query and an image document are directly comparable — a text search can retrieve a
/// photo. This is the default embedder for the app: it serves text exactly like the
/// text-only provider (same model id, same space) and additionally embeds images
/// when the bundle carries an image tower.
///
/// Core ML inference is synchronous and heavy; every call hops off the engine actor onto
/// a dedicated serial queue. Text is preloaded when the provider is created, while the
/// image tower is still paid for lazily the first time an image is embedded.
public final class GlossOmniEmbeddingProvider: Embedder, @unchecked Sendable {
    public let modelID: String
    public let dimension: Int
    public let embeddingSpaceID: String
    public let supportsImageEmbedding: Bool
    public let isModelBacked = true
    public var weakSimilarityThreshold: Float { GlossSimilarity.weakThreshold }

    private let omni: GlossOmniEmbedder
    private let textEmbedder: QueueConfinedGlossTextEmbedder
    private let queue = DispatchQueue(label: "indexengine.gloss.omni", qos: .userInitiated)

    init(
        omni: GlossOmniEmbedder,
        textEmbedder: GlossTextEmbedder,
        modelID: String,
        dimension: Int,
        embeddingSpaceID: String,
        supportsImageEmbedding: Bool
    ) {
        self.omni = omni
        self.textEmbedder = QueueConfinedGlossTextEmbedder(textEmbedder)
        self.modelID = modelID
        self.dimension = dimension
        self.embeddingSpaceID = embeddingSpaceID
        self.supportsImageEmbedding = supportsImageEmbedding
    }

    /// Load the omni model from a converted bundle (v1 manifests migrate in memory).
    /// Reuses the shared tokenizer staging so the text tower loads despite the bundle's
    /// missing `config.json`.
    public static func load(
        bundleURL: URL,
        compute: TextComputePreference = .efficiency
    ) async throws -> GlossOmniEmbeddingProvider {
        let loaded = try GlossBundleLoader.load(url: bundleURL)
        let manifest = loaded.bundle.manifest
        let tokenizerFolder = try GlossTokenizerStaging.resolvedFolder(for: loaded.bundle)

        var configuration = GlossOmniEmbedder.Configuration(manifest: manifest)
        // Absolute path → the omni embedder uses it verbatim instead of the bundle's folder.
        configuration.tokenizerFolder = tokenizerFolder.path
        configuration.textComputeUnits = compute.computeUnits
        // The preference governs the whole model, not just the text tower — otherwise
        // an efficiency/Low Power selection still lights up the GPU (~67 W) the first
        // time an image, audio clip, or video is ingested.
        configuration.encoderUnits = compute.computeUnits
        configuration.decoderUnits = compute.computeUnits

        let omni = GlossOmniEmbedder(bundle: loaded.bundle, configuration: configuration)
        // The omni embedder's own text embedder is internal to the SDK; build the provider's
        // from the same manifest so text inference can run synchronously on the serial queue.
        let textEmbedder = try await GlossTextEmbedder(
            multiFunctionModelURL: loaded.bundle.resolve(manifest.text.model),
            tokenizerFolder: tokenizerFolder,
            buckets: manifest.text.buckets,
            computeUnits: compute.computeUnits,
            prompts: .init(query: manifest.prompts.query, document: manifest.prompts.document),
            padTokenID: manifest.tokens.padID,
            batchSize: manifest.text.batch?.size,
            batchBuckets: manifest.text.batch?.buckets ?? []
        )
        return GlossOmniEmbeddingProvider(
            omni: omni,
            textEmbedder: textEmbedder,
            modelID: manifest.modelID,
            dimension: manifest.embeddingDimension,
            embeddingSpaceID: loaded.embeddingSpaceID,
            supportsImageEmbedding: loaded.bundle.capabilities.supportsImage
        )
    }

    public func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
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

    public func embedImage(at url: URL) async throws -> [Float] {
        guard supportsImageEmbedding else {
            throw EmbedderModalityError.imageEmbeddingUnsupported(modelID: modelID)
        }
        let omni = self.omni
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                let result: Result<[Float], any Error> = autoreleasepool {
                    Result { try omni.embed(imageURL: url) }
                }
                // Core ML can retain Objective-C temporaries until an autorelease pool drains.
                // Finish that cleanup on the serial inference queue before waking the caller;
                // otherwise the caller can race the image inference frame's teardown.
                continuation.resume(with: result)
            }
        }
    }
}
