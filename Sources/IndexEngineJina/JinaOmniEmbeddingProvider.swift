import Foundation
import IndexEngine
import JinaEmbeddings

/// Multimodal `Embedder` backed by `jina-embeddings-v5-omni-small`.
///
/// Text and images run through the *same* model into one shared 1024-d space, so a text
/// query and an image document are directly comparable — a text search can retrieve a
/// photo. This is the default embedder for the app: it serves text exactly like the
/// text-only provider (same model id, same space) and additionally embeds images.
///
/// Core ML inference is synchronous and heavy; every call hops off the engine actor onto
/// a dedicated serial queue. Text is preloaded when the provider is created, while the
/// image tower is still paid for lazily the first time an image is embedded.
public final class JinaOmniEmbeddingProvider: Embedder, @unchecked Sendable {
    public let modelID: String
    public let dimension: Int
    public let embeddingSpaceID: String
    public var supportsImageEmbedding: Bool { true }
    public let isModelBacked = true
    public var weakSimilarityThreshold: Float { JinaSimilarity.weakThreshold }

    private let omni: JinaOmniEmbedder
    private let textEmbedder: JinaTextEmbedder
    private let queue = DispatchQueue(label: "indexengine.jina.omni", qos: .userInitiated)

    init(
        omni: JinaOmniEmbedder,
        textEmbedder: JinaTextEmbedder,
        modelID: String,
        dimension: Int,
        embeddingSpaceID: String
    ) {
        self.omni = omni
        self.textEmbedder = textEmbedder
        self.modelID = modelID
        self.dimension = dimension
        self.embeddingSpaceID = embeddingSpaceID
    }

    /// Load the omni model from a converted bundle. Reuses the shared tokenizer staging so
    /// the text tower loads despite the bundle's missing `config.json`.
    public static func load(
        bundleURL: URL,
        compute: TextComputePreference = .efficiency
    ) async throws -> JinaOmniEmbeddingProvider {
        let bundle = try JinaModelBundle(url: bundleURL)
        let tokenizerFolder = try JinaTokenizerStaging.resolvedFolder(for: bundle)

        var configuration = JinaOmniEmbedder.Configuration(manifest: bundle.manifest)
        // Absolute path → the omni embedder uses it verbatim instead of the bundle's folder.
        configuration.tokenizerFolder = tokenizerFolder.path
        configuration.textComputeUnits = compute.computeUnits
        // The preference governs the whole model, not just the text tower — otherwise
        // an efficiency/Low Power selection still lights up the GPU (~67 W) the first
        // time an image, audio clip, or video is ingested.
        configuration.encoderUnits = compute.computeUnits
        configuration.decoderUnits = compute.computeUnits

        let omni = JinaOmniEmbedder(bundle: bundle, configuration: configuration)
        let textEmbedder = try await omni.textEmbedder()
        return JinaOmniEmbeddingProvider(
            omni: omni,
            textEmbedder: textEmbedder,
            modelID: bundle.manifest.modelID,
            dimension: bundle.manifest.embeddingDimension,
            embeddingSpaceID: bundle.embeddingSpaceID
        )
    }

    public func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
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

    public func embedImage(at url: URL) async throws -> [Float] {
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
