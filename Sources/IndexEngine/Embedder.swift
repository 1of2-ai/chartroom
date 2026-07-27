import Foundation

/// Whether text is being embedded as a search query or as a stored document.
/// jina-embeddings-v5 uses retrieval prefixes; a real embedder prepends
/// `jinaPrefix`. The hashing mock ignores it (it is not jina).
public enum EmbedKind: String, Codable, Hashable, Sendable {
    case query, document
    public var jinaPrefix: String { self == .query ? "Query: " : "Document: " }
}

/// The one thing the index needs from an embedding model: text in, a vector out.
/// The real jina-v5 omni (multimodal) / text-small (fallback) CoreML/ANE models
/// drop in behind this; `modelID` tags every stored vector so two models' spaces
/// are never silently mixed in one search.
public protocol Embedder: Sendable {
    var modelID: String { get }
    var dimension: Int { get }
    var embeddingSpaceID: String { get }

    /// Whether these vectors come from a real embedding model, or from a stand-in that only
    /// produces something vector-shaped.
    ///
    /// Deliberately has no default. A fallback embedder that inherited `true` is exactly the
    /// failure this exists to prevent: hashing-backed retrieval reported to the user as
    /// model-backed semantic search, indistinguishable in status from the real thing. Every
    /// conformance states which it is, and the compiler enforces that it states something.
    var isModelBacked: Bool { get }

    /// Cosine similarity below which a vector hit carries no retrieval signal.
    ///
    /// Vector search always returns its top-N by cosine, so a query that matches nothing
    /// still comes back full — the rank is real, the relevance is not. Retrieval marks a
    /// hit weak when its *only* evidence is a similarity under this value, which is what
    /// lets every client apply one relevance rule instead of inventing its own.
    ///
    /// The right value is a property of the model's similarity distribution, so it also has no
    /// default. It previously defaulted to a near-orthogonal 0.15, and every real model then
    /// inherited it: dense transformer encoders put unrelated text far above that, so the weak
    /// partition admitted noise as answers and the guard was inert exactly where it mattered. A
    /// conformance must state a value it has measured for its own model.
    var weakSimilarityThreshold: Float { get }

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float]

    /// Embed many texts at once. Batch-capable backends (Jina) group these by sequence-length
    /// bucket and run each group in one Core ML call, amortizing per-call overhead and keeping
    /// the ANE fed. Returns vectors in the input order.
    func embed(_ texts: [String], kind: EmbedKind) async throws -> [[Float]]

    /// Whether this embedder can embed image files into the *same* vector space as its
    /// text output. Multimodal models (Jina omni) do; text-only embedders do not.
    var supportsImageEmbedding: Bool { get }

    /// Embed an image file into the shared embedding space, so a text query and an image
    /// document are directly comparable. Defaults to unsupported.
    func embedImage(at url: URL) async throws -> [Float]
}

public extension Embedder {
    var embeddingSpaceID: String { "\(modelID):\(dimension)" }

    func embedQuery(_ text: String) async throws -> [Float] { try await embed(text, kind: .query) }
    func embedDocument(_ text: String) async throws -> [Float] { try await embed(text, kind: .document) }

    /// Default batch path: embed one at a time. Batch-capable embedders override this.
    func embed(_ texts: [String], kind: EmbedKind) async throws -> [[Float]] {
        var results: [[Float]] = []
        results.reserveCapacity(texts.count)
        for text in texts { results.append(try await embed(text, kind: kind)) }
        return results
    }

    var supportsImageEmbedding: Bool { false }

    func embedImage(at url: URL) async throws -> [Float] {
        throw EmbedderModalityError.imageEmbeddingUnsupported(modelID: modelID)
    }
}

/// Raised when an embedder is asked for a modality it does not support.
public enum EmbedderModalityError: Error, CustomStringConvertible {
    case imageEmbeddingUnsupported(modelID: String)

    public var description: String {
        switch self {
        case let .imageEmbeddingUnsupported(modelID):
            return "Embedder \(modelID) does not support image embedding."
        }
    }
}

/// Deterministic, dependency-free embedder for tests and bring-up: hashes tokens
/// into a fixed-width bag-of-words vector, then L2-normalizes. Overlapping text
/// yields high cosine similarity, so the retrieval pipeline is exercisable before
/// the real jina-v5 CoreML model is wired in behind the same `Embedder` API.
public struct HashingEmbedder: Embedder {
    public let modelID: String
    public let dimension: Int

    /// Not a model. Token hashing produces lexical overlap in vector clothing, so an index using
    /// this has no semantic channel at all — three lexical channels wearing the name of a hybrid.
    public let isModelBacked = false

    /// Exact-token overlap of an L2-normalized bag of words. Unrelated text shares no token bucket
    /// and scores at or near zero, so this cuts the genuinely orthogonal without discarding the
    /// partial-overlap matches the mock exists to demonstrate.
    public let weakSimilarityThreshold: Float = 0.15

    public init(modelID: String = "hashing-mock-v1", dimension: Int = 1024) {
        self.modelID = modelID
        self.dimension = dimension
    }

    public func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        var v = [Float](repeating: 0, count: dimension)
        for token in Self.tokens(text) {
            v[Int(Self.hash(token) % UInt64(dimension))] += 1
        }
        let norm = (v.reduce(Float(0)) { $0 + $1 * $1 }).squareRoot()
        if norm > 0 { for i in v.indices { v[i] /= norm } }
        return v
    }

    /// Lowercase, split on non-alphanumerics. Also the FTS5 query tokenizer.
    static func tokens(_ s: String) -> [String] {
        s.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init)
    }

    /// FNV-1a, stable across processes because Swift's `Hasher` is per-process seeded.
    static func hash(_ s: String) -> UInt64 {
        var h: UInt64 = 0xcbf29ce484222325
        for b in s.utf8 { h = (h ^ UInt64(b)) &* 0x100000001b3 }
        return h
    }
}
