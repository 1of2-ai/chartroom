import CoreML
import Foundation

/// High-level text embedding API: raw text -> task prefix -> tokenize -> Core ML encode (ANE)
/// -> L2-normalized embedding (optionally Matryoshka-truncated).
///
/// Two modes:
///  - single fixed-shape model (`init(modelURL:...)`): one bucket.
///  - multi-function model (`init(multiFunctionModelURL:...)`): functions `bucket_<S>` sharing
///    weights; the smallest fitting bucket is selected per text and lazily loaded.
public final class JinaTextEmbedder: @unchecked Sendable {
    private static let outputDimension = 1024

    public enum TextEmbedderError: Error {
        case noBucketsConfigured
        /// The text at this input position tokenized to nothing; embedding it is meaningless,
        /// and sending it on would fail the sibling texts sharing its Core ML batch.
        case emptyText(index: Int)
    }

    public enum Prompt: String {
        case query = "Query: "
        case document = "Document: "
        case none = ""
    }

    public let tokenizer: JinaTokenizer
    private let compiledURL: URL
    /// `nil` = ADAPTIVE placement (measured): ANE for buckets ≤128, GPU for ≥256. The text decoder is
    /// shallow (accurate on either), and has a latency crossover — ANE 7.5/12.3 ms vs GPU 9.6/15.8 ms
    /// at 32/128, but ANE 29/84 ms vs GPU 25/47 ms at 256/512 (GPU up to 1.76× faster on long text).
    private let forcedUnits: MLComputeUnits?
    private let isMultiFunction: Bool
    private let buckets: [Int]              // sorted ascending
    private let cacheLock = NSLock()        // guards `cache` for concurrent embed()
    private var cache: [Int: CoreMLTextEncoder] = [:]

    /// Latency-optimal unit for a length bucket (ANE ≤128, GPU ≥256), unless forced.
    private func units(forBucket b: Int) -> MLComputeUnits { forcedUnits ?? (b <= 128 ? .cpuAndNeuralEngine : .cpuAndGPU) }

    /// Largest sequence length this embedder can handle (longer is truncated keep-first).
    public var maxTokens: Int { buckets.last ?? 0 }
    public var availableBuckets: [Int] { buckets }

    /// Single fixed-shape model.
    public convenience init(
        modelURL: URL, tokenizerFolder: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws {
        let enc = try CoreMLTextEncoder(modelURL: modelURL, computeUnits: computeUnits)
        try await self.init(_compiledURL: enc.compiledURL, tokenizerFolder: tokenizerFolder,
                            forcedUnits: computeUnits, isMultiFunction: false,
                            buckets: [enc.seqLen], preloaded: [enc.seqLen: enc])
    }

    /// Multi-function model: functions named `bucket_<S>` for each S in `buckets`.
    /// `computeUnits: nil` (default) = ADAPTIVE placement (ANE for short buckets, GPU for long —
    /// measured 1.76× faster on 512-token text); pass a value to force all buckets onto one unit.
    public convenience init(
        multiFunctionModelURL: URL, tokenizerFolder: URL,
        buckets: [Int] = [32, 64, 128, 256, 512],
        computeUnits: MLComputeUnits? = nil
    ) async throws {
        let compiled: URL
        if multiFunctionModelURL.pathExtension == "mlmodelc" {
            compiled = multiFunctionModelURL
        } else {
            compiled = try await MLModel.compileModel(at: multiFunctionModelURL)
        }
        try await self.init(_compiledURL: compiled, tokenizerFolder: tokenizerFolder,
                            forcedUnits: computeUnits, isMultiFunction: true,
                            buckets: buckets.sorted(), preloaded: [:])
    }

    private init(
        _compiledURL: URL, tokenizerFolder: URL, forcedUnits: MLComputeUnits?,
        isMultiFunction: Bool, buckets: [Int], preloaded: [Int: CoreMLTextEncoder]
    ) async throws {
        self.tokenizer = try await JinaTokenizer(modelFolder: tokenizerFolder)
        self.compiledURL = _compiledURL
        self.forcedUnits = forcedUnits
        self.isMultiFunction = isMultiFunction
        self.buckets = buckets
        self.cache = preloaded
    }

    private func encoder(forTokenCount n: Int) throws -> (CoreMLTextEncoder, Int) {
        guard let bucket = buckets.first(where: { $0 >= n }) ?? buckets.last else {
            throw TextEmbedderError.noBucketsConfigured
        }
        cacheLock.lock()
        if let e = cache[bucket] {
            cacheLock.unlock()
            return (e, bucket)
        }
        cacheLock.unlock()

        // Load outside the lock — an MLModel load takes seconds, and holding the lock
        // across it stalls concurrent embeds whose bucket is already resident.
        let fn = isMultiFunction ? "bucket_\(bucket)" : nil
        let built = try CoreMLTextEncoder(modelURL: compiledURL, computeUnits: units(forBucket: bucket), functionName: fn)

        cacheLock.lock(); defer { cacheLock.unlock() }
        if let winner = cache[bucket] { return (winner, bucket) }
        cache[bucket] = built
        return (built, bucket)
    }

    /// Release every resident bucket model. The next embed reloads what it needs.
    public func unloadAll() {
        cacheLock.lock(); defer { cacheLock.unlock() }
        cache.removeAll()
    }

    /// Embed text. `dim` truncates to a Matryoshka dimension (re-normalized); nil = full 1024.
    public func embed(_ text: String, prompt: Prompt = .none, dim: Int? = nil) throws -> [Float] {
        try Self.validateRequestedDimension(dim)
        var ids = tokenizer.encode(prompt.rawValue + text)
        let (enc, bucket) = try encoder(forTokenCount: ids.count)
        if ids.count > bucket { ids = Array(ids.prefix(bucket)) }
        let full = try enc.encode(tokenIds: ids)
        if let d = dim { return try matryoshka(full, dim: d) }
        return full
    }

    /// Embed many texts, grouping them by sequence-length bucket so each bucket's texts run in
    /// one Core ML batch call. Vectors are returned in the input order.
    public func embed(batch texts: [String], prompt: Prompt = .none, dim: Int? = nil) throws -> [[Float]] {
        try Self.validateRequestedDimension(dim)
        guard !texts.isEmpty else { return [] }
        var perBucket: [Int: [(index: Int, ids: [Int32])]] = [:]
        for (index, text) in texts.enumerated() {
            var ids = tokenizer.encode(prompt.rawValue + text)
            guard !ids.isEmpty else { throw TextEmbedderError.emptyText(index: index) }
            guard let bucket = buckets.first(where: { $0 >= ids.count }) ?? buckets.last else {
                throw TextEmbedderError.noBucketsConfigured
            }
            if ids.count > bucket { ids = Array(ids.prefix(bucket)) }
            perBucket[bucket, default: []].append((index, ids))
        }
        var output = [[Float]](repeating: [], count: texts.count)
        for (bucket, items) in perBucket {
            let encoder = try encoder(forBucket: bucket)
            let embeddings = try encoder.encode(batch: items.map(\.ids))
            for (offset, item) in items.enumerated() {
                let full = embeddings[offset]
                if let dim {
                    output[item.index] = try matryoshka(full, dim: dim)
                } else {
                    output[item.index] = full
                }
            }
        }
        return output
    }

    static func validateRequestedDimension(_ dim: Int?) throws {
        guard let dim else { return }
        guard (1...outputDimension).contains(dim) else {
            throw MatryoshkaError.invalidDimension(requested: dim, available: outputDimension)
        }
    }

    private func encoder(forBucket bucket: Int) throws -> CoreMLTextEncoder {
        cacheLock.lock(); defer { cacheLock.unlock() }
        if let cached = cache[bucket] { return cached }
        let fn = isMultiFunction ? "bucket_\(bucket)" : nil
        let encoder = try CoreMLTextEncoder(modelURL: compiledURL, computeUnits: units(forBucket: bucket), functionName: fn)
        cache[bucket] = encoder
        return encoder
    }
}
