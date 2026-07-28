import CoreML
import Foundation

/// Low-level Core ML wrapper for the converted text tower (`text_b{S}.mlpackage`).
///
/// Inputs (built here): `input_ids` (1,S) int32 right-padded, `position_ids` (3,1,S) int32,
/// `selector` (1,S) one-hot at the last real token. Output: `embedding` (1,1024) L2-normed.
/// Causal attention + right padding means pad tokens never affect the pooled embedding.
public final class CoreMLTextEncoder: @unchecked Sendable {
    public static let padTokenID: Int32 = 151643  // <|endoftext|>; value is irrelevant (causal+selector)

    public let model: MLModel
    public let seqLen: Int
    public let embeddingDim: Int
    /// The compiled `.mlmodelc` URL (reuse to avoid recompiling the same package per function).
    public let compiledURL: URL

    public init(
        modelURL: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        functionName: String? = nil
    ) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        if let functionName { config.functionName = functionName }
        let compiled = modelURL.pathExtension == "mlmodelc"
            ? modelURL : try MLModel.compileModel(at: modelURL)
        self.compiledURL = compiled
        self.model = try MLModel(contentsOf: compiled, configuration: config)

        guard let idsConstraint = model.modelDescription
            .inputDescriptionsByName["input_ids"]?.multiArrayConstraint else {
            throw EncoderError.badModel("missing input_ids constraint")
        }
        guard let lastDimension = idsConstraint.shape.last else {
            throw EncoderError.badModel("input_ids constraint has an empty shape")
        }
        self.seqLen = lastDimension.intValue
        if let out = model.modelDescription.outputDescriptionsByName["embedding"]?.multiArrayConstraint {
            self.embeddingDim = out.shape.last?.intValue ?? 1024
        } else {
            self.embeddingDim = 1024
        }
    }

    public enum EncoderError: Error { case badModel(String), tooLong(Int, Int), emptyInput, noOutput }

    /// Build the (input_ids, position_ids, selector) provider for one right-padded sequence.
    /// The single and batch entry points share this — the fill loops must never drift apart.
    private func makeProvider(tokenIds: [Int32]) throws -> MLDictionaryFeatureProvider {
        let S = seqLen
        guard !tokenIds.isEmpty else { throw EncoderError.emptyInput }
        guard tokenIds.count <= S else {
            throw EncoderError.tooLong(tokenIds.count, S)
        }
        let last = tokenIds.count - 1
        let ids = try MLMultiArray(shape: [1, NSNumber(value: S)], dataType: .int32)
        let pos = try MLMultiArray(shape: [3, 1, NSNumber(value: S)], dataType: .int32)
        let sel = try MLMultiArray(shape: [1, NSNumber(value: S)], dataType: .float32)
        try withDenseTensor(ids, as: Int32.self) { idsPtr in
            for i in 0..<S { idsPtr[i] = i < tokenIds.count ? tokenIds[i] : Self.padTokenID }
        }
        try withDenseTensor(pos, as: Int32.self) { posPtr in
            for i in 0..<S { posPtr[i] = Int32(i); posPtr[S + i] = Int32(i); posPtr[2 * S + i] = Int32(i) }
        }
        try withDenseTensor(sel, as: Float.self) { selPtr in
            for i in 0..<S { selPtr[i] = (i == last) ? 1.0 : 0.0 }
        }
        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": ids, "position_ids": pos, "selector": sel,
        ])
    }

    private func embedding(from features: MLFeatureProvider) throws -> [Float] {
        guard let arr = features.featureValue(for: "embedding")?.multiArrayValue else {
            throw EncoderError.noOutput
        }
        return try floatValues(of: arr)
    }

    /// Encode a real (unpadded) token id sequence into an L2-normalized embedding.
    public func encode(tokenIds: [Int32]) throws -> [Float] {
        let out = try model.prediction(from: try makeProvider(tokenIds: tokenIds))
        return try embedding(from: out)
    }

    /// Encode many same-bucket sequences in one Core ML batch call. With today's batch-1 model
    /// this runs them back-to-back via `predictions(fromBatch:)` — less per-call overhead and a
    /// continuously-fed ANE; a batch-dim model would additionally amortize the weight stream.
    public func encode(batch sequences: [[Int32]]) throws -> [[Float]] {
        guard !sequences.isEmpty else { return [] }
        // Per-item autorelease pools: Core ML feature objects are autoreleased, and
        // without draining they accumulate across a large batch.
        let providers = try sequences.map { sequence in
            try autoreleasepool { try makeProvider(tokenIds: sequence) }
        }
        let outputs = try model.predictions(fromBatch: MLArrayBatchProvider(array: providers))
        var results: [[Float]] = []
        results.reserveCapacity(outputs.count)
        for index in 0 ..< outputs.count {
            results.append(try autoreleasepool { try embedding(from: outputs.features(at: index)) })
        }
        return results
    }
}

public enum MatryoshkaError: Error, Equatable, Sendable {
    case invalidDimension(requested: Int, available: Int)
}

/// L2-normalized truncation for Matryoshka dims: `normalize(full[:dim])`.
public func matryoshka(_ embedding: [Float], dim: Int) throws -> [Float] {
    guard dim > 0, dim <= embedding.count else {
        throw MatryoshkaError.invalidDimension(requested: dim, available: embedding.count)
    }
    var v = Array(embedding[0..<dim])
    var norm: Float = 0
    for x in v { norm += x * x }
    norm = max(norm.squareRoot(), 1e-12)
    for i in 0..<dim { v[i] /= norm }
    return v
}

public func cosine(_ a: [Float], _ b: [Float]) -> Double {
    let n = min(a.count, b.count)
    var dot = 0.0, na = 0.0, nb = 0.0
    for i in 0..<n { dot += Double(a[i]) * Double(b[i]); na += Double(a[i]) * Double(a[i]); nb += Double(b[i]) * Double(b[i]) }
    return (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : .nan
}
