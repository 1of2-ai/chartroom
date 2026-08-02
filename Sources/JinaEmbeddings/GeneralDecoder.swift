import CoreML
import Foundation

/// Prompt-token ids shared by every media embedder: the chat prefix `<|im_start|>user\n`
/// plus the modality start marker, the matching `<|modality_end|><|im_end|>\n` suffix, and
/// the per-modality pad token that encoder features are scattered over. One copy — these ids
/// must agree with the converted models, so they cannot be allowed to drift per type.
enum JinaPromptTokens {
    static let visionPrefix: [Int32] = [151644, 872, 198, 151652]   // ...<|vision_start|>
    static let visionSuffix: [Int32] = [151653, 151645, 198]        // <|vision_end|><|im_end|>\n
    static let imagePad: Int32 = 151655
    static let videoPad: Int32 = 151656
    static let audioPrefix: [Int32] = [151644, 872, 198, 151670]    // ...<|audio_start|>
    static let audioSuffix: [Int32] = [151671, 151645, 198]         // <|audio_end|><|im_end|>\n
    static let audioPad: Int32 = 151669
}

/// General media decoder: the unified `embed_multifunc` (input_ids -> inputs_embeds) plus
/// `decoder_embeds_multifunc` (inputs_embeds + position_ids + selector -> L2-normed embedding),
/// each a multi-function package over sequence-length buckets S ∈ {128,256,512,1024}.
///
/// The host builds `input_ids = [prefix, <media_pad>*L, suffix]` right-padded to the chosen S,
/// embeds them, scatters the L real media features (image/audio/...) into the media positions,
/// then decodes with a selector at the real last token. Causal attention + right padding means
/// the trailing pad never affects the pooled token, so ANY L ≤ S-overhead gets exact parity.
/// This is the single decoder shared by text/image/audio/omni (replaces fixed per-size decoders).
public final class GeneralMediaDecoder: @unchecked Sendable {
    public static let sBuckets = [128, 256, 512, 1024]
    public static let maxSequenceBucket = 1024
    public static let padTokenID: Int32 = 151643

    let embedCompiled: URL
    let decoderCompiled: URL
    /// `nil` = ADAPTIVE placement (measured optimal): ANE for small sequences, GPU for large. The
    /// decoder is shallow enough to be accurate on either, so placement is a pure latency choice and
    /// there's a crossover near S≈384 (ANE: 13/30/85/231 ms vs GPU: 18/31/51/132 ms at S=128/256/512/
    /// 1024). A non-nil value forces all buckets onto that unit (e.g. .cpuAndNeuralEngine for lowest
    /// power, .cpuAndGPU for lowest latency on large media).
    let forcedUnits: MLComputeUnits?
    let featDim: Int
    // Per-S loaded functions (loading is the expensive part; cache once used). `lock` guards the
    // lazy caches so a shared decoder is safe under concurrent embed() calls (predictions run
    // outside the lock — MLModel.prediction is itself thread-safe).
    let lock = NSLock()
    var embedModels: [Int: MLModel] = [:]
    var decoderModels: [Int: MLModel] = [:]

    public init(embedModelURL: URL, decoderModelURL: URL,
                computeUnits: MLComputeUnits? = nil, featDim: Int = 1024) throws {
        self.forcedUnits = computeUnits
        self.featDim = featDim
        embedCompiled = embedModelURL.pathExtension == "mlmodelc"
            ? embedModelURL : try MLModel.compileModel(at: embedModelURL)
        decoderCompiled = decoderModelURL.pathExtension == "mlmodelc"
            ? decoderModelURL : try MLModel.compileModel(at: decoderModelURL)
    }

    public enum DecoderError: Error { case noOutput, tooLong(Int), invalidFeatures }

    public static func sBucket(forSeq n: Int) -> Int {
        sBuckets.first { $0 >= n } ?? maxSequenceBucket
    }

    /// Latency-optimal unit for a bucket: ANE ≤256, GPU ≥512 (measured crossover near S≈384).
    func units(forS S: Int) -> MLComputeUnits { forcedUnits ?? (S <= 256 ? .cpuAndNeuralEngine : .cpuAndGPU) }

    private func embedModel(_ S: Int) throws -> MLModel {
        try cachedModel(S, in: \.embedModels, from: embedCompiled)
    }

    private func decoderModel(_ S: Int) throws -> MLModel {
        try cachedModel(S, in: \.decoderModels, from: decoderCompiled)
    }

    /// Double-checked: the multi-second MLModel load happens outside the lock so a
    /// concurrent decode whose bucket is already resident never stalls behind it.
    /// On a race, the first resident model wins and the duplicate is discarded.
    private func cachedModel(
        _ S: Int,
        in path: ReferenceWritableKeyPath<GeneralMediaDecoder, [Int: MLModel]>,
        from compiled: URL
    ) throws -> MLModel {
        lock.lock()
        if let m = self[keyPath: path][S] {
            lock.unlock()
            return m
        }
        lock.unlock()

        let cfg = MLModelConfiguration(); cfg.computeUnits = units(forS: S); cfg.functionName = "f\(S)"
        let built = try MLModel(contentsOf: compiled, configuration: cfg)

        lock.lock(); defer { lock.unlock() }
        if let winner = self[keyPath: path][S] { return winner }
        self[keyPath: path][S] = built
        return built
    }

    /// Release every resident bucket model. The next decode reloads what it needs.
    public func unloadAll() {
        lock.lock(); defer { lock.unlock() }
        embedModels.removeAll()
        decoderModels.removeAll()
    }

    /// `tokenIds` = the full real sequence (prefix + media pads + suffix), length ≤ S.
    /// `features` = (L * featDim) row-major, scattered into rows [scatterOffset, scatterOffset+L).
    /// Returns the L2-normalized embedding.
    public func decode(tokenIds: [Int32], features: [Float], scatterOffset: Int) throws -> [Float] {
        let realLen = tokenIds.count
        let S = Self.sBucket(forSeq: realLen)
        guard realLen <= S else { throw DecoderError.tooLong(realLen) }
        guard featDim > 0, features.count.isMultiple(of: featDim) else { throw DecoderError.invalidFeatures }
        let L = features.count / featDim
        guard scatterOffset >= 0, scatterOffset + L <= realLen else { throw DecoderError.invalidFeatures }

        // 1) embed_multifunc: input_ids (1,S) -> inputs_embeds (1,S,featDim)
        let ids = try MLMultiArray(shape: [1, NSNumber(value: S)], dataType: .int32)
        try withDenseTensor(ids, as: Int32.self) { idp in
            for i in 0..<S { idp[i] = i < realLen ? tokenIds[i] : Self.padTokenID }
        }
        let embOut = try embedModel(S).prediction(from: MLDictionaryFeatureProvider(dictionary: ["input_ids": ids]))
        guard let embArr = (embOut.featureValue(for: "out") ?? embOut.featureValue(for: embOut.featureNames.first ?? "out"))?.multiArrayValue,
              embArr.count == S * featDim else {
            throw DecoderError.noOutput
        }

        // 2) copy embeds into a fresh (1,S,featDim) array and scatter the media features
        let embeds = try MLMultiArray(shape: [1, NSNumber(value: S), NSNumber(value: featDim)], dataType: .float32)
        try withFloat32Tensor(embArr) { src in
            try withDenseTensor(embeds, as: Float.self) { ep in
                ep.baseAddress!.update(from: src.baseAddress!, count: S * featDim)
                try copyFloats(features, to: ep.baseAddress! + scatterOffset * featDim, count: features.count)
            }
        }

        // 3) decoder_embeds_multifunc: inputs_embeds + position_ids (3,1,S) + selector (1,S)
        let pos = try MLMultiArray(shape: [3, 1, NSNumber(value: S)], dataType: .int32)
        let sel = try MLMultiArray(shape: [1, NSNumber(value: S)], dataType: .float32)
        let last = realLen - 1
        try withDenseTensor(pos, as: Int32.self) { pp in
            for i in 0..<S { pp[i] = Int32(i); pp[S + i] = Int32(i); pp[2 * S + i] = Int32(i) }
        }
        try withDenseTensor(sel, as: Float.self) { sp in
            for i in 0..<S { sp[i] = (i == last) ? 1.0 : 0.0 }
        }
        let out = try decoderModel(S).prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "inputs_embeds": embeds, "position_ids": pos, "selector": sel,
        ]))
        guard let e = (out.featureValue(for: "embedding") ?? out.featureValue(for: out.featureNames.first ?? "embedding"))?.multiArrayValue else {
            throw DecoderError.noOutput
        }
        return try floatValues(of: e)
    }
}

/// Host-built masks for the runtime-masked audio encoder. Computed from a clip's real frame count
/// so a partial boundary chunk (and any fully-silent chunks beyond it) are masked out — giving exact
/// reference parity at ANY length, not just 200-frame multiples.
public struct AudioMasks {
    public static let chunkSize = 200      // n_window*2 mel frames per chunk
    public static let tpc = 100            // attention tokens per chunk (after conv1 stride-2)
    public static let neg: Float = -1e4    // fp16-safe additive mask

    public let bucketFrames: Int           // F (a converted bucket >= exactFrames)
    public let chunks: Int                  // C = F/200
    public let convMask: [Float]            // (C,1,200) row-major
    public let attnBias: [Float]            // (1,1,T,T) row-major, T = C*tpc
    public let realTokens: Int              // pooled token count to keep (= reference length)

    /// Round up to the next converted frame bucket; extra chunks are fully masked.
    public static func bucket(forFrames n: Int, buckets: [Int] = AudioCoreMLEncoderMasked.frameBuckets) -> Int {
        if let bucket = buckets.first(where: { $0 >= n }) { return bucket }
        if let largestBucket = buckets.last { return largestBucket }
        return n
    }

    public init(exactFrames: Int, bucketFrames F: Int) {
        bucketFrames = F
        let cs = Self.chunkSize, tpc = Self.tpc
        let C = F / cs; chunks = C
        let T = C * tpc
        let fullChunks = exactFrames / cs
        let rem = exactFrames - fullChunks * cs

        var cm = [Float](repeating: 0, count: C * cs)
        var rtok = [Int](repeating: 0, count: C)
        for c in 0..<C {
            let real = c < fullChunks ? cs : (c == fullChunks ? rem : 0)
            if real > 0 { for i in 0..<real { cm[c * cs + i] = 1.0 } }
            rtok[c] = real > 0 ? (real - 1) / 2 + 1 : 0
        }
        convMask = cm

        var bias = [Float](repeating: Self.neg, count: T * T)
        for c in 0..<C {
            let r = rtok[c]; if r == 0 { continue }
            let base = c * tpc
            for i in 0..<tpc {
                let row = (base + i) * T
                for j in 0..<r { bias[row + base + j] = 0.0 }
            }
        }
        attnBias = bias
        // reference pooled-token count: num_pooled = ((after_conv1) - 2)/2 + 1, after_conv1=(n-1)/2+1
        let afterConv1 = (exactFrames - 1) / 2 + 1
        realTokens = (afterConv1 - 2) / 2 + 1
    }
}

/// Runtime-masked audio encoder (`audio_tower_masked_multifunc`): per-bucket f{F} taking
/// packed_mel (nMels,F) + conv_mask (C,1,200) + attn_bias (1,1,T,T), returning the full-layout
/// pooled features (C*50, 1024). With host masks this matches the reference at ANY length ≤ 16 s.
/// Runs on GPU (fp16 matmul accumulation).
public final class AudioCoreMLEncoderMasked: @unchecked Sendable {
    /// audio_tower_masked_multifunc package functions: 2/4/8/16/32 s (32 s covers the model's
    /// 30 s / 3000-frame Whisper limit). Separate from AudioCoreMLEncoderMF (the non-masked package).
    public static let frameBuckets = [200, 400, 800, 1600, 3200]
    let compiledURL: URL
    let computeUnits: MLComputeUnits
    let lock = NSLock()   // guards the lazy cache for concurrent use
    var models: [Int: MLModel] = [:]

    public init(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        self.computeUnits = computeUnits
        compiledURL = modelURL.pathExtension == "mlmodelc" ? modelURL : try MLModel.compileModel(at: modelURL)
    }

    public enum EncoderError: Error { case noOutput }

    private func model(_ F: Int) throws -> MLModel {
        lock.lock(); defer { lock.unlock() }
        if let m = models[F] { return m }
        let cfg = MLModelConfiguration(); cfg.computeUnits = computeUnits; cfg.functionName = "f\(F)"
        let m = try MLModel(contentsOf: compiledURL, configuration: cfg)
        models[F] = m; return m
    }

    /// Returns the full-layout features (C*50 * 1024) row-major; caller truncates to `realTokens`.
    public func encode(packedMel: [Float], nMels: Int, masks: AudioMasks) throws -> [Float] {
        let F = masks.bucketFrames, C = masks.chunks
        let cs = AudioMasks.chunkSize, T = C * AudioMasks.tpc
        let pk = try MLMultiArray(shape: [NSNumber(value: nMels), NSNumber(value: F)], dataType: .float32)
        let cm = try MLMultiArray(shape: [NSNumber(value: C), 1, NSNumber(value: cs)], dataType: .float32)
        let ab = try MLMultiArray(shape: [1, 1, NSNumber(value: T), NSNumber(value: T)], dataType: .float32)
        try withDenseTensor(pk, as: Float.self) { try copyFloats(packedMel, to: $0.baseAddress!, count: packedMel.count) }
        try withDenseTensor(cm, as: Float.self) { try copyFloats(masks.convMask, to: $0.baseAddress!, count: masks.convMask.count) }
        try withDenseTensor(ab, as: Float.self) { try copyFloats(masks.attnBias, to: $0.baseAddress!, count: masks.attnBias.count) }
        let out = try model(F).prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "packed_mel": pk, "conv_mask": cm, "attn_bias": ab,
        ]))
        guard let f = (out.featureValue(for: "audio_features") ?? out.featureValue(for: out.featureNames.first ?? "audio_features"))?.multiArrayValue else {
            throw EncoderError.noOutput
        }
        return try floatValues(of: f)
    }
}
