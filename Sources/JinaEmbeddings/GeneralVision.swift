import AVFoundation
import CoreML
import Foundation

/// Extract evenly-spaced frames from a video file as square RGB buffers, for `embed(videoURL:)`.
/// NOTE: the model's reference samples frames by fps from metadata; this takes a fixed count evenly
/// across the duration — a reasonable, deterministic sampling, but NOT bit-matched to the processor
/// (a documented sampling caveat). Frames are resized to `size`×`size` (CoreGraphics; the resample
/// caveat applies). `count` must be even (temporal_patch_size=2 pairs consecutive frames).
public enum JinaVideoFile {
    public enum DecodeError: Error, Equatable {
        case noVideoTrack
        case oddFrameCount(Int)
        case badFrameSize(size: Int, factor: Int)
        case frameFailed(Int)
    }

    public static func extractSquareFrames(_ url: URL, count: Int, size: Int,
                                           preprocessor: JinaImagePreprocessor) async throws -> [[UInt8]] {
        guard count > 0, count % 2 == 0 else { throw DecodeError.oddFrameCount(count) }
        let factor = preprocessor.patch * preprocessor.merge
        guard size > 0, size % factor == 0 else {
            throw DecodeError.badFrameSize(size: size, factor: factor)
        }
        let asset = AVURLAsset(url: url)
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = .zero
        let duration: CMTime
        do {
            duration = try await asset.load(.duration)
        } catch {
            throw DecodeError.noVideoTrack
        }
        let seconds = duration.seconds.isFinite && duration.seconds > 0 ? duration.seconds : 0
        guard seconds > 0 else { throw DecodeError.noVideoTrack }
        var frames = [[UInt8]]()
        frames.reserveCapacity(count)
        for i in 0..<count {
            let frac = count == 1 ? 0 : Double(i) / Double(count - 1)
            let t = CMTime(seconds: seconds * frac, preferredTimescale: 600)
            do {
                let (cg, _) = try await gen.image(at: t)
                frames.append(try preprocessor.resizedRGB(cg, w: size, h: size))
            } catch { throw DecodeError.frameFailed(i) }
        }
        return frames
    }
}

/// Host-side position computation for the runtime-position ViT — the Swift port of Qwen3VL's
/// `get_vision_position_ids` + `get_vision_bilinear_indices_and_weights` + vision RoPE. For a patch
/// grid (gh, gw) it produces, in the merger's 2×2-block order, the bilinear-interpolated `posEmbeds`
/// (N·hidden) and the rotary `cos`/`sin` (N·ropeDim), N = gh·gw. These feed the converted masked ViT
/// (which also takes pixel_values + an attn_bias masking padding patches).
public struct VisionPositions {
    public let numGridPerSide: Int   // 48 (sqrt of the learned pos_embed table rows)
    public let hidden: Int           // 1024
    public let mergeSize: Int        // 2
    public let patchSize: Int        // 16
    let posTable: [Float]            // (numGridPerSide^2 * hidden) row-major
    let invFreq: [Float]             // (rope_inv_freq_len,) e.g. 16
    /// rope dim = 2 positional axes × invFreq × 2 (the cat([rot,rot]) doubling) = 64.
    public var ropeDim: Int { invFreq.count * 4 }

    public struct Meta: Decodable {
        public let num_grid_per_side: Int; public let hidden: Int; public let spatial_merge_size: Int
        public let patch_size: Int; public let rope_theta: Double
    }

    public enum PosError: Error { case badTable }

    public init(metaURL: URL, posTableURL: URL, invFreqURL: URL) throws {
        let m = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        numGridPerSide = m.num_grid_per_side; hidden = m.hidden
        mergeSize = m.spatial_merge_size; patchSize = m.patch_size
        posTable = try loadF32(posTableURL)
        invFreq = try loadF32(invFreqURL)
        guard posTable.count == numGridPerSide * numGridPerSide * hidden else { throw VisionPositions.PosError.badTable }
    }

    /// linspace(0, side-1, n) matching torch.linspace.
    private func linspace(_ n: Int) -> [Float] {
        if n <= 1 { return [0] }
        let step = Float(numGridPerSide - 1) / Float(n - 1)
        return (0..<n).map { Float($0) * step }
    }

    /// Returns merge-ordered (posEmbeds, cos, sin) and the merged-token count for grid (gh,gw).
    public func compute(gh: Int, gw: Int) -> (posEmbeds: [Float], cos: [Float], sin: [Float], merged: Int) {
        let m = mergeSize, side = numGridPerSide, N = gh * gw, D = ropeDim, K = invFreq.count
        let hGrid = linspace(gh), wGrid = linspace(gw)
        var posEmbeds = [Float](repeating: 0, count: N * hidden)
        var cosA = [Float](repeating: 0, count: N * D)
        var sinA = [Float](repeating: 0, count: N * D)
        posTable.withUnsafeBufferPointer { tbl in
            var t = 0
            for bi in 0..<(gh / m) {
                for bj in 0..<(gw / m) {
                    for pi in 0..<m {
                        for pj in 0..<m {
                            let hp = bi * m + pi, wp = bj * m + pj
                            // bilinear interpolation of the pos_embed table at (hGrid[hp], wGrid[wp])
                            let hg = hGrid[hp], wg = wGrid[wp]
                            let hf = Int(hg), wf = Int(wg)
                            let hc = Swift.min(hf + 1, side - 1), wc = Swift.min(wf + 1, side - 1)
                            let hfr = hg - Float(hf), wfr = wg - Float(wf)
                            let w0 = (1 - hfr) * (1 - wfr), w1 = (1 - hfr) * wfr
                            let w2 = hfr * (1 - wfr), w3 = hfr * wfr
                            let c0 = (hf * side + wf) * hidden, c1 = (hf * side + wc) * hidden
                            let c2 = (hc * side + wf) * hidden, c3 = (hc * side + wc) * hidden
                            let base = t * hidden
                            for d in 0..<hidden {
                                posEmbeds[base + d] = w0 * tbl[c0 + d] + w1 * tbl[c1 + d]
                                    + w2 * tbl[c2 + d] + w3 * tbl[c3 + d]
                            }
                            // RoPE: rot = [hp·invF (K), wp·invF (K)]; emb = [rot, rot] (2K each axis)
                            let rb = t * D, half = 2 * K
                            for k in 0..<K {
                                let fh = Float(hp) * invFreq[k], fw = Float(wp) * invFreq[k]
                                let ch = cosf(fh), cw = cosf(fw), sh = sinf(fh), sw = sinf(fw)
                                cosA[rb + k] = ch; cosA[rb + K + k] = cw
                                cosA[rb + half + k] = ch; cosA[rb + half + K + k] = cw
                                sinA[rb + k] = sh; sinA[rb + K + k] = sw
                                sinA[rb + half + k] = sh; sinA[rb + half + K + k] = sw
                            }
                            t += 1
                        }
                    }
                }
            }
        }
        return (posEmbeds, cosA, sinA, (gh / m) * (gw / m))
    }

    /// VIDEO: `t` frames share the same spatial positions (no temporal RoPE — verified), so tile the
    /// single-frame block `t` times. Returns the tiled (posEmbeds, cos, sin) and merged = t·(gh/2)(gw/2).
    public func computeVideo(t: Int, gh: Int, gw: Int) -> (posEmbeds: [Float], cos: [Float], sin: [Float], merged: Int) {
        let (pe, cv, sv, merged1) = compute(gh: gh, gw: gw)
        func tile(_ a: [Float]) -> [Float] { var o = [Float](); o.reserveCapacity(a.count * t); for _ in 0..<t { o.append(contentsOf: a) }; return o }
        return (tile(pe), tile(cv), tile(sv), merged1 * t)
    }
}

/// Runtime-position masked ViT (`vision_tower_masked_multifunc`): per N_max patch-bucket f{N} taking
/// pixel_values (N,1536) + pos_embeds (N,hidden) + rope cos/sin (N,ropeDim) + attn_bias (1,1,1,N).
/// The host pads the real L patches to N and masks the padding; output is the full-layout merged
/// features (N/4, 1024), truncated by the caller to the real merged count. Runs on GPU (fp16 accum).
public final class VisionCoreMLEncoderMasked: @unchecked Sendable {
    public static let patchBuckets = [1024, 1600, 2304, 3072, 4032]
    public static let maxPatchBucket = 4032
    public static let neg: Float = -1e4
    let compiledURL: URL
    let computeUnits: MLComputeUnits
    let lock = NSLock()   // guards the lazy cache for concurrent use
    var models: [Int: MLModel] = [:]

    public init(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        self.computeUnits = computeUnits
        compiledURL = modelURL.pathExtension == "mlmodelc" ? modelURL : try MLModel.compileModel(at: modelURL)
    }

    public enum EncoderError: Error { case noOutput, tooLarge(Int), inputTooLarge(actual: Int, capacity: Int) }

    public static func bucket(forPatches L: Int) -> Int { patchBuckets.first { $0 >= L } ?? maxPatchBucket }

    private func model(_ N: Int) throws -> MLModel {
        lock.lock(); defer { lock.unlock() }
        if let m = models[N] { return m }
        let cfg = MLModelConfiguration(); cfg.computeUnits = computeUnits; cfg.functionName = "f\(N)"
        let m = try MLModel(contentsOf: compiledURL, configuration: cfg)
        models[N] = m; return m
    }

    /// `pixelValues` = (L·pixelDim), `posEmbeds` = (L·hidden), `cos`/`sin` = (L·ropeDim). Returns the
    /// full-layout merged features (N/4 · 1024); caller keeps the first `merged` tokens.
    public func encode(pixelValues: [Float], pixelDim: Int, posEmbeds: [Float], hidden: Int,
                       cos: [Float], sin: [Float], ropeDim: Int, patches L: Int) throws -> [Float] {
        let N = Self.bucket(forPatches: L)
        guard L <= N else { throw EncoderError.tooLarge(L) }
        let pv = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: pixelDim)], dataType: .float32)
        let pe = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: hidden)], dataType: .float32)
        let cv = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: ropeDim)], dataType: .float32)
        let sv = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: ropeDim)], dataType: .float32)
        // Vision attention is full -> a (1,1,1,N) key-padding mask (N floats, not N²) suffices.
        let ab = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: N)], dataType: .float32)
        func fill(_ a: MLMultiArray, _ src: [Float]) throws {
            guard src.count <= a.count else {
                throw EncoderError.inputTooLarge(actual: src.count, capacity: a.count)
            }
            // MLMultiArray is NOT zero-initialized — zero the padding region (rows ≥ L) explicitly,
            // else garbage (possibly inf/NaN) poisons real tokens through the masked softmax.
            try withDenseTensor(a, as: Float.self) { ptr in
                ptr.update(repeating: 0)
                try copyFloats(src, to: ptr.baseAddress!, count: src.count)
            }
        }
        try fill(pv, pixelValues); try fill(pe, posEmbeds); try fill(cv, cos); try fill(sv, sin)
        try withDenseTensor(ab, as: Float.self) { abp in
            for j in 0..<N { abp[j] = j < L ? 0.0 : Self.neg }   // mask padding keys
        }
        let out = try model(N).prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": pv, "pos_embeds": pe, "rope_cos": cv, "rope_sin": sv, "attn_bias": ab,
        ]))
        guard let f = (out.featureValue(for: "vision_features") ?? out.featureValue(for: out.featureNames.first ?? "vision_features"))?.multiArrayValue else {
            throw EncoderError.noOutput
        }
        return try floatValues(of: f)
    }
}

/// True arbitrary-resolution image embedder: runtime-position masked ViT + the unified general
/// decoder. Given pixel_values (processor output) for grid (gh,gw), produces the L2-normalized
/// embedding. (Faithful smart-resize + patchify from a raw image is the remaining host piece;
/// `embed(pixelValues:gh:gw:)` lets the converted ViT path be used + verified independently.)
public final class JinaImageEmbedderMasked: @unchecked Sendable {
    static let prefix = JinaPromptTokens.visionPrefix
    static let suffix = JinaPromptTokens.visionSuffix
    static let imagePad = JinaPromptTokens.imagePad

    public let positions: VisionPositions
    public let encoder: VisionCoreMLEncoderMasked
    public let decoder: GeneralMediaDecoder
    public let preprocessor = JinaImagePreprocessor()

    /// `encoderUnits`: `.cpuAndGPU` (default) is the recommended choice — BOTH more accurate (fp32
    /// accumulation ~0.99996 vs the ANE's fp16 ~0.9995 end-to-end) AND much faster (measured 219ms vs
    /// 496ms for 512²; the (1,1,1,N) key-mask + masked-SDPA are not ANE-friendly). `.cpuAndNeuralEngine`
    /// runs the ViT on the ANE too (full-ANE) but is slower and less accurate — for GPU-contended cases only.
    /// `decoderUnits`: `nil` (default) = adaptive placement (ANE for S≤256, GPU for S≥512). Pass
    /// `.cpuAndNeuralEngine` together with `encoderUnits: .cpuAndNeuralEngine` for a TRUE full-ANE
    /// deployment (encoder + decoder both on the ANE) — measured end-to-end cos 0.999495, above the
    /// model's bf16 floor (lowest-power, GPU-free; slower than the hybrid default).
    public init(visionModelURL: URL, embedModelURL: URL, decoderModelURL: URL,
                metaURL: URL, posTableURL: URL, invFreqURL: URL,
                encoderUnits: MLComputeUnits = .cpuAndGPU, decoderUnits: MLComputeUnits? = nil) throws {
        positions = try VisionPositions(metaURL: metaURL, posTableURL: posTableURL, invFreqURL: invFreqURL)
        encoder = try VisionCoreMLEncoderMasked(modelURL: visionModelURL, computeUnits: encoderUnits)
        decoder = try GeneralMediaDecoder(embedModelURL: embedModelURL, decoderModelURL: decoderModelURL, computeUnits: decoderUnits)
    }

    /// Convenience: the 3 host-side resources (`meta.json`, `pos_embed_table.f32`, `rope_inv_freq.f32`,
    /// produced by `export_vision_swift_refs.py`) are loaded by name from `resourcesDir`.
    public convenience init(visionModelURL: URL, embedModelURL: URL, decoderModelURL: URL,
                            resourcesDir: URL, encoderUnits: MLComputeUnits = .cpuAndGPU,
                            decoderUnits: MLComputeUnits? = nil) throws {
        try self.init(visionModelURL: visionModelURL, embedModelURL: embedModelURL, decoderModelURL: decoderModelURL,
                      metaURL: resourcesDir.appendingPathComponent("meta.json"),
                      posTableURL: resourcesDir.appendingPathComponent("pos_embed_table.f32"),
                      invFreqURL: resourcesDir.appendingPathComponent("rope_inv_freq.f32"),
                      encoderUnits: encoderUnits, decoderUnits: decoderUnits)
    }

    /// The largest image (in mel-patch terms) the converted ViT buckets hold exactly.
    public var maxPatches: Int { VisionCoreMLEncoderMasked.maxPatchBucket }

    /// Raw RGB (h*w*3, h/w factor-aligned) -> embedding. Exact (no resample) — the full host path.
    /// Requires (h/16)*(w/16) ≤ maxPatches; use `embed(imageURL:)` for arbitrary sizes (it downscales).
    public func embed(rgb: [UInt8], h: Int, w: Int, dim: Int? = nil) throws -> [Float] {
        let (pv, gh, gw) = try preprocessor.pixelValues(rgb: rgb, h: h, w: w)
        guard gh * gw <= maxPatches else { throw VisionCoreMLEncoderMasked.EncoderError.noOutput }
        return try embed(pixelValues: pv, gh: gh, gw: gw, dim: dim)
    }

    /// Image file -> embedding for ANY resolution: smart-resize to the model grid (capped to the
    /// largest converted bucket so big images downscale gracefully instead of being unsupported),
    /// patchify, run. Non-factor-aligned native sizes carry the documented CoreGraphics resample caveat.
    public func embed(imageURL: URL, dim: Int? = nil) throws -> [Float] {
        let cg = try JinaImagePreprocessor.loadCGImage(imageURL)
        let (hbar, wbar) = preprocessor.smartResize(h: cg.height, w: cg.width, maxPixelsOverride: maxPatches * 256)
        let rgb = try preprocessor.resizedRGB(cg, w: wbar, h: hbar)
        return try embed(rgb: rgb, h: hbar, w: wbar, dim: dim)
    }

    /// `pixelValues` = (gh·gw · pixelDim) row-major (processor output). Returns the embedding.
    public func embed(pixelValues: [Float], gh: Int, gw: Int, dim: Int? = nil) throws -> [Float] {
        let shape = try preprocessor.validatedTensorShape(
            pixelValuesCount: pixelValues.count,
            t: 1,
            gh: gh,
            gw: gw
        )
        let L = shape.patchCount
        let pixelDim = shape.pixelDimension
        let (pe, cosv, sinv, merged) = positions.compute(gh: gh, gw: gw)
        let full = try encoder.encode(pixelValues: pixelValues, pixelDim: pixelDim,
                                      posEmbeds: pe, hidden: positions.hidden,
                                      cos: cosv, sin: sinv, ropeDim: positions.ropeDim, patches: L)
        let used = Array(full[0 ..< (merged * 1024)])
        let ids = Self.prefix + Array(repeating: Self.imagePad, count: merged) + Self.suffix
        let emb = try decoder.decode(tokenIds: ids, features: used, scatterOffset: Self.prefix.count)
        if let d = dim { return try matryoshka(emb, dim: d) }
        return emb
    }
}

/// VIDEO ViT encoder (`vision_tower_video_multifunc`): per-N f{N} taking pixel_values (N,1536) +
/// pos_embeds + rope cos/sin + a dense (1,1,N,N) attn_bias that the host builds PER-FRAME block-
/// diagonal (each frame's gh·gw patches attend within the frame, cf. docs/VIDEO_PATH.md) + key-pad.
/// Output is the full-layout merged features (N/4, 1024); caller truncates to the real merged count.
public final class VideoCoreMLEncoderMasked: @unchecked Sendable {
    public static let patchBuckets = [256, 512, 1024, 2048]
    public static let maxPatchBucket = 2048
    public static let neg: Float = -1e4
    let compiledURL: URL
    let computeUnits: MLComputeUnits
    let lock = NSLock()   // guards the lazy cache for concurrent use
    var models: [Int: MLModel] = [:]

    public init(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        self.computeUnits = computeUnits
        compiledURL = modelURL.pathExtension == "mlmodelc" ? modelURL : try MLModel.compileModel(at: modelURL)
    }

    public enum EncoderError: Error { case noOutput, tooLarge(Int) }
    public static func bucket(forPatches L: Int) -> Int { patchBuckets.first { $0 >= L } ?? maxPatchBucket }

    private func model(_ N: Int) throws -> MLModel {
        lock.lock(); defer { lock.unlock() }
        if let m = models[N] { return m }
        let cfg = MLModelConfiguration(); cfg.computeUnits = computeUnits; cfg.functionName = "f\(N)"
        let m = try MLModel(contentsOf: compiledURL, configuration: cfg)
        models[N] = m; return m
    }

    /// `t` frames, `fp` = patches per frame (gh·gw); L = t·fp. Returns full-layout features (N/4·1024).
    public func encode(pixelValues: [Float], pixelDim: Int, posEmbeds: [Float], hidden: Int,
                       cos: [Float], sin: [Float], ropeDim: Int, frames t: Int, framePatches fp: Int) throws -> [Float] {
        let L = t * fp
        let N = Self.bucket(forPatches: L)
        guard L <= N else { throw EncoderError.tooLarge(L) }
        let pv = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: pixelDim)], dataType: .float32)
        let pe = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: hidden)], dataType: .float32)
        let cv = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: ropeDim)], dataType: .float32)
        let sv = try MLMultiArray(shape: [NSNumber(value: N), NSNumber(value: ropeDim)], dataType: .float32)
        let ab = try MLMultiArray(shape: [1, 1, NSNumber(value: N), NSNumber(value: N)], dataType: .float32)
        func fill(_ a: MLMultiArray, _ src: [Float]) throws {
            try withDenseTensor(a, as: Float.self) { ptr in
                ptr.update(repeating: 0)
                try copyFloats(src, to: ptr.baseAddress!, count: src.count)
            }
        }
        try fill(pv, pixelValues); try fill(pe, posEmbeds); try fill(cv, cos); try fill(sv, sin)
        try withDenseTensor(ab, as: Float.self) { abp in
            for i in 0..<(N * N) { abp[i] = Self.neg }
            for f in 0..<t {   // per-frame block-diagonal: rows/cols [f·fp, f·fp+fp)
                for i in (f * fp)..<((f + 1) * fp) {
                    let row = i * N
                    for j in (f * fp)..<((f + 1) * fp) { abp[row + j] = 0.0 }
                }
            }
        }
        let out = try model(N).prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": pv, "pos_embeds": pe, "rope_cos": cv, "rope_sin": sv, "attn_bias": ab,
        ]))
        guard let fr = (out.featureValue(for: "vision_features") ?? out.featureValue(for: out.featureNames.first ?? "vision_features"))?.multiArrayValue else {
            throw EncoderError.noOutput
        }
        return try floatValues(of: fr)
    }
}

/// On-device VIDEO embedder: per-frame block-diagonal ViT + general decoder. `embed(pixelValues:...)`
/// takes the video processor's pixel_values (frame patchify + AVFoundation decoding are deferred —
/// see docs/VIDEO_PATH.md) and runs the full ViT→decoder path. Video prompt uses <|video_pad|> 151656.
public final class JinaVideoEmbedderMasked: @unchecked Sendable {
    static let prefix = JinaPromptTokens.visionPrefix
    static let suffix = JinaPromptTokens.visionSuffix
    static let videoPad = JinaPromptTokens.videoPad

    public let positions: VisionPositions
    public let encoder: VideoCoreMLEncoderMasked
    public let decoder: GeneralMediaDecoder
    public let preprocessor = JinaImagePreprocessor()

    /// `encoderUnits`/`decoderUnits` as `JinaImageEmbedderMasked`: default = GPU ViT + adaptive decoder;
    /// pass both as `.cpuAndNeuralEngine` for a true full-ANE deployment (above the bf16 floor, slower).
    public init(visionModelURL: URL, embedModelURL: URL, decoderModelURL: URL,
                metaURL: URL, posTableURL: URL, invFreqURL: URL,
                encoderUnits: MLComputeUnits = .cpuAndGPU, decoderUnits: MLComputeUnits? = nil) throws {
        positions = try VisionPositions(metaURL: metaURL, posTableURL: posTableURL, invFreqURL: invFreqURL)
        encoder = try VideoCoreMLEncoderMasked(modelURL: visionModelURL, computeUnits: encoderUnits)
        decoder = try GeneralMediaDecoder(embedModelURL: embedModelURL, decoderModelURL: decoderModelURL, computeUnits: decoderUnits)
    }

    /// Convenience: vision resources loaded by name from `resourcesDir` (see `export_vision_swift_refs.py`).
    public convenience init(visionModelURL: URL, embedModelURL: URL, decoderModelURL: URL,
                            resourcesDir: URL, encoderUnits: MLComputeUnits = .cpuAndGPU,
                            decoderUnits: MLComputeUnits? = nil) throws {
        try self.init(visionModelURL: visionModelURL, embedModelURL: embedModelURL, decoderModelURL: decoderModelURL,
                      metaURL: resourcesDir.appendingPathComponent("meta.json"),
                      posTableURL: resourcesDir.appendingPathComponent("pos_embed_table.f32"),
                      invFreqURL: resourcesDir.appendingPathComponent("rope_inv_freq.f32"),
                      encoderUnits: encoderUnits, decoderUnits: decoderUnits)
    }

    /// Raw RGB frames (count = 2·t, each h*w*3, h/w factor-aligned) -> embedding. The full host path:
    /// frame-patchify -> block-diagonal ViT -> general decoder. (Frame *sampling* is the caller's job.)
    public func embed(frames: [[UInt8]], h: Int, w: Int, dim: Int? = nil) throws -> [Float] {
        let (pv, t, gh, gw) = try preprocessor.videoPixelValues(frames: frames, h: h, w: w)
        return try embed(pixelValues: pv, t: t, gh: gh, gw: gw, dim: dim)
    }

    /// Video file -> embedding: extracts `frameCount` evenly-spaced frames (must be even), resizes each
    /// to `frameSize`² (factor-aligned), and runs the validated frame path. `frameCount/2 · (frameSize/16)²`
    /// must fit a video ViT bucket (≤2048 patches). The fps frame-sampling is best-effort, NOT bit-
    /// matched to the reference processor (documented caveat); the frame→embedding path is validated.
    public func embed(videoURL: URL, frameCount: Int = 8, frameSize: Int = 256, dim: Int? = nil) async throws -> [Float] {
        let frames = try await JinaVideoFile.extractSquareFrames(videoURL, count: frameCount, size: frameSize, preprocessor: preprocessor)
        return try embed(frames: frames, h: frameSize, w: frameSize, dim: dim)
    }

    /// `pixelValues` = (t·gh·gw · pixelDim) row-major (video processor output); grid (t,gh,gw).
    public func embed(pixelValues: [Float], t: Int, gh: Int, gw: Int, dim: Int? = nil) throws -> [Float] {
        let shape = try preprocessor.validatedTensorShape(
            pixelValuesCount: pixelValues.count,
            t: t,
            gh: gh,
            gw: gw
        )
        let fp = shape.framePatchCount
        let pixelDim = shape.pixelDimension
        let (pe, cosv, sinv, merged) = positions.computeVideo(t: t, gh: gh, gw: gw)
        let full = try encoder.encode(pixelValues: pixelValues, pixelDim: pixelDim, posEmbeds: pe,
                                      hidden: positions.hidden, cos: cosv, sin: sinv,
                                      ropeDim: positions.ropeDim, frames: t, framePatches: fp)
        let used = Array(full[0 ..< (merged * 1024)])
        let ids = Self.prefix + Array(repeating: Self.videoPad, count: merged) + Self.suffix
        let emb = try decoder.decode(tokenIds: ids, features: used, scatterOffset: Self.prefix.count)
        if let d = dim { return try matryoshka(emb, dim: d) }
        return emb
    }
}
