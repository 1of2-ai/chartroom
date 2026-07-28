import AVFoundation
import CoreML
import Foundation
import Testing
@testable import JinaEmbeddings

/// Transparency handling (no model) — CHARACTERIZATION of a documented caveat. The reference loads
/// images via PIL `Image.open(...).convert("RGB")`, which DROPS alpha keeping the raw RGB. The Swift
/// path resizes through a premultiplied CGContext, so a fully-transparent pixel composites to BLACK
/// (0,0,0) rather than keeping its stored RGB. This is a real but narrow divergence (only RGBA inputs
/// with actual transparency; the RGB *under* a transparent pixel is semantically arbitrary), documented
/// in README "Limitations". A faithful fix is hard (CGContext is premultiplied-only; PIL works in
/// non-premultiplied space) and would risk the validated opaque-image parity — so it's documented, not
/// papered over. This test pins the behavior so any future change to it is intentional.
@Test func transparentImageAlphaHandling() throws {
    let w = 32, h = 32
    var px = [UInt8](repeating: 0, count: w * h * 4)
    for i in 0..<(w * h) { px[i*4] = 200; px[i*4+1] = 100; px[i*4+2] = 50; px[i*4+3] = 0 }  // non-premultiplied RGBA, alpha=0
    let cs = CGColorSpaceCreateDeviceRGB()
    let provider = try #require(CGDataProvider(data: Data(px) as CFData))
    let cg = try #require(CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                                  space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    let rgb = try JinaImagePreprocessor().resizedRGB(cg, w: w, h: h)
    let c = (16 * w + 16) * 3   // a center pixel
    // Documented behavior: premultiplied compositing -> transparent pixel becomes black (NOT PIL's (200,100,50)).
    #expect(rgb[c] == 0 && rgb[c+1] == 0 && rgb[c+2] == 0,
            "transparent-pixel RGB = (\(rgb[c]),\(rgb[c+1]),\(rgb[c+2])); expected premultiplied black (0,0,0)")
    // Opaque pixels are unaffected (the common case) — sanity-check the path is otherwise faithful.
    for i in 0..<(w * h) { px[i*4+3] = 255 }
    let provider2 = try #require(CGDataProvider(data: Data(px) as CFData))
    let cg2 = try #require(CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                                   space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                                   provider: provider2, decode: nil,
                                   shouldInterpolate: false, intent: .defaultIntent))
    let rgb2 = try JinaImagePreprocessor().resizedRGB(cg2, w: w, h: h)
    #expect(rgb2[c] == 200 && rgb2[c+1] == 100 && rgb2[c+2] == 50, "opaque RGBA pixel must keep its RGB")
}

/// AVFoundation audio-file decode round-trip (no model artifacts needed): write a known 16 kHz mono
/// waveform to a .wav, decode it back via JinaAudioFile, and confirm it matches — validates the
/// `embed(audioURL:)` decode path for the exact (no-resample) case.
@Test func audioFileDecodeRoundTrip() throws {
    let sr = 16000, n = sr   // 1 s
    var wave = [Float](repeating: 0, count: n)
    for i in 0..<n { wave[i] = 0.5 * sinf(2 * .pi * 220 * Float(i) / Float(sr)) }
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("jina_\(UUID().uuidString).wav")
    let fmt = try #require(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false))
    do {   // scope the write file so it flushes/closes before we read it back
        let file = try AVAudioFile(forWriting: url, settings: fmt.settings)
        let buf = try #require(AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(n)))
        buf.frameLength = AVAudioFrameCount(n)
        let channelData = try #require(buf.floatChannelData)
        for i in 0..<n { channelData[0][i] = wave[i] }
        try file.write(from: buf)
    }
    let decoded = try JinaAudioFile.decode16kMono(url)
    try? FileManager.default.removeItem(at: url)
    // decode reads the file's PCM directly (16kHz mono path = no resampling) -> sample VALUES are exact
    // on the overlap. (The count can differ slightly from the AVAudioFile write's buffering.)
    #expect(decoded.count > n * 9 / 10, "decoded \(decoded.count) of \(n)")
    var maxAbs: Float = 0
    for i in 0..<min(decoded.count, n) { maxAbs = max(maxAbs, abs(decoded[i] - wave[i])) }
    #expect(maxAbs < 1e-4, "16kHz .wav decode value match maxAbs=\(maxAbs)")
}

/// Integration tests against the converted CoreML artifacts + reference embeddings. They are
/// skipped (return early) when the artifacts haven't been built (`python/build_all.sh`), so a
/// fresh checkout's package tests still pass on the unit tests.
private let root: URL = {
    var url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    while url.path != "/" {
        if FileManager.default.fileExists(atPath: url.appendingPathComponent("reference").path),
           FileManager.default.fileExists(atPath: url.appendingPathComponent("python").path) {
            return url
        }
        url.deleteLastPathComponent()
    }
    return URL(fileURLWithPath: #filePath).deletingLastPathComponent()
}()
private func path(_ rel: String) -> URL { root.appendingPathComponent(rel) }
private func exists(_ rel: String) -> Bool { FileManager.default.fileExists(atPath: path(rel).path) }

/// Model-gated tests use `.enabled(if: artifacts(...))` so an absent artifact tree shows up
/// as a SKIPPED test in the report — a fresh checkout must not silently pass as green.
private func artifacts(_ rels: String...) -> Bool { rels.allSatisfy(exists) }
private let artifactsHint: Comment = "Core ML artifacts not built on this machine (python/build_all.sh)"

private struct TextRef: Decodable {
    let text: String; let prompt_name: String; let prompted: String
    let token_ids: [Int32]; let embedding: [String: [Float]]
}
private struct ModalRef: Decodable { let embedding_1024: [Float] }

private func textRefs() throws -> [TextRef] {
    try JSONDecoder().decode([TextRef].self, from: Data(contentsOf: path("reference/text_reference.json")))
}

@Test func bundledMelResourcesLoad() throws {
    // No model artifacts needed — just confirms the bundled mel constants load + produce output.
    let fe = try JinaMelFrontend()
    let mel = try fe.packedMel([Float](repeating: 0.1, count: 32000))
    #expect(mel.count == fe.nMels * fe.nFrames)
    #expect(mel.allSatisfy { $0.isFinite })
}

@Test(.enabled(if: artifacts("artifacts/hf/jina-v5-omni-small/tokenizer.json"), artifactsHint))
func tokenizerByteExact() async throws {
    let tok = try await JinaTokenizer(modelFolder: path("artifacts/hf/jina-v5-omni-small"))
    for r in try textRefs().prefix(6) {
        #expect(tok.encode(r.prompted) == r.token_ids)
    }
}

@Test(.enabled(if: artifacts("artifacts/coreml/text_multifunc.mlpackage"), artifactsHint))
func textEmbeddingParity() async throws {
    let embedder = try await JinaTextEmbedder(
        multiFunctionModelURL: path("artifacts/coreml/text_multifunc.mlpackage"),
        tokenizerFolder: path("artifacts/hf/jina-v5-omni-small"))
    for r in try textRefs().prefix(5) {
        let prompt: JinaTextEmbedder.Prompt = r.prompt_name == "query" ? .query : .document
        let emb = try embedder.embed(r.text, prompt: prompt)
        let reference = try #require(r.embedding["1024"])
        let c = cosine(emb, reference)
        #expect(c > 0.999, "text \"\(r.text)\" cos=\(c)")
    }
}

@Test(.enabled(if: artifacts(
    "artifacts/coreml/audio_tower_masked_multifunc.mlpackage",
    "artifacts/coreml/embed_multifunc.mlpackage",
    "reference/audio_swift/manifest_offbucket.json"
), artifactsHint))
func maskedAudioArbitraryLength() async throws {
    struct Off: Decodable { let tag: String }
    let offs = try JSONDecoder().decode([Off].self, from: Data(contentsOf: path("reference/audio_swift/manifest_offbucket.json")))
    let embedder = try JinaAudioEmbedderMasked(
        audioModelURL: path("artifacts/coreml/audio_tower_masked_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"))
    func loadF32(_ rel: String) throws -> [Float] {
        try Data(contentsOf: path(rel)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    // Off-bucket (partial-chunk) lengths are where truncate-real fails and the masked encoder must
    // match the reference — exercise 3s/5s explicitly.
    for o in offs {
        let wave = try loadF32("reference/audio_swift/off_\(o.tag)_wave.f32")
        let ref = try loadF32("reference/audio_swift/off_\(o.tag)_emb.f32")
        let c = cosine(try embedder.embed(wave), ref)
        #expect(c > 0.999, "masked audio \(o.tag) cos=\(c)")
    }
}

/// Full-ANE deployment: force BOTH the encoder AND the decoder onto the Neural Engine (the
/// `decoderUnits` knob, alongside `encoderUnits`). Regression-guards that the full-ANE path actually
/// RUNS end-to-end on the ANE (no shape/compile error), is unit-norm, and stays above the model's
/// bf16 audio floor vs the fp32 reference (documented full-ANE end-to-end ≈0.9964; the 0.99 guard
/// below leaves margin for ANE fp16 run variance). Lowest-power, GPU-free mode; slower than the
/// hybrid default, but the recovered end-to-end cos stays at/above native bf16 precision.
@Test(.enabled(if: artifacts(
    "artifacts/coreml/audio_tower_masked_multifunc.mlpackage",
    "artifacts/coreml/embed_multifunc.mlpackage",
    "reference/audio_swift/manifest_offbucket.json"
), artifactsHint))
func fullANEDeploymentRuns() throws {
    struct Off: Decodable { let tag: String }
    let offs = try JSONDecoder().decode([Off].self, from: Data(contentsOf: path("reference/audio_swift/manifest_offbucket.json")))
    func loadF32(_ rel: String) throws -> [Float] {
        try Data(contentsOf: path(rel)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    let fullANE = try JinaAudioEmbedderMasked(
        audioModelURL: path("artifacts/coreml/audio_tower_masked_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"),
        encoderUnits: .cpuAndNeuralEngine, decoderUnits: .cpuAndNeuralEngine)
    for o in offs {
        let wave = try loadF32("reference/audio_swift/off_\(o.tag)_wave.f32")
        let ref = try loadF32("reference/audio_swift/off_\(o.tag)_emb.f32")
        let emb = try fullANE.embed(wave)
        let norm = sqrt(emb.reduce(0) { $0 + $1 * $1 })
        #expect(abs(norm - 1.0) < 1e-3, "full-ANE \(o.tag) not L2-normalized (norm=\(norm))")
        // above the bf16 audio floor (0.994951); use 0.99 as the regression guard (ANE fp16 + run variance)
        let c = cosine(emb, ref)
        #expect(c > 0.99, "full-ANE audio \(o.tag) below floor (cos=\(c))")
    }
}

/// Full-ANE for the VISION path — the harder, slower encoder (key-mask masked ViT; ANE features drop
/// to ~0.9958 over 24 layers, which the decoder pool + L2 recover to ~0.9995 end-to-end). Guards that
/// the vision full-ANE pipeline (ViT + decoder both on the ANE via `encoderUnits`+`decoderUnits`) runs
/// end-to-end, is unit-norm, and stays above the vision bf16 floor (0.993308) vs the fp32 reference.
/// One grid only (the ANE ViT compile is expensive) — runs concurrently with the audio full-ANE test.
@Test(.enabled(if: artifacts(
    "artifacts/coreml/vision_tower_masked_multifunc.mlpackage",
    "artifacts/coreml/embed_multifunc.mlpackage",
    "reference/vision_swift/manifest.json"
), artifactsHint))
func fullANEVisionRuns() throws {
    struct Grid: Decodable { let tag: String; let gh: Int; let gw: Int }
    func vp(_ r: String) -> URL { path("reference/vision_swift/\(r)") }
    func f32(_ r: String) throws -> [Float] { try Data(contentsOf: vp(r)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    func u8(_ r: String) throws -> [UInt8] { try Data(contentsOf: vp(r)).withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) } }
    let grids = try JSONDecoder().decode([Grid].self, from: Data(contentsOf: vp("manifest.json")))
    guard let g = grids.first else { return }
    let fullANE = try JinaImageEmbedderMasked(
        visionModelURL: path("artifacts/coreml/vision_tower_masked_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"),
        resourcesDir: path("reference/vision_swift"),
        encoderUnits: .cpuAndNeuralEngine, decoderUnits: .cpuAndNeuralEngine)
    let emb = try fullANE.embed(rgb: try u8("g\(g.tag)_rgb.u8"), h: g.gh * 16, w: g.gw * 16)
    let norm = sqrt(emb.reduce(0) { $0 + $1 * $1 })
    #expect(abs(norm - 1.0) < 1e-3, "full-ANE vision \(g.tag) not L2-normalized (norm=\(norm))")
    let c = cosine(emb, try f32("g\(g.tag)_emb.f32"))
    #expect(c > 0.99, "full-ANE vision \(g.tag) below floor (cos=\(c))")
}

// Cheap (no CoreML): the bilinear pos_embeds + 2D RoPE + patchify match the golden exports.
@Test(.enabled(if: artifacts(
    "reference/vision_swift/manifest.json",
    "reference/vision_swift/pos_embed_table.f32"
), artifactsHint))
func visionPositionAndPatchifyPort() throws {
    struct Grid: Decodable { let tag: String; let gh: Int; let gw: Int }
    func vp(_ r: String) -> URL { path("reference/vision_swift/\(r)") }
    func f32(_ r: String) throws -> [Float] { try Data(contentsOf: vp(r)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    func u8(_ r: String) throws -> [UInt8] { try Data(contentsOf: vp(r)).withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) } }
    func maxAbs(_ a: [Float], _ b: [Float]) -> Float { var m: Float = 0; for i in 0..<min(a.count, b.count) { m = Swift.max(m, abs(a[i] - b[i])) }; return m }
    let grids = try JSONDecoder().decode([Grid].self, from: Data(contentsOf: vp("manifest.json")))
    let positions = try VisionPositions(metaURL: vp("meta.json"), posTableURL: vp("pos_embed_table.f32"), invFreqURL: vp("rope_inv_freq.f32"))
    let prep = JinaImagePreprocessor()
    for g in grids {
        let (pe, cv, sv, _) = positions.compute(gh: g.gh, gw: g.gw)
        #expect(maxAbs(pe, try f32("g\(g.tag)_pos_embeds.f32")) < 1e-4)
        #expect(maxAbs(cv, try f32("g\(g.tag)_rope_cos.f32")) < 1e-4)
        #expect(maxAbs(sv, try f32("g\(g.tag)_rope_sin.f32")) < 1e-4)
        let (pv, _, _) = try prep.pixelValues(rgb: u8("g\(g.tag)_rgb.u8"), h: g.gh * 16, w: g.gw * 16)
        #expect(maxAbs(pv, try f32("g\(g.tag)_pixel_values.f32")) < 1e-4)
    }
}

@Test(.enabled(if: artifacts(
    "artifacts/coreml/vision_tower_masked_multifunc.mlpackage",
    "artifacts/coreml/embed_multifunc.mlpackage",
    "reference/vision_swift/manifest.json"
), artifactsHint))
func visionMaskedArbitraryResolution() throws {
    struct Grid: Decodable { let tag: String; let gh: Int; let gw: Int }
    func vp(_ r: String) -> URL { path("reference/vision_swift/\(r)") }
    func f32(_ r: String) throws -> [Float] { try Data(contentsOf: vp(r)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    func u8(_ r: String) throws -> [UInt8] { try Data(contentsOf: vp(r)).withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) } }
    let grids = try JSONDecoder().decode([Grid].self, from: Data(contentsOf: vp("manifest.json")))
    // exercises the resourcesDir convenience init (vision_swift holds meta/pos_embed_table/rope_inv_freq)
    let embedder = try JinaImageEmbedderMasked(
        visionModelURL: path("artifacts/coreml/vision_tower_masked_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"),
        resourcesDir: path("reference/vision_swift"))
    // Non-square grids (40x30/30x40) are the cases the fixed-square MVP could not handle.
    for g in grids {
        let c = cosine(try embedder.embed(rgb: try u8("g\(g.tag)_rgb.u8"), h: g.gh * 16, w: g.gw * 16), try f32("g\(g.tag)_emb.f32"))
        #expect(c > 0.999, "vision \(g.tag) cos=\(c)")
    }
}

@Test(.enabled(if: artifacts(
    "artifacts/coreml/vision_tower_video_multifunc.mlpackage",
    "reference/video_swift/manifest.json",
    "reference/vision_swift/meta.json"
), artifactsHint))
func videoOnDevicePath() throws {
    struct Case: Decodable { let tag: String; let nframes: Int; let fsize: Int; let t: Int; let gh: Int; let gw: Int }
    func vd(_ r: String) -> URL { path("reference/video_swift/\(r)") }
    func rs(_ r: String) -> URL { path("reference/vision_swift/\(r)") }
    func f32(_ u: URL) throws -> [Float] { try Data(contentsOf: u).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }
    func u8(_ u: URL) throws -> [UInt8] { try Data(contentsOf: u).withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) } }
    let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: vd("manifest.json")))
    let embedder = try JinaVideoEmbedderMasked(
        visionModelURL: path("artifacts/coreml/vision_tower_video_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"),
        metaURL: rs("meta.json"), posTableURL: rs("pos_embed_table.f32"), invFreqURL: rs("rope_inv_freq.f32"))
    for c in cases {
        let ref = try f32(vd("v\(c.tag)_emb.f32"))
        // full path from raw frames (frame-patchify + block-diagonal ViT + decoder)
        let raw = try u8(vd("v\(c.tag)_frames.u8")); let fb = c.fsize * c.fsize * 3
        let frames = (0..<c.nframes).map { Array(raw[$0 * fb ..< ($0 + 1) * fb]) }
        #expect(cosine(try embedder.embed(frames: frames, h: c.fsize, w: c.fsize), ref) > 0.999, "video \(c.tag)")
    }
}

/// End-to-end `embed(videoURL:)`: a synthetic .mp4 -> default-param extraction (8f@256² -> 1024
/// patches, an exact video ViT bucket) -> real ViT + decoder. Closes the gap between the extractor
/// (videoFileExtraction) and the frame path (videoOnDevicePath): proves the DEFAULT params map to a
/// real bucket and run through the real model, and that the result equals the manual compose.
@Test(.enabled(if: artifacts(
    "artifacts/coreml/vision_tower_video_multifunc.mlpackage",
    "reference/vision_swift/meta.json"
), artifactsHint))
func videoURLEndToEnd() async throws {
    guard let url = makeSyntheticMP4(frames: 12, size: 256) else { return }   // skip if no encoder
    defer { try? FileManager.default.removeItem(at: url) }
    func rs(_ r: String) -> URL { path("reference/vision_swift/\(r)") }
    let embedder = try JinaVideoEmbedderMasked(
        visionModelURL: path("artifacts/coreml/vision_tower_video_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"),
        metaURL: rs("meta.json"), posTableURL: rs("pos_embed_table.f32"), invFreqURL: rs("rope_inv_freq.f32"))
    // default params (8 frames @ 256² -> 4·16² = 1024 patches, an exact bucket)
    let emb = try await embedder.embed(videoURL: url)
    #expect(emb.count == 1024)
    let norm = sqrt(emb.reduce(0) { $0 + $1 * $1 })
    #expect(abs(norm - 1.0) < 1e-3, "embed(videoURL:) not L2-normalized (norm=\(norm))")
    // determinism + equivalence to the manual compose (extractSquareFrames + embed(frames:))
    let manual = try embedder.embed(
        frames: try await JinaVideoFile.extractSquareFrames(url, count: 8, size: 256, preprocessor: embedder.preprocessor),
        h: 256, w: 256)
    #expect(cosine(emb, manual) > 0.99999, "embed(videoURL:) != manual compose")
}

private final class ConcurrentResults: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [[Float]] = []

    func append(_ value: [Float]) {
        lock.lock()
        storage.append(value)
        lock.unlock()
    }

    var values: [[Float]] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

// A SHARED masked-audio embedder hit from many threads at once must not race on its lazy model
// caches and must return identical, correct embeddings. (Exercises the NSLock cache guards.)
@Test(.enabled(if: artifacts(
    "artifacts/coreml/audio_tower_masked_multifunc.mlpackage",
    "reference/audio_swift/manifest_offbucket.json"
), artifactsHint))
func concurrentEmbedIsSafeAndConsistent() throws {
    let embedder = try JinaAudioEmbedderMasked(
        audioModelURL: path("artifacts/coreml/audio_tower_masked_multifunc.mlpackage"),
        embedModelURL: path("artifacts/coreml/embed_multifunc.mlpackage"),
        decoderModelURL: path("artifacts/coreml/decoder_embeds_multifunc.mlpackage"))
    func load(_ tag: String) throws -> [Float] {
        try Data(contentsOf: path("reference/audio_swift/off_\(tag)_wave.f32")).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    // mix of durations -> different frame + S buckets loaded concurrently (max cache contention)
    let waves = [try load("3s"), try load("6s"), try load("20s")]
    let serial = try waves.map { try embedder.embed($0) }
    let results = ConcurrentResults()
    DispatchQueue.concurrentPerform(iterations: 24) { i in
        if let e = try? embedder.embed(waves[i % waves.count]) { results.append(e) }
    }
    let values = results.values
    #expect(values.count == 24, "some concurrent embeds failed (\(values.count)/24)")
    for e in values {
        // every concurrent result must match one of the serial baselines exactly (no torn cache)
        #expect(serial.contains { cosine($0, e) > 0.99999 }, "concurrent embed diverged")
    }
}

/// `JinaOmniEmbedder` façade — no model artifacts needed. Verifies construction is lazy (pointing at
/// an empty directory never touches a file until you embed) and that every modality surfaces a clear
/// `JinaOmniError.missingArtifact` (naming the resolved path) instead of an opaque Core ML failure.
/// Also pins the `.fullANE` configuration preset.
@Test func omniFacadeLazyConstructionAndErrors() async throws {
    let empty = FileManager.default.temporaryDirectory.appendingPathComponent("jina_omni_\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: empty) }

    let jina = JinaOmniEmbedder(modelsDirectory: empty)   // must not throw / must not load anything
    #expect(jina.modelsDirectory == empty)
    #expect(jina.modelID == "jinaai/jina-embeddings-v5-omni-small")
    #expect(jina.embeddingDimension == 1024)

    let manifest = """
    {
      "formatVersion": 1,
      "modelID": "jinaai/jina-embeddings-v5-omni-small",
      "embeddingDimension": 1024,
      "minimumDeployment": { "macOS": "15.0", "iOS": "18.0" },
      "text": { "model": "custom_text.mlpackage", "tokenizer": "custom_tokenizer", "buckets": [16, 32] },
      "image": { "encoder": "custom_image.mlpackage", "resources": "custom_vision_resources", "patchBuckets": [1024] },
      "audio": { "encoder": "custom_audio.mlpackage", "frameBuckets": [200] },
      "video": { "encoder": "custom_video.mlpackage", "patchBuckets": [256] },
      "decoder": { "embed": "custom_embed.mlpackage", "model": "custom_decoder.mlpackage", "sequenceBuckets": [128] }
    }
    """
    let manifestData = try #require(manifest.data(using: .utf8))
    try manifestData.write(to: empty.appendingPathComponent("manifest.json"))
    let bundle = try JinaModelBundle(url: empty)
    let fromBundle = JinaOmniEmbedder(bundle: bundle)
    #expect(fromBundle.modelsDirectory == empty)
    #expect(fromBundle.configuration.textModel == "custom_text.mlpackage")
    #expect(fromBundle.configuration.tokenizerFolder == "custom_tokenizer")
    #expect(fromBundle.configuration.imageModel == "custom_image.mlpackage")
    #expect(fromBundle.configuration.audioModel == "custom_audio.mlpackage")
    #expect(fromBundle.configuration.videoModel == "custom_video.mlpackage")
    #expect(fromBundle.configuration.embedModel == "custom_embed.mlpackage")
    #expect(fromBundle.configuration.decoderModel == "custom_decoder.mlpackage")
    #expect(fromBundle.configuration.textBuckets == [16, 32])

    do {
        _ = try fromBundle.audioEmbedder()
        #expect(Bool(false), "bundle-backed embedder should report the custom missing artifact path")
    } catch let error as JinaOmniError {
        #expect("\(error)".contains("custom_audio.mlpackage"), "error should use manifest paths: \(error)")
    }

    // Each modality reports its missing artifact (resolved under the models directory) and throws.
    #expect(throws: JinaOmniError.self) { _ = try jina.imageEmbedder() }
    #expect(throws: JinaOmniError.self) { _ = try jina.audioEmbedder() }
    #expect(throws: JinaOmniError.self) { _ = try jina.videoEmbedder() }
    #expect(throws: JinaOmniError.self) { _ = try jina.embed(imageURL: empty.appendingPathComponent("x.png")) }
    await #expect(throws: JinaOmniError.self) { _ = try await jina.embed(videoURL: empty.appendingPathComponent("x.mp4")) }
    await #expect(throws: JinaOmniError.self) { _ = try await jina.embed(text: "hello", prompt: .query) }

    do {
        _ = try await jina.embed(text: "hello", prompt: .query, dim: 2048)
        #expect(Bool(false), "invalid dimensions should be rejected before loading artifacts")
    } catch let error as JinaOmniError {
        #expect("\(error)".contains("invalid embedding dimension"), "unexpected dimension error: \(error)")
    }

    // The error message names the resolved path so callers can see exactly what to bundle.
    do {
        _ = try jina.audioEmbedder()
    } catch let error as JinaOmniError {
        #expect("\(error)".contains(empty.path), "error should name the resolved path: \(error)")
    }

    // .fullANE pins every tower to the Neural Engine.
    let ane = JinaOmniEmbedder.Configuration.fullANE()
    #expect(ane.encoderUnits == .cpuAndNeuralEngine)
    #expect(ane.decoderUnits == .cpuAndNeuralEngine)
    #expect(ane.textComputeUnits == .cpuAndNeuralEngine)

    let bundleANE = JinaOmniEmbedder.Configuration.fullANE(manifest: bundle.manifest)
    #expect(bundleANE.audioModel == "custom_audio.mlpackage")
    #expect(bundleANE.encoderUnits == .cpuAndNeuralEngine)
    #expect(bundleANE.decoderUnits == .cpuAndNeuralEngine)
    #expect(bundleANE.textComputeUnits == .cpuAndNeuralEngine)
}

@Test func externalModelBundleManifestAndArtifactsLoad() throws {
    guard let bundlePath = ProcessInfo.processInfo.environment["JINA_MODEL_BUNDLE"], !bundlePath.isEmpty else {
        return
    }

    let root = URL(fileURLWithPath: bundlePath)
    let bundle = try JinaModelBundle(url: root)
    let jina = JinaOmniEmbedder(bundle: bundle)

    #expect(jina.modelsDirectory == root)
    #expect(jina.modelID == "jinaai/jina-embeddings-v5-omni-small")
    #expect(jina.embeddingDimension == 1024)

    let manifest = bundle.manifest
    let requiredPaths = [
        manifest.text.model,
        manifest.text.tokenizer,
        manifest.image.encoder,
        manifest.image.resources,
        manifest.audio.encoder,
        manifest.video.encoder,
        manifest.decoder.embed,
        manifest.decoder.model,
    ]

    for path in requiredPaths {
        let url = bundle.resolve(path)
        #expect(FileManager.default.fileExists(atPath: url.path), "missing bundle artifact: \(url.path)")
    }
}

/// Make a tiny synthetic H.264 .mp4 (distinct gray per frame). Returns nil if encoding is
/// unavailable (headless/CI) so the round-trip test skips rather than flaking.
private func makeSyntheticMP4(frames: Int, size: Int) -> URL? {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("jina_\(UUID().uuidString).mp4")
    guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return nil }
    let settings: [String: Any] = [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: size, AVVideoHeightKey: size]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
    guard writer.canAdd(input) else { return nil }
    writer.add(input)
    guard writer.startWriting() else { return nil }
    writer.startSession(atSourceTime: .zero)
    for i in 0..<frames {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32ARGB, nil, &pb)
        guard let buf = pb else { return nil }
        CVPixelBufferLockBaseAddress(buf, [])
        guard let baseAddress = CVPixelBufferGetBaseAddress(buf) else {
            CVPixelBufferUnlockBaseAddress(buf, [])
            return nil
        }
        let g = Int32(20 + i * (200 / max(frames, 1)))
        for y in 0..<size { memset(baseAddress.advanced(by: y * CVPixelBufferGetBytesPerRow(buf)), g, size * 4) }
        CVPixelBufferUnlockBaseAddress(buf, [])
        while !input.isReadyForMoreMediaData { usleep(2000) }
        _ = adaptor.append(buf, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: 10))
    }
    input.markAsFinished()
    let sem = DispatchSemaphore(value: 0)
    writer.finishWriting { sem.signal() }
    sem.wait()
    return writer.status == .completed ? url : nil
}

/// Video file decode: error cases (always) + a synthetic .mp4 frame-extraction round-trip (skips if
/// the platform can't encode). Validates the embed(videoURL:) extractor mechanics.
@Test func videoFileExtraction() async throws {
    let prep = JinaImagePreprocessor()
    // odd / empty frame counts and a bad URL must throw, not crash
    await #expect(throws: JinaVideoFile.DecodeError.self) {
        _ = try await JinaVideoFile.extractSquareFrames(URL(fileURLWithPath: "/nonexistent.mp4"), count: 4, size: 64, preprocessor: prep)
    }
    await #expect(throws: JinaVideoFile.DecodeError.self) {
        _ = try await JinaVideoFile.extractSquareFrames(FileManager.default.temporaryDirectory, count: 3, size: 64, preprocessor: prep)
    }
    await #expect(throws: JinaVideoFile.DecodeError.self) {
        _ = try await JinaVideoFile.extractSquareFrames(FileManager.default.temporaryDirectory, count: 4, size: 63, preprocessor: prep)
    }
    guard let url = makeSyntheticMP4(frames: 6, size: 64) else { return }   // skip if no encoder
    defer { try? FileManager.default.removeItem(at: url) }
    let frames = try await JinaVideoFile.extractSquareFrames(url, count: 4, size: 64, preprocessor: prep)
    #expect(frames.count == 4)
    #expect(frames.allSatisfy { $0.count == 64 * 64 * 3 })
    // distinct gray per source frame -> extracted frames should not all be identical
    #expect(Set(frames.map { $0.first ?? 0 }).count > 1, "extracted frames are all identical")
}

/// Bad input must THROW (recoverable), not crash the host app — no model artifacts needed.
@Test func badInputThrowsNotCrashes() throws {
    let mel = try JinaMelFrontend()   // bundled constants
    #expect(throws: JinaMelFrontend.MelError.self) {
        _ = try mel.packedMel([Float](repeating: 0, count: 50))   // < FFT half-window
    }
    _ = try mel.packedMel([Float](repeating: 0, count: 8000))     // valid clip: does not throw
    let prep = JinaImagePreprocessor()
    #expect(throws: JinaImagePreprocessor.ImageError.self) {
        _ = try prep.videoPixelValues(frames: [[UInt8](repeating: 0, count: 128 * 128 * 3)], h: 128, w: 128)  // odd frame count
    }
}

@Test(.enabled(if: artifacts("artifacts/coreml/text_multifunc.mlpackage"), artifactsHint))
func matryoshkaDimsAreUnitNorm() async throws {
    let embedder = try await JinaTextEmbedder(
        multiFunctionModelURL: path("artifacts/coreml/text_multifunc.mlpackage"),
        tokenizerFolder: path("artifacts/hf/jina-v5-omni-small"))
    for invalidDimension in [-1, 0, 1025] {
        #expect(throws: MatryoshkaError.self) {
            _ = try embedder.embed(batch: [], dim: invalidDimension)
        }
    }
    #expect(try embedder.embed(batch: [], dim: 1024).isEmpty)
    for d in [64, 256, 512] {
        let emb = try embedder.embed("semantic search", prompt: .query, dim: d)
        #expect(emb.count == d)
        var n: Float = 0; for x in emb { n += x * x }
        #expect(abs(n.squareRoot() - 1.0) < 1e-4)
    }
}
