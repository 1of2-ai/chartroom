import AVFoundation
import CoreML
import Foundation

/// Decode an audio file to a 16 kHz mono `[Float]` (the model's input rate) via AVFoundation.
/// EXACT for files already 16 kHz mono (passthrough); other sample rates are resampled by
/// AVAudioConverter, which is not bit-identical to the reference's librosa resampler — a small
/// resample-only caveat (analogous to the image CoreGraphics-vs-PIL note).
public enum JinaAudioFile {
    public enum DecodeError: Error {
        case converter
        case noData
        case invalidMaximumSampleCount(Int)
        case frameCapacityOverflow(
            sourceFrameCount: AVAudioFramePosition,
            inputSampleRate: Double?
        )
    }

    struct BoundedDecode: Sendable {
        var samples: [Float]
        var sourceFramesRead: AVAudioFramePosition
    }

    private static let outputSampleRate = 16_000.0
    private static let boundedInputChunkFrames: AVAudioFrameCount = 8_192
    /// Keep a small bounded conversion tail so the returned prefix has the same resampler context
    /// as an unbounded conversion. This preserves the existing decoder's `+ 4096` output slack.
    static let resamplingLookaheadSampleCount = 4_096

    /// Decode a complete file to 16 kHz mono.
    public static func decode16kMono(_ url: URL) throws -> [Float] {
        try decodeUnbounded(AVAudioFile(forReading: url))
    }

    /// Decode a bounded prefix. Resampling reads a separately bounded lookahead so the returned
    /// samples match the same prefix of a full conversion.
    public static func decode16kMono(
        _ url: URL,
        maximumSampleCount: Int
    ) throws -> [Float] {
        try decodeBounded(
            url,
            maximumSampleCount: maximumSampleCount
        ).samples
    }

    static func decodeBounded(
        _ url: URL,
        maximumSampleCount: Int
    ) throws -> BoundedDecode {
        try decodeBounded(
            AVAudioFile(forReading: url),
            maximumSampleCount: maximumSampleCount
        )
    }

    private static func decodeUnbounded(_ file: AVAudioFile) throws -> [Float] {
        let inFormat = file.processingFormat   // always float32, file's rate + channels
        let inputFrameCapacity = try checkedFrameCapacity(file.length)
        guard let inBuf = AVAudioPCMBuffer(
            pcmFormat: inFormat,
            frameCapacity: inputFrameCapacity
        ) else {
            throw DecodeError.converter
        }
        try file.read(into: inBuf)
        // Already 16 kHz mono -> read the PCM directly (no converter, hence no priming latency): exact.
        if inFormat.sampleRate == 16000, inFormat.channelCount == 1, let ch = inBuf.floatChannelData {
            return Array(UnsafeBufferPointer(start: ch[0], count: Int(inBuf.frameLength)))
        }
        // Otherwise resample/downmix via AVAudioConverter (small resample-only caveat vs librosa).
        guard let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                            channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
            throw DecodeError.converter
        }
        let outCap = try resampledFrameCapacity(
            sourceFrameCount: file.length,
            inputSampleRate: inFormat.sampleRate
        )
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCap) else { throw DecodeError.converter }
        let input = AudioConverterInput(buffer: inBuf)
        var convErr: NSError?
        let status = converter.convert(to: outBuf, error: &convErr) { _, inStatus in
            input.next(status: inStatus)
        }
        if let convErr { throw convErr }
        guard status != .error else { throw DecodeError.noData }
        let n = Int(outBuf.frameLength)
        guard n > 0, let ch = outBuf.floatChannelData else { throw DecodeError.noData }
        return Array(UnsafeBufferPointer(start: ch[0], count: n))
    }

    static func checkedFrameCapacity(
        _ sourceFrameCount: AVAudioFramePosition
    ) throws -> AVAudioFrameCount {
        guard let capacity = AVAudioFrameCount(exactly: sourceFrameCount) else {
            throw DecodeError.frameCapacityOverflow(
                sourceFrameCount: sourceFrameCount,
                inputSampleRate: nil
            )
        }
        return capacity
    }

    static func resampledFrameCapacity(
        sourceFrameCount: AVAudioFramePosition,
        inputSampleRate: Double
    ) throws -> AVAudioFrameCount {
        let lookahead = AVAudioFrameCount(resamplingLookaheadSampleCount)
        let maximumConvertedFrameCount = AVAudioFrameCount.max - lookahead
        let convertedFrameCount =
            Double(sourceFrameCount) * outputSampleRate / inputSampleRate
        guard sourceFrameCount >= 0,
              inputSampleRate.isFinite,
              inputSampleRate > 0,
              convertedFrameCount.isFinite,
              convertedFrameCount >= 0,
              convertedFrameCount <= Double(maximumConvertedFrameCount) else {
            throw DecodeError.frameCapacityOverflow(
                sourceFrameCount: sourceFrameCount,
                inputSampleRate: inputSampleRate
            )
        }
        return AVAudioFrameCount(convertedFrameCount) + lookahead
    }

    private static func decodeBounded(
        _ file: AVAudioFile,
        maximumSampleCount: Int
    ) throws -> BoundedDecode {
        let (conversionSampleCount, conversionSampleCountOverflow) =
            maximumSampleCount.addingReportingOverflow(
                resamplingLookaheadSampleCount
            )
        guard maximumSampleCount > 0,
              !conversionSampleCountOverflow,
              conversionSampleCount <= Int(AVAudioFrameCount.max) else {
            throw DecodeError.invalidMaximumSampleCount(maximumSampleCount)
        }
        let inFormat = file.processingFormat

        // Already 16 kHz mono -> bounded direct PCM read, still bit-exact.
        if inFormat.sampleRate == outputSampleRate, inFormat.channelCount == 1 {
            let frameCount = min(
                AVAudioFramePosition(maximumSampleCount),
                file.length
            )
            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: inFormat,
                frameCapacity: AVAudioFrameCount(frameCount)
            ) else {
                throw DecodeError.converter
            }
            try file.read(into: buffer, frameCount: AVAudioFrameCount(frameCount))
            let count = Int(buffer.frameLength)
            guard count > 0, let channels = buffer.floatChannelData else {
                throw DecodeError.noData
            }
            return BoundedDecode(
                samples: Array(UnsafeBufferPointer(start: channels[0], count: count)),
                sourceFramesRead: AVAudioFramePosition(count)
            )
        }

        guard let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: outputSampleRate,
            channels: 1,
            interleaved: false
        ),
        let converter = AVAudioConverter(from: inFormat, to: outFormat),
        let outBuffer = AVAudioPCMBuffer(
            pcmFormat: outFormat,
            frameCapacity: AVAudioFrameCount(conversionSampleCount)
        ) else {
            throw DecodeError.converter
        }
        converter.primeMethod = .normal
        let frameLimit = try sourceFrameLimit(
            conversionOutputSampleCount: conversionSampleCount,
            inputSampleRate: inFormat.sampleRate,
            trailingFrames: converter.primeInfo.trailingFrames
        )
        let input = BoundedAudioConverterInput(
            file: file,
            frameLimit: min(file.length, frameLimit),
            maximumChunkFrames: boundedInputChunkFrames
        )
        var conversionError: NSError?
        let status = converter.convert(
            to: outBuffer,
            error: &conversionError
        ) { packetCount, inputStatus in
            input.next(
                requestedPacketCount: packetCount,
                status: inputStatus
            )
        }
        if let readError = input.capturedError() { throw readError }
        if let conversionError { throw conversionError }
        guard status != .error else { throw DecodeError.noData }
        let count = Int(outBuffer.frameLength)
        guard count > 0, let channels = outBuffer.floatChannelData else {
            throw DecodeError.noData
        }
        let returnedSampleCount = min(count, maximumSampleCount)
        return BoundedDecode(
            samples: Array(
                UnsafeBufferPointer(
                    start: channels[0],
                    count: returnedSampleCount
                )
            ),
            sourceFramesRead: input.suppliedFrames()
        )
    }

    static func sourceFrameLimit(
        conversionOutputSampleCount: Int,
        inputSampleRate: Double,
        trailingFrames: AVAudioFrameCount
    ) throws -> AVAudioFramePosition {
        guard conversionOutputSampleCount > 0,
              inputSampleRate.isFinite,
              inputSampleRate > 0 else {
            throw DecodeError.invalidMaximumSampleCount(conversionOutputSampleCount)
        }
        let convertedFrameCount = (
            Double(conversionOutputSampleCount) * inputSampleRate / outputSampleRate
        ).rounded(.up)
        let totalFrameCount = convertedFrameCount + Double(trailingFrames)
        guard totalFrameCount.isFinite,
              totalFrameCount <= Double(AVAudioFramePosition.max) else {
            throw DecodeError.converter
        }
        return AVAudioFramePosition(totalFrameCount)
    }
}

private final class AudioConverterInput: @unchecked Sendable {
    private let buffer: AVAudioPCMBuffer
    private let lock = NSLock()
    private var didFeed = false

    init(buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }

    func next(status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioPCMBuffer? {
        lock.lock()
        defer { lock.unlock() }
        guard !didFeed else {
            status.pointee = .endOfStream
            return nil
        }
        didFeed = true
        status.pointee = .haveData
        return buffer
    }
}

private final class BoundedAudioConverterInput: @unchecked Sendable {
    private let file: AVAudioFile
    private let frameLimit: AVAudioFramePosition
    private let maximumChunkFrames: AVAudioFrameCount
    private let lock = NSLock()
    private var suppliedFrameCount: AVAudioFramePosition = 0
    private var readError: Error?

    init(
        file: AVAudioFile,
        frameLimit: AVAudioFramePosition,
        maximumChunkFrames: AVAudioFrameCount
    ) {
        self.file = file
        self.frameLimit = frameLimit
        self.maximumChunkFrames = maximumChunkFrames
    }

    func next(
        requestedPacketCount: AVAudioPacketCount,
        status: UnsafeMutablePointer<AVAudioConverterInputStatus>
    ) -> AVAudioPCMBuffer? {
        lock.lock()
        defer { lock.unlock() }

        guard readError == nil else {
            status.pointee = .endOfStream
            return nil
        }
        let remainingFrameCount = frameLimit - suppliedFrameCount
        guard remainingFrameCount > 0 else {
            status.pointee = .endOfStream
            return nil
        }
        let frameCount = min(
            AVAudioFramePosition(requestedPacketCount),
            remainingFrameCount,
            AVAudioFramePosition(maximumChunkFrames)
        )
        guard frameCount > 0 else {
            status.pointee = .noDataNow
            return nil
        }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat,
            frameCapacity: AVAudioFrameCount(frameCount)
        ) else {
            readError = JinaAudioFile.DecodeError.converter
            status.pointee = .endOfStream
            return nil
        }
        do {
            try file.read(
                into: buffer,
                frameCount: AVAudioFrameCount(frameCount)
            )
        } catch {
            readError = error
            status.pointee = .endOfStream
            return nil
        }
        guard buffer.frameLength > 0 else {
            status.pointee = .endOfStream
            return nil
        }
        suppliedFrameCount += AVAudioFramePosition(buffer.frameLength)
        status.pointee = .haveData
        return buffer
    }

    func capturedError() -> Error? {
        lock.lock()
        defer { lock.unlock() }
        return readError
    }

    func suppliedFrames() -> AVAudioFramePosition {
        lock.lock()
        defer { lock.unlock() }
        return suppliedFrameCount
    }
}

/// True arbitrary-length audio embedder: the runtime-MASKED encoder (host-built conv_mask +
/// attn_bias mask the partial boundary chunk) + the unified general decoder. Unlike the
/// truncate-real path, this matches the reference at ANY clip length up to ~30 s (the model's
/// WhisperFeatureExtractor limit; buckets 2/4/8/16/32 s). Encoder on GPU (fp32 matmul accumulation),
/// general decoder on the ANE.
public final class JinaAudioEmbedderMasked: @unchecked Sendable {
    static let prefix = JinaPromptTokens.audioPrefix
    static let suffix = JinaPromptTokens.audioSuffix
    static let audioPad = JinaPromptTokens.audioPad

    public let mel: JinaMelFrontend
    public let encoder: AudioCoreMLEncoderMasked
    public let decoder: GeneralMediaDecoder

    /// `encoderUnits` selects the encoder's compute placement. `.cpuAndGPU` (default) is the
    /// recommended choice — it is BOTH more accurate (fp32 matmul accumulation: ~0.99999 vs the
    /// ANE's fp16 ~0.998) AND faster (measured 83ms vs 129ms end-to-end for an 8s clip; the large
    /// attn_bias + masked-SDPA don't map well to the ANE). `.cpuAndNeuralEngine` runs the encoder on
    /// the ANE too (full-ANE) but is slower and less accurate — provided only for GPU-contended cases.
    /// `decoderUnits`: `nil` (default) = adaptive (ANE for S≤256, GPU for S≥512). Pass
    /// `.cpuAndNeuralEngine` with `encoderUnits: .cpuAndNeuralEngine` for a true full-ANE deployment
    /// (measured end-to-end cos 0.996365, above the model's bf16 audio floor 0.994951; slower, GPU-free).
    public init(audioModelURL: URL, embedModelURL: URL, decoderModelURL: URL,
                encoderUnits: MLComputeUnits = .cpuAndGPU, decoderUnits: MLComputeUnits? = nil) throws {
        mel = try JinaMelFrontend()
        encoder = try AudioCoreMLEncoderMasked(modelURL: audioModelURL, computeUnits: encoderUnits)
        decoder = try GeneralMediaDecoder(embedModelURL: embedModelURL, decoderModelURL: decoderModelURL, computeUnits: decoderUnits)
    }

    /// The model's audio limit: WhisperFeatureExtractor caps at 30 s = 3000 mel frames. Longer clips
    /// are truncated here to match (the reference can't see past 30 s either).
    public static let maxFrames = 3000

    /// Audio file -> embedding (AVFoundation decode to 16 kHz mono, then `embed(_:)`). Exact for
    /// already-16 kHz-mono files; other rates carry the documented resample caveat. Clips >30 s truncate.
    public func embed(audioURL: URL, dim: Int? = nil) throws -> [Float] {
        try embed(Self.decodeForEmbedding(audioURL).samples, dim: dim)
    }

    public static let maximumDecodedSampleCount = maxFrames * 160 + 400

    typealias BoundedDecoder = (URL, Int) throws -> JinaAudioFile.BoundedDecode

    static func decodeForEmbedding(
        _ url: URL,
        using decoder: BoundedDecoder = JinaAudioFile.decodeBounded
    ) throws -> JinaAudioFile.BoundedDecode {
        try decoder(url, maximumDecodedSampleCount)
    }

    /// 16 kHz mono waveform -> embedding. Exact reference parity for any length up to ~30 s.
    public func embed(_ audio: [Float], dim: Int? = nil) throws -> [Float] {
        let exactFrames = min(audio.count / mel.hop, Self.maxFrames)
        let F = AudioMasks.bucket(forFrames: exactFrames)
        let masks = AudioMasks(exactFrames: exactFrames, bucketFrames: F)
        // Zero the packed mel beyond the real frames (mel-space zeros, NOT the log-mel floor): conv1
        // runs per-chunk before the mask, so its kernel reaches from the last real frame into the
        // partial chunk's padding — floor values there contaminate it. Reference pads with zeros.
        var packed = try mel.packedMel(audio, frames: F)
        if exactFrames < F {
            for m in 0..<mel.nMels { for t in exactFrames..<F { packed[m * F + t] = 0.0 } }
        }
        let full = try encoder.encode(packedMel: packed, nMels: mel.nMels, masks: masks)
        let L = masks.realTokens
        let used = Array(full[0 ..< (L * 1024)])
        let ids = Self.prefix + Array(repeating: Self.audioPad, count: L) + Self.suffix
        let emb = try decoder.decode(tokenIds: ids, features: used, scatterOffset: Self.prefix.count)
        if let d = dim { return try matryoshka(emb, dim: d) }
        return emb
    }
}
