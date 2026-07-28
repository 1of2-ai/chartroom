import AVFoundation
import Foundation
import Testing
@testable import JinaEmbeddings

@Test func unboundedAudioDecoderRetainsItsOneArgumentFunctionType() {
    let decoder: (URL) throws -> [Float] = JinaAudioFile.decode16kMono
    _ = decoder
}

@Test func unboundedAudioDecoderRejectsFrameCapacityOverflow() throws {
    #expect(
        try JinaAudioFile.checkedFrameCapacity(44_100) == 44_100
    )
    #expect(
        try JinaAudioFile.resampledFrameCapacity(
            sourceFrameCount: 44_100,
            inputSampleRate: 44_100
        ) == 20_096
    )

    let oversizedSource = AVAudioFramePosition(AVAudioFrameCount.max) + 1
    #expect(throws: JinaAudioFile.DecodeError.self) {
        _ = try JinaAudioFile.checkedFrameCapacity(oversizedSource)
    }
    #expect(throws: JinaAudioFile.DecodeError.self) {
        _ = try JinaAudioFile.resampledFrameCapacity(
            sourceFrameCount: AVAudioFramePosition(AVAudioFrameCount.max),
            inputSampleRate: 8_000
        )
    }
}

@Test func cappedAudioDecodePreservesUnboundedResampledPrefix() throws {
    let maximumSampleCount = 3_000 * 160 + 400
    let url = try makeChunkedAudioFixture(
        sampleRate: 44_100,
        channelCount: 2,
        duration: 31.5
    )
    defer { try? FileManager.default.removeItem(at: url) }

    let sourceFile = try AVAudioFile(forReading: url)
    let outputFormat = try #require(
        AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        )
    )
    let converter = try #require(
        AVAudioConverter(
            from: sourceFile.processingFormat,
            to: outputFormat
        )
    )
    converter.primeMethod = .normal
    let leadingFrames = converter.primeInfo.leadingFrames
    let trailingFrames = converter.primeInfo.trailingFrames
    let conversionSampleCount =
        maximumSampleCount + JinaAudioFile.resamplingLookaheadSampleCount
    let sourceFrameLimit = try JinaAudioFile.sourceFrameLimit(
        conversionOutputSampleCount: conversionSampleCount,
        inputSampleRate: sourceFile.processingFormat.sampleRate,
        trailingFrames: trailingFrames
    )
    #expect(sourceFrameLimit == 1_335_393 + AVAudioFramePosition(trailingFrames))

    let unbounded = try JinaAudioFile.decode16kMono(url)
    #expect(unbounded.count > maximumSampleCount)

    let bounded = try JinaAudioFile.decodeBounded(
        url,
        maximumSampleCount: maximumSampleCount
    )
    let capped = bounded.samples
    #expect(capped.count == maximumSampleCount)
    #expect(
        bounded.sourceFramesRead < sourceFile.length,
        "the bounded decoder must not consume the complete long source"
    )
    #expect(
        bounded.sourceFramesRead <= sourceFrameLimit,
        "the bounded decoder exceeded its calculated source-frame budget"
    )

    var maximumDifference: Float = 0
    var maximumDifferenceIndex = 0
    var firstSignificantDifferenceIndex: Int?
    for index in capped.indices {
        let difference = abs(capped[index] - unbounded[index])
        if difference >= 1e-5, firstSignificantDifferenceIndex == nil {
            firstSignificantDifferenceIndex = index
        }
        if difference > maximumDifference {
            maximumDifference = difference
            maximumDifferenceIndex = index
        }
    }
    #expect(
        maximumDifference < 1e-5,
        "maximum prefix difference \(maximumDifference) at \(maximumDifferenceIndex); first significant difference \(String(describing: firstSignificantDifferenceIndex)); prime frames \(leadingFrames)/\(trailingFrames)"
    )
}

@Test func audioEmbedderRequestsExactlyItsModelVisibleDecodeBudget() throws {
    let url = URL(fileURLWithPath: "/unused-audio-fixture")
    let expected = JinaAudioFile.BoundedDecode(
        samples: [0.25, -0.25],
        sourceFramesRead: 17
    )
    var requestedURL: URL?
    var requestedMaximumSampleCount: Int?

    let actual = try JinaAudioEmbedderMasked.decodeForEmbedding(url) { url, maximumSampleCount in
        requestedURL = url
        requestedMaximumSampleCount = maximumSampleCount
        return expected
    }

    #expect(requestedURL == url)
    #expect(requestedMaximumSampleCount == JinaAudioEmbedderMasked.maximumDecodedSampleCount)
    #expect(actual.samples == expected.samples)
    #expect(actual.sourceFramesRead == expected.sourceFramesRead)
}

@Test func cappedAudioDecodeBoundsExactPCMWithoutChangingSamples() throws {
    let maximumSampleCount = 8_000
    let url = try makeChunkedAudioFixture(
        sampleRate: 16_000,
        channelCount: 1,
        duration: 1.5
    )
    defer { try? FileManager.default.removeItem(at: url) }

    let unbounded = try JinaAudioFile.decode16kMono(url)
    let capped = try JinaAudioFile.decode16kMono(
        url,
        maximumSampleCount: maximumSampleCount
    )

    #expect(unbounded.count > maximumSampleCount)
    #expect(capped.count == maximumSampleCount)
    #expect(capped == Array(unbounded.prefix(maximumSampleCount)))
}

private func makeChunkedAudioFixture(
    sampleRate: Double,
    channelCount: AVAudioChannelCount,
    duration: Double
) throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("jina_long_\(UUID().uuidString).wav")
    let format = try #require(
        AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: channelCount,
            interleaved: false
        )
    )
    let totalFrameCount = Int(sampleRate * duration)
    let chunkCapacity: AVAudioFrameCount = 4_096

    do {
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        var frameOffset = 0
        while frameOffset < totalFrameCount {
            let frameCount = min(Int(chunkCapacity), totalFrameCount - frameOffset)
            let buffer = try #require(
                AVAudioPCMBuffer(pcmFormat: format, frameCapacity: chunkCapacity)
            )
            buffer.frameLength = AVAudioFrameCount(frameCount)
            let channels = try #require(buffer.floatChannelData)
            for channel in 0..<Int(channelCount) {
                for frame in 0..<frameCount {
                    let sourceFrame = frameOffset + frame
                    let saw = Float(sourceFrame % 997) / 997 - 0.5
                    channels[channel][frame] = saw * (channel == 0 ? 0.5 : -0.25)
                }
            }
            try file.write(from: buffer)
            frameOffset += frameCount
        }
    }

    return url
}
