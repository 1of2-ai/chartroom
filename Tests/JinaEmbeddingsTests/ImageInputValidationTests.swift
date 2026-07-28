import Testing
@testable import JinaEmbeddings

@Test func rawImageRejectsInvalidDimensionsAndByteCounts() {
    let preprocessor = JinaImagePreprocessor()
    let cases: [(rgb: [UInt8], h: Int, w: Int)] = [
        ([], 0, 32),
        ([], -1, 32),
        ([UInt8](repeating: 0, count: 16 * 32 * 3), 16, 32),
        ([UInt8](repeating: 0, count: 48 * 32 * 3), 48, 32),
        ([UInt8](repeating: 0, count: 32 * 32 * 3 - 1), 32, 32),
        ([UInt8](repeating: 0, count: 32 * 32 * 3 + 1), 32, 32),
    ]

    for value in cases {
        #expect(throws: JinaImagePreprocessor.ImageError.self) {
            _ = try preprocessor.pixelValues(rgb: value.rgb, h: value.h, w: value.w)
        }
    }
}

@Test func rawVideoRejectsInvalidDimensionsAndPerFrameByteCounts() {
    let preprocessor = JinaImagePreprocessor()
    let alignedFrame = [UInt8](repeating: 0, count: 32 * 32 * 3)

    #expect(throws: JinaImagePreprocessor.ImageError.self) {
        _ = try preprocessor.videoPixelValues(frames: [alignedFrame, alignedFrame], h: 16, w: 32)
    }
    #expect(throws: JinaImagePreprocessor.ImageError.self) {
        let shortFrame = [UInt8](repeating: 0, count: 32 * 32 * 3 - 1)
        _ = try preprocessor.videoPixelValues(frames: [alignedFrame, shortFrame], h: 32, w: 32)
    }
    #expect(throws: JinaImagePreprocessor.ImageError.self) {
        let oversizedFrame = [UInt8](repeating: 0, count: 32 * 32 * 3 + 1)
        _ = try preprocessor.videoPixelValues(frames: [alignedFrame, oversizedFrame], h: 32, w: 32)
    }
}

@Test func rawImageAndVideoRejectOutputCountOverflowBeforeAllocation() {
    let preprocessor = JinaImagePreprocessor()
    let h = 1 << 31
    let w = 1 << 30

    do {
        _ = try preprocessor.pixelValues(rgb: [], h: h, w: w)
        Issue.record("Expected image tensor-size overflow")
    } catch let error as JinaImagePreprocessor.ImageError {
        if case let .tensorSizeOverflow(t, gh, gw) = error {
            #expect(t == 1)
            #expect(gh == h / preprocessor.patch)
            #expect(gw == w / preprocessor.patch)
        } else {
            Issue.record("Expected tensorSizeOverflow, got \(error)")
        }
    } catch {
        Issue.record("Expected tensorSizeOverflow, got \(error)")
    }

    do {
        _ = try preprocessor.videoPixelValues(frames: [[], []], h: h, w: w)
        Issue.record("Expected video tensor-size overflow")
    } catch let error as JinaImagePreprocessor.ImageError {
        if case let .tensorSizeOverflow(t, gh, gw) = error {
            #expect(t == 1)
            #expect(gh == h / preprocessor.patch)
            #expect(gw == w / preprocessor.patch)
        } else {
            Issue.record("Expected tensorSizeOverflow, got \(error)")
        }
    } catch {
        Issue.record("Expected tensorSizeOverflow, got \(error)")
    }
}

@Test func validRawImageAndVideoBuffersStillPatchify() throws {
    let preprocessor = JinaImagePreprocessor()
    let frame = [UInt8](repeating: 127, count: 32 * 32 * 3)

    let image = try preprocessor.pixelValues(rgb: frame, h: 32, w: 32)
    #expect(image.gh == 2)
    #expect(image.gw == 2)
    #expect(image.pixels.count == 4 * 1_536)

    let video = try preprocessor.videoPixelValues(frames: [frame, frame], h: 32, w: 32)
    #expect(video.t == 1)
    #expect(video.gh == 2)
    #expect(video.gw == 2)
    #expect(video.pixels.count == 4 * 1_536)
}

@Test func lowLevelTensorShapesRejectZeroOverflowAndInconsistentCounts() {
    let preprocessor = JinaImagePreprocessor()
    let invalidShapes: [(count: Int, t: Int, gh: Int, gw: Int)] = [
        (0, 1, 0, 2),
        (0, 0, 2, 2),
        (4 * 1_536 - 1, 1, 2, 2),
        (0, Int.max, 32, 32),
    ]

    for shape in invalidShapes {
        #expect(throws: JinaImagePreprocessor.ImageError.self) {
            _ = try preprocessor.validatedTensorShape(
                pixelValuesCount: shape.count,
                t: shape.t,
                gh: shape.gh,
                gw: shape.gw
            )
        }
    }
}

@Test func lowLevelTensorShapesAcceptConsistentImageAndVideoGrids() throws {
    let preprocessor = JinaImagePreprocessor()

    let image = try preprocessor.validatedTensorShape(
        pixelValuesCount: 4 * 1_536,
        t: 1,
        gh: 2,
        gw: 2
    )
    #expect(image.patchCount == 4)
    #expect(image.pixelDimension == 1_536)

    let video = try preprocessor.validatedTensorShape(
        pixelValuesCount: 8 * 1_536,
        t: 2,
        gh: 2,
        gw: 2
    )
    #expect(video.patchCount == 8)
}

@Test func lowLevelTensorShapesRejectUnmergeableImageAndVideoGrids() {
    let preprocessor = JinaImagePreprocessor()
    let invalidGrids: [(t: Int, gh: Int, gw: Int)] = [
        (1, 3, 2),
        (1, 2, 3),
        (2, 3, 2),
        (2, 2, 3),
    ]

    for grid in invalidGrids {
        #expect(throws: JinaImagePreprocessor.ImageError.self) {
            _ = try preprocessor.validatedTensorShape(
                pixelValuesCount: grid.t * grid.gh * grid.gw * 1_536,
                t: grid.t,
                gh: grid.gh,
                gw: grid.gw
            )
        }
    }
}
