import Foundation
import Testing
@testable import JinaEmbeddings

/// Model resources are raw float32 files; a corrupt or truncated one must fail at load
/// with a cause-naming error, not floor-divide into fewer floats and surface later as an
/// unrelated shape mismatch (or as silently wrong rotations).
@Suite("Model resource loading")
struct ResourceLoadingTests {
    private func write(_ bytes: [UInt8], name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("resource-\(UUID().uuidString)-\(name)")
        try Data(bytes).write(to: url)
        return url
    }

    private func writeFloats(_ floats: [Float], name: String) throws -> URL {
        var data = Data()
        for f in floats {
            withUnsafeBytes(of: f.bitPattern.littleEndian) { data.append(contentsOf: $0) }
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("resource-\(UUID().uuidString)-\(name)")
        try data.write(to: url)
        return url
    }

    @Test("a trailing partial word is rejected, not truncated")
    func partialWordRejected() throws {
        let url = try write([0, 0, 128, 63, 0, 0, 128], name: "truncated.f32") // 7 bytes
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: (any Error).self) {
            _ = try loadF32(url)
        }
    }

    @Test("well-formed float data round-trips")
    func wellFormedLoads() throws {
        let url = try writeFloats([1.5, -2.25], name: "ok.f32")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(try loadF32(url) == [1.5, -2.25])
    }

    @Test("an invFreq table shorter than meta's declared length is rejected at init")
    func truncatedInvFreqRejected() throws {
        let meta = """
        {"num_grid_per_side": 2, "hidden": 4, "spatial_merge_size": 2,
         "patch_size": 14, "rope_theta": 10000, "rope_inv_freq_len": 4}
        """
        let metaURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("meta-\(UUID().uuidString).json")
        try Data(meta.utf8).write(to: metaURL)
        let posTableURL = try writeFloats(Array(repeating: 0, count: 2 * 2 * 4), name: "pos.f32")
        let goodInvFreqURL = try writeFloats([1, 0.5, 0.25, 0.125], name: "freq.f32")
        let shortInvFreqURL = try writeFloats([1, 0.5, 0.25], name: "short-freq.f32")
        defer {
            for url in [metaURL, posTableURL, goodInvFreqURL, shortInvFreqURL] {
                try? FileManager.default.removeItem(at: url)
            }
        }

        let valid = try VisionPositions(metaURL: metaURL, posTableURL: posTableURL, invFreqURL: goodInvFreqURL)
        #expect(valid.ropeDim == 16)

        #expect(throws: (any Error).self) {
            _ = try VisionPositions(metaURL: metaURL, posTableURL: posTableURL, invFreqURL: shortInvFreqURL)
        }
    }
}
