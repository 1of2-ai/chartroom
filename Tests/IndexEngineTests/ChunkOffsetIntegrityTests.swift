import Foundation
import Testing
@testable import IndexEngine

/// Offsets are the contract downstream consumers rely on: `characterStart..<characterEnd`
/// sliced from the document must be exactly the chunk's text. Whitespace runs near chunk
/// boundaries are the adversarial case — trimming moves chunk starts while the overlap
/// walks backwards, and any bookkeeping slip between the two corrupts offsets silently.
@Suite("Chunk offset integrity")
struct ChunkOffsetIntegrityTests {
    private static let layouts: [(name: String, body: String)] = {
        let text = { (n: Int, seed: String) in
            (0..<n).map { "\(seed)\($0) " }.joined()
        }
        return [
            ("leading whitespace exceeds the overlap",
             text(200, "alpha") + String(repeating: " ", count: 1500) + text(200, "beta")),
            ("an entire chunk of whitespace",
             text(200, "gamma") + String(repeating: " ", count: 1700) + text(200, "delta")),
            ("newline runs instead of spaces",
             text(200, "epsilon") + String(repeating: "\n", count: 1500) + text(200, "zeta")),
            ("alternating text islands in an ocean of whitespace",
             (0..<6).map { text(30, "island\($0)x") }.joined(separator: String(repeating: " ", count: 1450))),
            ("whitespace run exactly at the overlap width",
             text(200, "eta") + String(repeating: " ", count: 160) + text(200, "theta")),
            ("trailing whitespace pushing the final chunk",
             text(400, "iota") + String(repeating: " ", count: 1600)),
        ]
    }()

    @Test("character and byte offsets always slice the chunk's exact text", arguments: layouts.map(\.name))
    func offsetsSliceExactText(layoutName: String) async throws {
        let body = try #require(Self.layouts.first { $0.name == layoutName }).body
        let store = try IndexStore(path: ":memory:", embedder: HashingEmbedder(dimension: 64))
        try await store.upsert(.init(id: "doc", type: "note", title: "Offsets", body: body))

        let chunks = try await store.chunkSummaries(documentID: "doc")
        #expect(!chunks.isEmpty)

        let utf8 = Array(body.utf8)
        for chunk in chunks {
            let charStart = body.index(body.startIndex, offsetBy: chunk.characterStart)
            let charEnd = body.index(charStart, offsetBy: chunk.characterEnd - chunk.characterStart)
            #expect(
                String(body[charStart..<charEnd]) == chunk.text,
                "ordinal \(chunk.ordinal): characterStart \(chunk.characterStart) does not slice the chunk's text"
            )
            let byteSlice = String(decoding: utf8[chunk.byteStart..<chunk.byteEnd], as: UTF8.self)
            #expect(
                byteSlice == chunk.text,
                "ordinal \(chunk.ordinal): byteStart \(chunk.byteStart) does not slice the chunk's text"
            )
        }
    }
}
