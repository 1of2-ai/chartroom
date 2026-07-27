import Foundation
import Testing
@testable import IndexEngine

/// A hit that carries a path and 240 characters of snippet forces the consumer to re-read the whole
/// file — which is what grep would have done for free. Line numbers are the unit an editor jump, a
/// file read, and an edit all actually take, so retrieval has to carry them.
@Suite("Chunk location")
struct ChunkLocationTests {
    private func temporaryStorePath() -> String {
        FileManager.default.temporaryDirectory
            .appending(path: "chartroom-lines-\(UUID().uuidString).sqlite")
            .path
    }

    /// Lines are computed at ingest from the representation text. Deriving them at query time would
    /// mean re-reading the source, which may have changed since indexing.
    @Test("chunk line ranges track the document's real lines")
    func lineRangesFollowDocumentLines() async throws {
        let store = try IndexStore(path: ":memory:", embedder: HashingEmbedder(dimension: 64))
        // Long enough to split into several chunks, with predictable line structure.
        let lines = (1...400).map { "line \($0) retrieval marker content for the chunker to split on" }
        try await store.upsert(.init(id: "doc-1", type: "note", title: "Lines", body: lines.joined(separator: "\n")))

        let chunks = try await store.chunkSummaries(documentID: "doc-1")

        #expect(chunks.count > 1, "the fixture must actually chunk for this to mean anything")
        for chunk in chunks {
            let start = try #require(chunk.lineStart, "every freshly ingested chunk carries a line range")
            let end = try #require(chunk.lineEnd)
            #expect(start >= 1, "line numbers are 1-based, matching every editor")
            #expect(end >= start)
            #expect(end <= lines.count)
        }

        let first = try #require(chunks.first)
        #expect(first.lineStart == 1)

        // Ordinal order is document order, so line ranges must advance monotonically.
        let starts = chunks.compactMap(\.lineStart)
        #expect(starts == starts.sorted())
    }

    @Test("a single-line document occupies exactly one line")
    func singleLineDocumentIsOneLine() async throws {
        let store = try IndexStore(path: ":memory:", embedder: HashingEmbedder(dimension: 64))
        try await store.upsert(.init(id: "doc-1", type: "note", title: "One", body: "a single line of text"))

        let chunk = try #require(try await store.chunkSummaries(documentID: "doc-1").first)

        #expect(chunk.lineStart == 1)
        #expect(chunk.lineEnd == 1)
    }

    /// The line range must survive the projection into search results, which is the surface both
    /// the app and the MCP server consume.
    @Test("search results carry the matching chunk's line range")
    func searchResultsCarryLineRanges() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 64))
        )
        let body = (1...300).map { "line \($0) padding" }.joined(separator: "\n")
            + "\nthe distinctive thermal throttling marker\n"
            + (301...600).map { "line \($0) padding" }.joined(separator: "\n")
        _ = try await engine.ingest(.init(payloads: [
            SourcePayload(
                documentID: "doc-1",
                sourceID: "local-files",
                sourceURI: URL(filePath: "/tmp/lines.md"),
                displayName: "Lines",
                contentType: "net.daringfireball.markdown",
                body: .text(body)
            )
        ]))

        let response = try await engine.search(.init(query: "thermal throttling marker", limit: 5))
        let best = try #require(response.results.first)

        let start = try #require(best.lineStart, "a result without a line range is barely better than a filename")
        let end = try #require(best.lineEnd)
        #expect(start >= 1)
        #expect(end >= start)
        // The marker sits around line 301, so the matching chunk must span it rather than
        // reporting the top of the file.
        #expect(start <= 301 && end >= 301)
    }

    @Test("chunks carry the surrounding text so a match can be placed in its document")
    func chunksCarryContext() async throws {
        let store = try IndexStore(path: ":memory:", embedder: HashingEmbedder(dimension: 64))
        let lines = (1...400).map { "line \($0) retrieval marker content for the chunker to split on" }
        try await store.upsert(.init(id: "doc-1", type: "note", title: "Context", body: lines.joined(separator: "\n")))

        let chunks = try await store.chunkSummaries(documentID: "doc-1")

        let first = try #require(chunks.first)
        #expect(first.contextPrefix.isEmpty, "the first chunk has nothing before it")
        #expect(!first.contextSuffix.isEmpty, "a mid-document chunk boundary has text after it")

        let last = try #require(chunks.last)
        #expect(!last.contextPrefix.isEmpty)
    }

    /// Existing stores carry chunks written before these columns existed. Those chunks have
    /// *unknown* lines, and reporting them as line 0 would be a false claim rather than a missing one.
    @Test("a chunk indexed before line tracking reports unknown, not zero")
    func preMigrationChunksReportUnknownLines() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))
        try await store.upsert(.init(id: "doc-1", type: "note", title: "Old", body: "some indexed text"))

        // Simulate a row written by the previous schema, where the columns did not exist.
        try await store.clearChunkLineRangesForTesting()

        let chunk = try #require(try await store.chunkSummaries(documentID: "doc-1").first)
        #expect(chunk.lineStart == nil)
        #expect(chunk.lineEnd == nil)

        let hits = try await store.search("indexed text", limit: 5)
        #expect(hits.first?.lineStart == nil)
    }

    /// The migration runs on stores that already contain chunks; it must add the columns without
    /// disturbing the rows already there.
    @Test("reopening an existing store adds the columns and preserves its documents")
    func migrationPreservesExistingRows() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let first = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))
        try await first.upsert(.init(id: "doc-1", type: "note", title: "Kept", body: "thermal throttling marker"))
        try await first.clearChunkLineRangesForTesting()

        // A second open re-runs the schema install, exercising the ALTER TABLE guard.
        let reopened = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))
        #expect(try await reopened.counts().documentCount == 1)

        // Re-ingesting the same document backfills the range.
        try await reopened.upsert(.init(id: "doc-1", type: "note", title: "Kept", body: "thermal throttling marker"))
        let chunk = try #require(try await reopened.chunkSummaries(documentID: "doc-1").first)
        #expect(chunk.lineStart == 1)
    }
}

extension IndexStore {
    /// Blanks the line columns to stand in for rows written by the pre-migration schema.
    func clearChunkLineRangesForTesting() throws {
        try db.exec("UPDATE chunks SET line_start = NULL, line_end = NULL")
    }
}
