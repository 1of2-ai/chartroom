import Foundation
import Testing
@testable import IndexEngine

/// Embedding IDs are space-scoped (`chunk:embedding:<space>`), but deletion was not: re-ingesting
/// a document under one model removed every model's vectors for it. Latent while one space is
/// active per store, data loss the moment two share one.
@Suite("Embedding space isolation")
struct EmbeddingSpaceIsolationTests {
    private func temporaryStorePath() -> String {
        FileManager.default.temporaryDirectory
            .appending(path: "chartroom-space-isolation-\(UUID().uuidString).sqlite")
            .path
    }

    private func document() -> IndexedObject {
        .init(
            id: "doc-1",
            type: "note",
            title: "Thermal",
            body: "the m-series rig is thermal throttling on long capture runs"
        )
    }

    @Test("re-ingesting under one model preserves another model's vectors")
    func reingestPreservesOtherSpaces() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let alpha = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-alpha", dimension: 64))
        try await alpha.upsert(document())
        let alphaCount = try await alpha.counts().embeddingCount
        #expect(alphaCount > 0)

        // Same content, so chunk IDs are identical and both spaces describe the same chunks.
        let beta = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-beta", dimension: 64))
        try await beta.upsert(document())
        #expect(try await beta.counts().embeddingCount == alphaCount)

        #expect(
            try await alpha.counts().embeddingCount == alphaCount,
            "model-alpha's vectors must survive a re-ingest performed under model-beta"
        )
    }

    @Test("re-ingesting under the same model replaces rather than duplicates")
    func reingestReplacesOwnSpace() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let store = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-alpha", dimension: 64))
        try await store.upsert(document())
        let first = try await store.counts().embeddingCount

        try await store.upsert(document())

        #expect(try await store.counts().embeddingCount == first)
    }

    /// Deleting the document is the case where clearing every space *is* correct — nothing can
    /// reference those chunks again.
    @Test("deleting a document clears every embedding space")
    func deleteClearsAllSpaces() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let alpha = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-alpha", dimension: 64))
        try await alpha.upsert(document())
        let beta = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-beta", dimension: 64))
        try await beta.upsert(document())

        #expect(try await beta.delete(id: "doc-1"))

        #expect(try await alpha.counts().embeddingCount == 0)
        #expect(try await beta.counts().embeddingCount == 0)
    }

    /// Content change retires the old chunk IDs, so their vectors are orphans in every space.
    @Test("chunks that disappear take every space's vectors with them")
    func replacedChunksAreFullyCleared() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let alpha = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-alpha", dimension: 64))
        try await alpha.upsert(document())

        let beta = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-beta", dimension: 64))
        try await beta.upsert(.init(id: "doc-1", type: "note", title: "Thermal", body: "entirely different text now"))

        // Alpha never embedded the new chunk, and its old chunk no longer exists.
        #expect(try await alpha.counts().embeddingCount == 0)
        #expect(try await beta.counts().embeddingCount > 0)
    }

    /// C3: swapping the embedder leaves every vector in a space the new model cannot read.
    /// Retrieval used to go quiet with no error and no degraded flag, which is indistinguishable
    /// from an empty index.
    @Test("an embedder swap is reported rather than returning silence")
    func embedderSwapIsReported() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let original = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-alpha", dimension: 64))
        try await original.upsert(document())
        let healthy = try await original.searchDetailed("thermal throttling", limit: 5)
        #expect(!healthy.hits.isEmpty)
        #expect(healthy.diagnostics.embeddingSpaceMismatch == false)

        // Same store, different model identity — exactly what a regenerated Jina manifest does.
        let swapped = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-beta", dimension: 64))
        let coverage = try await swapped.embeddingSpaceCoverage()
        #expect(coverage.isOrphaned)
        #expect(coverage.activeSpaceEmbeddingCount == 0)
        #expect(coverage.totalEmbeddingCount > 0)

        let afterSwap = try await swapped.searchDetailed("thermal throttling", limit: 5)
        #expect(afterSwap.diagnostics.embeddingSpaceMismatch)
        #expect(afterSwap.diagnostics.degraded)
    }

    /// Detecting the orphan is only half of C3 — without a rebuild there is no way back, because
    /// the vectors are unreadable and nothing re-embeds them. The chunk text is still in the store,
    /// so recovery needs no source files.
    @Test("a rebuild restores an index orphaned by an embedder swap")
    func rebuildRecoversOrphanedIndex() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let original = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-alpha", dimension: 64))
        try await original.upsert(document())

        let swapped = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-beta", dimension: 64))
        #expect(try await swapped.searchDetailed("thermal throttling", limit: 5).diagnostics.embeddingSpaceMismatch)

        let summary = try await swapped.rebuildActiveEmbeddingSpace()
        #expect(summary.rebuiltChunkCount > 0)
        #expect(summary.isComplete)

        let recovered = try await swapped.searchDetailed("thermal throttling", limit: 5)
        #expect(!recovered.hits.isEmpty)
        #expect(recovered.diagnostics.embeddingSpaceMismatch == false)
        #expect(try await swapped.embeddingSpaceCoverage().isOrphaned == false)

        // The rebuild adds a space rather than destroying one: the original model still reads its
        // own vectors, which is the same isolation guarantee re-ingest honours.
        #expect(try await original.embeddingSpaceCoverage().activeSpaceEmbeddingCount > 0)
    }

    @Test("a rebuild is idempotent and resumable")
    func rebuildIsIdempotent() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let original = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-alpha", dimension: 64))
        try await original.upsert(document())
        let swapped = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "model-beta", dimension: 64))

        let first = try await swapped.rebuildActiveEmbeddingSpace()
        #expect(first.rebuiltChunkCount > 0)
        let countAfterFirst = try await swapped.counts().embeddingCount

        // Nothing is left missing, so a second pass has no work and cannot duplicate vectors.
        let second = try await swapped.rebuildActiveEmbeddingSpace()
        #expect(second.rebuiltChunkCount == 0)
        #expect(try await swapped.counts().embeddingCount == countAfterFirst)
    }

    @Test("an empty index is not reported as a space mismatch")
    func emptyIndexIsNotAMismatch() async throws {
        let store = try IndexStore(path: ":memory:", embedder: HashingEmbedder(modelID: "model-alpha", dimension: 64))

        let response = try await store.searchDetailed("anything at all", limit: 5)

        #expect(response.hits.isEmpty)
        #expect(response.diagnostics.embeddingSpaceMismatch == false)
        #expect(try await store.embeddingSpaceCoverage().isOrphaned == false)
    }
}
