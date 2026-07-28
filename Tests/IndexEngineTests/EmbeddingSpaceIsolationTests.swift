import ChartroomTestSupport
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

    @Test("exact and FTS retrieval do not require an active-space embedding")
    func lexicalRetrievalSurvivesEmbedderSwap() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let original = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "model-alpha", dimension: 64)
        )
        try await original.upsert(.init(
            id: "doc-lexical",
            type: "note",
            title: "Needle Exact Title",
            body: "a bodyonlymarker that the full text index can retrieve"
        ))

        let swapped = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "model-beta", dimension: 64)
        )
        let exactOnly = RetrievalProfile(
            id: "exact-only-test",
            version: 1,
            maxFTSCandidates: 0,
            maxVectorCandidates: 0,
            maxSnippets: 10
        )
        let exact = try await swapped.searchDetailed(
            "Needle Exact Title",
            limit: 5,
            profile: exactOnly
        )
        #expect(exact.hits.map(\.documentID) == ["doc-lexical"])
        #expect(exact.hits.first?.exactRank == 1)
        #expect(exact.hits.first?.embeddingSpaceID == nil)

        let ftsOnly = RetrievalProfile(
            id: "fts-only-test",
            version: 1,
            maxFTSCandidates: 10,
            maxVectorCandidates: 0,
            maxSnippets: 0
        )
        let fts = try await swapped.searchDetailed(
            "bodyonlymarker",
            limit: 5,
            profile: ftsOnly
        )
        #expect(fts.hits.map(\.documentID) == ["doc-lexical"])
        #expect(fts.hits.first?.keywordRank == 1)
        #expect(fts.hits.first?.embeddingSpaceID == nil)
    }

    @Test("a foreign-space filter degrades to lexical retrieval without querying the active embedder")
    func foreignSpaceFilterSkipsActiveVectorScoring() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let recorder = QueryEmbedRecorder()
        let alpha = try IndexStore(
            path: path,
            embedder: QueryRecordingEmbedder(modelID: "model-alpha", recorder: recorder)
        )
        try await alpha.upsert(document())
        let beta = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "model-beta", dimension: 64)
        )
        try await beta.upsert(document())

        let response = try await alpha.searchDetailed(
            "thermal throttling",
            filters: .init(embeddingSpaceID: "model-beta:64"),
            limit: 5,
            allowDegradedResults: true
        )

        #expect(response.hits.map(\.documentID) == ["doc-1"])
        #expect(response.hits.first?.embeddingSpaceID == "model-beta:64")
        #expect(response.hits.first?.vectorRank == nil)
        #expect(response.diagnostics.degraded)
        #expect(response.diagnostics.missingChannels == [.vector])
        #expect(await recorder.queryCount == 0)
    }

    @Test("strict search rejects an orphaned active embedding space")
    func strictSearchRejectsOrphanedActiveSpace() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let original = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "model-alpha", dimension: 64)
        )
        try await original.upsert(document())

        let engine = try await IndexEngine.open(
            storeURL: URL(filePath: path),
            configuration: .init(embedder: HashingEmbedder(modelID: "model-beta", dimension: 64))
        )
        do {
            _ = try await engine.search(.init(
                query: "thermal throttling",
                allowDegradedResults: false
            ))
            Issue.record("Expected strict search against an orphaned active space to throw")
        } catch let error as IndexEngineError {
            #expect(error.category == .embeddingSpaceUnavailable)
            #expect(error.code == "index.search.embedding-space-mismatch")
        }
    }

    @Test("embedding-space filters do not multiply lexical candidate ranks")
    func embeddingSpaceFiltersDoNotDuplicateLexicalRanks() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let store = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "model-alpha", dimension: 64)
        )
        try await store.upsert(.init(
            id: "doc-duplicate",
            type: "note",
            title: "Needle Exact Title",
            body: "bodyonlymarker for filtered full text retrieval"
        ))

        let db = try SQLite(path: path)
        try db.exec("""
        INSERT INTO embeddings(
          id,chunk_id,embedding_space_id,model_id,model_version,dimension,modality,prompt_kind,
          vector_backend_id,vector_backend_version,vector_hash,created_at
        )
        SELECT id || ':duplicate',chunk_id,embedding_space_id,model_id,model_version,dimension,
          modality,prompt_kind,vector_backend_id,vector_backend_version,vector_hash,created_at
        FROM embeddings
        LIMIT 1;
        """)
        let filter = SearchFilters(embeddingSpaceID: "model-alpha:64")

        let exact = try await store.searchDetailed(
            "Needle Exact Title",
            filters: filter,
            limit: 5,
            profile: .init(
                id: "exact-filter-test",
                version: 1,
                maxFTSCandidates: 0,
                maxVectorCandidates: 0,
                maxSnippets: 5
            )
        )
        #expect(exact.hits.count == 1)
        #expect(exact.hits.first?.exactRank == 1)

        let fts = try await store.searchDetailed(
            "bodyonlymarker",
            filters: filter,
            limit: 5,
            profile: .init(
                id: "fts-filter-test",
                version: 1,
                maxFTSCandidates: 5,
                maxVectorCandidates: 0,
                maxSnippets: 0
            )
        )
        #expect(fts.hits.count == 1)
        #expect(fts.hits.first?.keywordRank == 1)
    }

    @Test("strict foreign-space search returns the typed engine mismatch")
    func strictForeignSpaceSearchThrowsTypedMismatch() async throws {
        let path = temporaryStorePath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let recorder = QueryEmbedRecorder()
        let alpha = try IndexStore(
            path: path,
            embedder: QueryRecordingEmbedder(modelID: "model-alpha", recorder: recorder)
        )
        try await alpha.upsert(document())
        let beta = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "model-beta", dimension: 64)
        )
        try await beta.upsert(document())

        let engine = try await IndexEngine.open(
            storeURL: URL(filePath: path),
            configuration: .init(embedder: QueryRecordingEmbedder(modelID: "model-alpha", recorder: recorder))
        )
        do {
            _ = try await engine.search(.init(
                query: "thermal throttling",
                filters: .init(embeddingSpaceID: "model-beta:64"),
                allowDegradedResults: false
            ))
            Issue.record("Expected strict search across a foreign embedding space to throw")
        } catch let error as IndexEngineError {
            #expect(error.category == .embeddingSpaceUnavailable)
            #expect(error.code == "index.search.embedding-space-mismatch")
        }
        #expect(await recorder.queryCount == 0)
    }

    @Test("upsert rejects an object that claims a foreign embedding space")
    func upsertRejectsForeignSpaceClaim() async throws {
        let store = try IndexStore(
            path: ":memory:",
            embedder: HashingEmbedder(modelID: "model-alpha", dimension: 64)
        )
        let object = IndexedObject(
            id: "lying-object",
            type: "note",
            title: "Mismatch",
            body: "this object claims vectors from another space",
            embeddingSpaceID: "model-beta:64"
        )

        do {
            try await store.upsert(object)
            Issue.record("Expected a foreign embedding-space claim to be rejected")
        } catch let error as IndexStoreError {
            #expect(
                error == .embeddingSpaceMismatch(
                    expected: "model-alpha:64",
                    actual: "model-beta:64"
                )
            )
        } catch {
            Issue.record("Expected IndexStoreError, got \(type(of: error)): \(error)")
        }
        #expect(try await store.counts().documentCount == 0)
        #expect(try await store.counts().embeddingCount == 0)
    }
}

private actor QueryEmbedRecorder {
    private(set) var queryCount = 0

    func recordQuery() {
        queryCount += 1
    }
}

private struct QueryRecordingEmbedder: FixtureEmbedder {
    let modelID: String
    let dimension = 64
    let recorder: QueryEmbedRecorder

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        if kind == .query {
            await recorder.recordQuery()
        }
        return [Float](repeating: 1 / 8, count: dimension)
    }
}
