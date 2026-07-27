import ChartroomTestSupport
import Foundation
import Testing
@testable import IndexEngine

@Suite("IndexStore — hybrid BM25 + vector search")
struct IndexStoreTests {
    private func makeStore() throws -> IndexStore {
        try IndexStore(path: ":memory:", embedder: HashingEmbedder(dimension: 256))
    }

    @Test("SQLite connections wait briefly for busy writers")
    func sqliteBusyTimeoutIsConfigured() throws {
        let db = try SQLite(path: ":memory:")
        let statement = try db.prepare("PRAGMA busy_timeout")

        #expect(try statement.step())
        #expect(statement.int(0) == 5_000)
    }

    @Test("keyword (BM25) ranks the doc with the query terms first")
    func keywordRanksExactTerm() async throws {
        let store = try makeStore()
        try await store.upsert(.init(id: "a", type: "note", title: "Thermal",
            body: "the m-series rig is thermal throttling on long capture runs"))
        try await store.upsert(.init(id: "b", type: "note", title: "Budget",
            body: "the studio budget ask is sitting with finance since tuesday"))
        let hits = try await store.search("thermal throttling", limit: 5)
        #expect(hits.first?.documentID == "a")
        #expect(hits.first?.keywordRank == 1)
    }

    @Test("vector signal contributes to the fused ranking")
    func vectorContributes() async throws {
        let store = try makeStore()
        try await store.upsert(.init(id: "a", type: "note", title: "x",
            body: "rig throttling thermal capture pipeline"))
        try await store.upsert(.init(id: "b", type: "note", title: "y",
            body: "finance budget approval quarterly forecast"))
        let hits = try await store.search("throttling rig", limit: 5)
        #expect(hits.first?.documentID == "a")
        #expect(hits.first?.vectorRank != nil)
    }

    @Test("hard scope locks retrieval to one cluster")
    func hardScopeFilters() async throws {
        let store = try makeStore()
        try await store.upsert(.init(id: "a", type: "note", title: "x",
            body: "codec licensing question for the product team", clusterID: "c1"))
        try await store.upsert(.init(id: "b", type: "note", title: "y",
            body: "codec licensing question for the product team", clusterID: "c2"))
        let hits = try await store.search("codec licensing", scope: .cluster("c1", hard: true), limit: 5)
        #expect(hits.contains { $0.documentID == "a" })
        #expect(!hits.contains { $0.documentID == "b" })
    }

    @Test("soft scope boosts in-cluster hits without excluding others")
    func softScopeBoosts() async throws {
        let store = try makeStore()
        try await store.upsert(.init(id: "in", type: "note", title: "x",
            body: "thermal pad spec discussion", clusterID: "c1"))
        try await store.upsert(.init(id: "out", type: "note", title: "y",
            body: "thermal pad spec discussion", clusterID: "c2"))
        let hits = try await store.search("thermal pad", scope: .cluster("c1"), limit: 5)
        #expect(hits.first?.documentID == "in")
        #expect(hits.contains { $0.documentID == "out" })
    }

    @Test("upsert is idempotent — re-indexing does not duplicate")
    func upsertIdempotent() async throws {
        let store = try makeStore()
        let obj = IndexedObject(id: "a", type: "note", title: "x", body: "unique zebra token thermal")
        try await store.upsert(obj)
        try await store.upsert(obj)
        #expect(try await store.count() == 1)
        let hits = try await store.search("zebra", limit: 5)
        #expect(hits.count == 1)
    }

    @Test("overlapping upserts cannot let stale embedding completion overwrite newer content")
    func overlappingUpsertsKeepNewestContent() async throws {
        let gate = UpsertRaceGate()
        let store = try IndexStore(path: ":memory:", embedder: GatedRaceEmbedder(gate: gate))

        let olderTask = Task {
            try await store.upsert(.init(id: "doc", type: "note", title: "Race", body: "older content"))
        }
        await gate.waitForOlderEmbed()
        try await store.upsert(.init(id: "doc", type: "note", title: "Race", body: "newer content"))
        try await olderTask.value

        let newerHits = try await store.search("newer", limit: 5)
        #expect(newerHits.map(\.documentID) == ["doc"])
        #expect(newerHits.first?.snippet.contains("newer") == true)

        let olderHits = try await store.search("older", limit: 5)
        #expect(olderHits.first?.snippet.contains("older") != true)
    }

    @Test("search hits retain source and policy provenance")
    func searchHitsRetainProvenance() async throws {
        let store = try makeStore()
        let sourceURI = URL(filePath: "/tmp/capture.md")
        try await store.upsert(.init(
            id: "a",
            type: "note",
            title: "Capture",
            body: "thermal capture plan",
            sourceID: "local-files",
            sourceURI: sourceURI,
            policyID: "default",
            representationID: "a:representation:plainText"
        ))

        let hit = try await store.search("thermal capture", limit: 5).first
        #expect(hit?.documentID == "a")
        #expect(hit?.chunkID != "a")
        #expect(hit?.sourceID == "local-files")
        #expect(hit?.sourceURI == sourceURI)
        #expect(hit?.policyID == "default")
        #expect(hit?.representationID?.hasPrefix("a:representation:plainText:") == true)
        #expect(hit?.embeddingSpaceID == "hashing-mock-v1:256")
    }

    @Test("metadata columns migrate additively on existing stores")
    func metadataColumnsMigrateAdditively() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("legacy-index-\(UUID().uuidString).sqlite")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        do {
            let db = try SQLite(path: path)
            try db.exec("""
            CREATE TABLE objects (
              id TEXT PRIMARY KEY, type TEXT NOT NULL, title TEXT NOT NULL,
              cluster_id TEXT, model_id TEXT NOT NULL, updated_at REAL NOT NULL
            );
            """)
        }

        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 256))
        try await store.upsert(.init(
            id: "legacy",
            type: "note",
            title: "Legacy",
            body: "legacy migration retrieval",
            sourceID: "local-files",
            sourceURI: URL(filePath: "/tmp/legacy.md"),
            policyID: "policy-a",
            representationID: "legacy:representation:plainText"
        ))

        let hit = try await store.search("legacy retrieval", limit: 5).first
        #expect(hit?.documentID == "legacy")
        #expect(hit?.sourceID == "local-files")
        #expect(hit?.sourceURI == URL(filePath: "/tmp/legacy.md"))
        #expect(hit?.policyID == "policy-a")
        #expect(hit?.representationID?.hasPrefix("legacy:representation:plainText:") == true)
    }

    @Test("failure recoverability migrates additively and persists typed values")
    func failureRecoverabilityMigratesAndPersists() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("legacy-failures-\(UUID().uuidString).sqlite")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        do {
            let db = try SQLite(path: path)
            try db.exec("""
            CREATE TABLE failures (
              id TEXT PRIMARY KEY,
              category TEXT NOT NULL,
              message TEXT NOT NULL,
              detail TEXT NOT NULL,
              source_id TEXT,
              document_id TEXT,
              is_recoverable INTEGER NOT NULL,
              occurred_at REAL NOT NULL
            );
            INSERT INTO failures(
              id,category,message,detail,source_id,document_id,is_recoverable,occurred_at
            ) VALUES(
              'legacy-failure','storageFailure','legacy message','legacy detail',NULL,'legacy-doc',0,1
            );
            """)
        }

        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 256))
        try await store.recordFailure(.init(
            id: "typed-failure",
            category: .extractionFailure,
            message: "Needs a readable source",
            detail: "fixture detail",
            documentID: "typed-doc",
            recoverability: .needsUserAction,
            occurredAt: Date(timeIntervalSince1970: 2)
        ))

        let failures = try await store.failureSnapshots(limit: 10)
        let legacy = try #require(failures.first { $0.id == "legacy-failure" })
        #expect(legacy.recoverability == .unrecoverable)
        #expect(!legacy.isRecoverable)

        let typed = try #require(failures.first { $0.id == "typed-failure" })
        #expect(typed.recoverability == .needsUserAction)
        #expect(typed.isRecoverable)
    }

    @Test("delete removes the object from every index")
    func deleteRemovesEverywhere() async throws {
        let store = try makeStore()
        try await store.upsert(.init(id: "a", type: "note", title: "x", body: "unique zebra token thermal"))
        try await store.delete(id: "a")
        #expect(try await store.count() == 0)
        #expect(try await store.search("zebra", limit: 5).isEmpty)
    }

    @Test("superseded and deleted chunks remove their embedding vector blobs")
    func upsertAndDeleteRemoveDeadVectors() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("vector-cleanup-\(UUID().uuidString).sqlite")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))
        try await store.upsert(.init(id: "doc", type: "note", title: "Doc", body: "first vector body"))
        #expect(try Self.rawCount(path: path, table: "embeddings") == 1)
        #expect(try Self.rawCount(path: path, table: "vectors") == 1)

        try await store.upsert(.init(id: "doc", type: "note", title: "Doc", body: "second vector body"))
        #expect(try Self.rawCount(path: path, table: "embeddings") == 1)
        #expect(try Self.rawCount(path: path, table: "vectors") == 1)

        try await store.delete(id: "doc")
        #expect(try Self.rawCount(path: path, table: "embeddings") == 0)
        #expect(try Self.rawCount(path: path, table: "vectors") == 0)
    }

    @Test("repeated edits do not accumulate superseded chunk or representation rows")
    func upsertAndDeleteRemoveSupersededRows() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("row-cleanup-\(UUID().uuidString).sqlite")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))
        for revision in 0..<5 {
            try await store.upsert(.init(id: "doc", type: "note", title: "Doc", body: "revision \(revision) body"))
        }
        // One active chunk, one representation — not one per edit.
        #expect(try Self.rawCount(path: path, table: "chunks") == 1)
        #expect(try Self.rawCount(path: path, table: "representations") == 1)

        // Reverting to earlier content must keep working (chunk IDs repeat).
        try await store.upsert(.init(id: "doc", type: "note", title: "Doc", body: "revision 0 body"))
        #expect(try Self.rawCount(path: path, table: "chunks") == 1)
        #expect(try await store.search("revision", limit: 5).count == 1)

        try await store.delete(id: "doc")
        #expect(try Self.rawCount(path: path, table: "chunks") == 0)
        #expect(try Self.rawCount(path: path, table: "representations") == 0)
        #expect(try Self.rawCount(path: path, table: "representation_lineages") == 0)
    }

    @Test("repeated identical chunk text keeps distinct chunk identities")
    func repeatedChunkTextGetsDistinctIdentities() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("chunk-identity-\(UUID().uuidString).sqlite")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        // Three identical 1,000-character paragraphs. The overlap-window chunker makes the
        // middle chunks textually identical, so identity must disambiguate by occurrence —
        // a purely content-addressed ID would collide and silently drop a chunk row.
        let paragraph = String(repeating: "note ", count: 199) + "wrap\n"
        let body = paragraph + paragraph + paragraph

        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))
        try await store.upsert(.init(id: "doc", type: "note", title: "Doc", body: body))

        let chunkCount = try Self.rawCount(path: path, table: "chunks")
        let distinctIDs = try Self.rawDistinctCount(path: path, table: "chunks", column: "id")
        #expect(chunkCount == distinctIDs)
        #expect(chunkCount >= 3)

        // Determinism: re-upserting unchanged content reuses the same identities.
        try await store.upsert(.init(id: "doc", type: "note", title: "Doc", body: body))
        #expect(try Self.rawCount(path: path, table: "chunks") == chunkCount)
    }

    @Test("chunk summaries expose a document's active chunks in ordinal order")
    func chunkSummariesProjection() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("chunk-summaries-\(UUID().uuidString).sqlite")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        let body = String(repeating: "quill ", count: 600).trimmingCharacters(in: .whitespaces)
        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))
        try await store.upsert(.init(id: "doc", type: "note", title: "Doc", body: body))

        let summaries = try await store.chunkSummaries(documentID: "doc")
        #expect(summaries.count >= 2)
        #expect(summaries.map(\.ordinal) == Array(0..<summaries.count))
        #expect(summaries.allSatisfy { $0.hasEmbedding })
        #expect(summaries.allSatisfy { !$0.text.isEmpty && !$0.contentHash.isEmpty })
        #expect(summaries.first?.characterStart == 0)

        // Unknown documents are an empty projection, not an error.
        #expect(try await store.chunkSummaries(documentID: "missing").isEmpty)
    }

    @Test("adjacent chunks overlap and cover the whole document")
    func chunkOverlapContinuity() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("chunk-overlap-\(UUID().uuidString).sqlite")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        // No newlines, so every boundary is a hard split with the configured overlap.
        let body = String(repeating: "lumen ", count: 900).trimmingCharacters(in: .whitespaces)

        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))
        try await store.upsert(.init(id: "doc", type: "note", title: "Doc", body: body))

        let offsets = try Self.rawChunkOffsets(path: path)
        #expect(offsets.count >= 3)
        #expect(offsets.first?.characterStart == 0)
        // Whitespace trimming may drop a trailing space per boundary, nothing more.
        #expect((offsets.last?.characterEnd ?? 0) >= body.count - 1)
        for (previous, next) in zip(offsets, offsets.dropFirst()) {
            #expect(next.characterStart > previous.characterStart)
            #expect(
                next.characterStart < previous.characterEnd,
                "chunks must overlap: \(next.characterStart) is past \(previous.characterEnd)"
            )
        }
    }

    @Test("two stores on the same file coexist through the busy timeout")
    func twoStoresOneFileConcurrentWrites() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("two-stores-\(UUID().uuidString).sqlite")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        let first = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))
        let second = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                for index in 0..<15 {
                    try await first.upsert(.init(id: "a-\(index)", type: "note", title: "A \(index)", body: "alpha body \(index)"))
                }
            }
            group.addTask {
                for index in 0..<15 {
                    try await second.upsert(.init(id: "b-\(index)", type: "note", title: "B \(index)", body: "beta body \(index)"))
                }
            }
            try await group.waitForAll()
        }

        let counts = try await first.counts()
        #expect(counts.documentCount == 30)
    }

    @Test("search during ingest returns consistent results without errors")
    func searchDuringIngest() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("search-during-ingest-\(UUID().uuidString).sqlite")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))
        try await store.upsert(.init(id: "seed", type: "note", title: "Seed", body: "stable searchable seed"))

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                for index in 0..<40 {
                    try await store.upsert(.init(id: "doc-\(index)", type: "note", title: "Doc \(index)", body: "flowing corpus \(index)"))
                }
            }
            group.addTask {
                for _ in 0..<20 {
                    // Every interleaved search must complete and keep seeing the seed.
                    let hits = try await store.search("seed", limit: 5)
                    #expect(hits.contains { $0.documentID == "seed" })
                }
            }
            try await group.waitForAll()
        }

        let counts = try await store.counts()
        #expect(counts.documentCount == 41)
    }

    /// The legacy FTS twin is no longer created at all. Asserting it holds no rows was the weaker
    /// claim available while the table still existed; now the table's absence is the guarantee, and
    /// a reintroduced dual-write would fail at the write rather than in a row count.
    @Test("the legacy object FTS twin is not created")
    func objectProjectionDoesNotMaintainLegacyFTS() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("legacy-objects-fts-\(UUID().uuidString).sqlite")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))
        try await store.upsert(.init(id: "doc", type: "note", title: "Doc", body: "legacy object fts marker"))
        #expect(try Self.rawCount(path: path, table: "objects") == 1)
        #expect(try Self.tableExists(path: path, table: "objects_fts") == false)
        #expect(try Self.tableExists(path: path, table: "search_diagnostics") == false)
    }

    @Test("chunk metadata stores byte ranges and token offsets, not character placeholders")
    func chunkMetadataStoresByteAndTokenRanges() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("chunk-offsets-\(UUID().uuidString).sqlite")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        let body = "å " + String(repeating: "alpha ", count: 340) + "\n" + String(repeating: "beta ", count: 340)
        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 64))
        try await store.upsert(.init(id: "doc", type: "note", title: "Doc", body: body))

        let rows = try Self.rawChunkOffsets(path: path)
        #expect(rows.count > 1)
        #expect(rows.allSatisfy { $0.byteStart <= $0.byteEnd })
        #expect(rows.allSatisfy { $0.characterStart <= $0.characterEnd })
        #expect(rows.allSatisfy { $0.tokenStart <= $0.tokenEnd })

        let first = try #require(rows.first)
        #expect(first.byteEnd > first.characterEnd)

        let later = try #require(rows.dropFirst().first)
        #expect(later.tokenStart > 0)
    }

    @Test("blank queries return no arbitrary vector hits")
    func blankQueryReturnsNoHits() async throws {
        let store = try makeStore()
        try await store.upsert(.init(id: "a", type: "note", title: "x", body: "thermal capture"))
        #expect(try await store.search("   \n\t  ", limit: 5).isEmpty)
    }

    @Test("hard scope constrains candidates before the global pool is trimmed")
    func hardScopeConstrainCandidatesBeforeLimit() async throws {
        let store = try makeStore()
        for i in 0..<45 {
            try await store.upsert(.init(
                id: "out-\(i)",
                type: "note",
                title: "out \(i)",
                body: "codec licensing question for the product team",
                clusterID: "outside"
            ))
        }
        try await store.upsert(.init(
            id: "target",
            type: "note",
            title: "target",
            body: "codec licensing question for the product team",
            clusterID: "inside"
        ))

        let hits = try await store.search("codec licensing", scope: .cluster("inside", hard: true), limit: 5)
        #expect(hits.map(\.documentID) == ["target"])
    }

    @Test("search ignores rows indexed with another embedding model")
    func searchFiltersToCurrentModelSpace() async throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("index-\(UUID().uuidString).sqlite")
            .path
        defer { try? FileManager.default.removeItem(atPath: path) }

        do {
            let oldStore = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "old-model", dimension: 256))
            try await oldStore.upsert(.init(id: "a", type: "note", title: "old", body: "thermal capture"))
        }

        let currentStore = try IndexStore(path: path, embedder: HashingEmbedder(modelID: "current-model", dimension: 256))
        #expect(try await currentStore.search("thermal", limit: 5).isEmpty)
    }

    @Test("counts binds embedding space IDs instead of interpolating SQL")
    func countsBindEmbeddingSpaceID() async throws {
        let store = try IndexStore(path: ":memory:", embedder: HashingEmbedder(modelID: "hashing'mock", dimension: 16))
        try await store.upsert(.init(id: "a", type: "note", title: "Quote", body: "quoted model id"))

        let counts = try await store.counts()
        #expect(counts.documentCount == 1)
        #expect(counts.embeddingCount == 1)
    }

    @Test("search clamps huge caller-provided limits before computing candidate pools")
    func searchClampsHugeLimit() async throws {
        let store = try makeStore()
        try await store.upsert(.init(id: "a", type: "note", title: "Thermal", body: "thermal capture"))

        let hits = try await store.search("thermal", limit: Int.max)
        #expect(hits.first?.documentID == "a")
    }

    @Test("exact title matching treats LIKE wildcards literally")
    func exactTitleMatchingEscapesWildcards() async throws {
        let store = try makeStore()
        try await store.upsert(.init(id: "literal", type: "note", title: "zzz 100%_done", body: "literal marker"))
        try await store.upsert(.init(id: "wildcard", type: "note", title: "aaa 100Xdone", body: "unrelated marker"))

        let hits = try await store.search("100%_done", limit: 1)
        #expect(hits.map(\.documentID) == ["literal"])
    }

    @Test("snippets use ranges from the original string under Unicode case folding")
    func snippetRangeUsesOriginalStringIndices() async throws {
        let store = try makeStore()
        let prefix = String(repeating: "İ", count: 260)
        try await store.upsert(.init(id: "unicode", type: "note", title: "Unicode", body: "\(prefix) needle phrase"))

        let hit = try #require(try await store.search("needle", limit: 1).first)
        #expect(hit.documentID == "unicode")
        #expect(hit.snippet.contains("needle"))
    }

    @Test("MIME image content types route through image embedding")
    func mimeImageContentTypesUseImageEmbedding() async throws {
        let recorder = ImageEmbedRecorder()
        let store = try IndexStore(path: ":memory:", embedder: ImageFixtureEmbedder(recorder: recorder))
        let imageURL = FileManager.default.temporaryDirectory.appending(path: "image-\(UUID().uuidString).png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: imageURL)
        defer { try? FileManager.default.removeItem(at: imageURL) }

        try await store.upsert(.init(
            id: "image",
            type: "image/png",
            title: "Image",
            body: "image filename fallback",
            sourceURI: imageURL
        ))

        #expect(await recorder.urls == [imageURL])
    }

    @Test("upsert rejects embeddings with the wrong dimension")
    func upsertRejectsWrongEmbeddingDimension() async throws {
        let store = try IndexStore(path: ":memory:", embedder: WrongDimensionEmbedder(dimension: 8, actualCount: 3))

        do {
            try await store.upsert(.init(id: "a", type: "note", title: "x", body: "thermal capture"))
            #expect(Bool(false), "Expected a dimension mismatch")
        } catch let error as IndexStoreError {
            #expect(error == .embeddingDimensionMismatch(kind: .document, expected: 8, actual: 3))
        }
    }

    @Test("search rejects query embeddings with the wrong dimension")
    func searchRejectsWrongQueryEmbeddingDimension() async throws {
        let store = try IndexStore(path: ":memory:", embedder: WrongQueryDimensionEmbedder())
        try await store.upsert(.init(id: "a", type: "note", title: "x", body: "thermal capture"))

        do {
            _ = try await store.search("thermal", limit: 5)
            #expect(Bool(false), "Expected a query dimension mismatch")
        } catch let error as IndexStoreError {
            #expect(error == .embeddingDimensionMismatch(kind: .query, expected: 8, actual: 3))
        }
    }

    private static func rawCount(path: String, table: String) throws -> Int {
        let db = try SQLite(path: path)
        let statement = try db.prepare("SELECT COUNT(*) FROM \(table)")
        return try statement.step() ? statement.int(0) : 0
    }

    private static func tableExists(path: String, table: String) throws -> Bool {
        let db = try SQLite(path: path)
        let statement = try db.prepare("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1")
        statement.bind(1, table)
        return try statement.step() && statement.int(0) > 0
    }

    private static func rawDistinctCount(path: String, table: String, column: String) throws -> Int {
        let db = try SQLite(path: path)
        let statement = try db.prepare("SELECT COUNT(DISTINCT \(column)) FROM \(table)")
        return try statement.step() ? statement.int(0) : 0
    }

    private static func rawChunkOffsets(path: String) throws -> [ChunkOffsetRow] {
        let db = try SQLite(path: path)
        let statement = try db.prepare("""
        SELECT ordinal,byte_start,byte_end,character_start,character_end,token_start,token_end
        FROM chunks ORDER BY ordinal ASC
        """)
        var rows: [ChunkOffsetRow] = []
        while try statement.step() {
            rows.append(
                ChunkOffsetRow(
                    ordinal: statement.int(0),
                    byteStart: statement.int(1),
                    byteEnd: statement.int(2),
                    characterStart: statement.int(3),
                    characterEnd: statement.int(4),
                    tokenStart: statement.int(5),
                    tokenEnd: statement.int(6)
                )
            )
        }
        return rows
    }
}

private struct ChunkOffsetRow {
    var ordinal: Int
    var byteStart: Int
    var byteEnd: Int
    var characterStart: Int
    var characterEnd: Int
    var tokenStart: Int
    var tokenEnd: Int
}

private struct WrongDimensionEmbedder: FixtureEmbedder {
    var modelID = "wrong-dimension"
    var dimension: Int
    var actualCount: Int

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        [Float](repeating: 1, count: actualCount)
    }
}

private struct WrongQueryDimensionEmbedder: FixtureEmbedder {
    var modelID = "wrong-query-dimension"
    var dimension = 8

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        [Float](repeating: 1, count: kind == .document ? dimension : 3)
    }
}

private actor UpsertRaceGate {
    private var olderStarted = false
    private var olderStartedContinuation: CheckedContinuation<Void, Never>?

    func delayIfOlder(_ text: String) async throws {
        guard text.contains("older") else { return }
        olderStarted = true
        olderStartedContinuation?.resume()
        olderStartedContinuation = nil
        try await Task.sleep(for: .milliseconds(200))
    }

    func waitForOlderEmbed() async {
        if olderStarted { return }
        await withCheckedContinuation { continuation in
            olderStartedContinuation = continuation
        }
    }
}

private struct GatedRaceEmbedder: FixtureEmbedder {
    let modelID = "race-fixture"
    let dimension = 4
    let gate: UpsertRaceGate

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        try await gate.delayIfOlder(text)
        if text.contains("older") {
            return [0, 1, 0, 0]
        }
        return [1, 0, 0, 0]
    }
}

private actor ImageEmbedRecorder {
    private(set) var urls: [URL] = []

    func record(_ url: URL) {
        urls.append(url)
    }
}

private struct ImageFixtureEmbedder: FixtureEmbedder {
    let modelID = "image-fixture"
    let dimension = 4
    let recorder: ImageEmbedRecorder
    var supportsImageEmbedding: Bool { true }

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        [0, 1, 0, 0]
    }

    func embedImage(at url: URL) async throws -> [Float] {
        await recorder.record(url)
        return [1, 0, 0, 0]
    }
}
