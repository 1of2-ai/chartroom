import ChartroomTestSupport
import Foundation
import Testing
@testable import IndexEngine

@Suite("Ingestion reliability")
struct IngestionReliabilityTests {
    private func temporaryStorePath(_ label: String) -> String {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("\(label)-\(UUID().uuidString).sqlite")
            .path
    }

    @Test("upsert rejects a short embedding batch without persisting the document")
    func upsertRejectsShortEmbeddingBatch() async throws {
        let store = try IndexStore(path: ":memory:", embedder: ShortBatchEmbedder())

        do {
            try await store.upsert(.init(
                id: "short-batch",
                type: "note",
                title: "Short",
                body: stablePrefixBody(tail: String(repeating: "short ", count: 300))
            ))
            Issue.record("Expected the short embedding batch to be rejected")
        } catch let error as IndexStoreError {
            guard case let .embeddingBatchCountMismatch(kind, expected, actual) = error else {
                Issue.record("Expected a batch-count mismatch, got \(error)")
                return
            }
            #expect(kind == .document)
            #expect(expected > 1)
            #expect(actual == expected - 1)
        }

        #expect(try await store.count() == 0)
    }

    @Test("rebuild rejects a short batch without partially writing it")
    func rebuildRejectsShortEmbeddingBatch() async throws {
        let path = temporaryStorePath("short-rebuild-batch")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let original = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "original", dimension: 4)
        )
        try await original.upsert(.init(
            id: "rebuild-short-batch",
            type: "note",
            title: "Rebuild",
            body: "the replacement embedder returns too few vectors"
        ))

        let swapped = try IndexStore(path: path, embedder: ShortBatchEmbedder())
        do {
            _ = try await swapped.rebuildActiveEmbeddingSpace(batchSize: 8)
            Issue.record("Expected the short embedding batch to be rejected")
        } catch let error as IndexStoreError {
            #expect(
                error == .embeddingBatchCountMismatch(
                    kind: .document,
                    expected: 1,
                    actual: 0
                )
            )
        }

        #expect(try await swapped.embeddingSpaceCoverage().activeSpaceEmbeddingCount == 0)
    }

    @Test("rebuild pages are bounded and cover every missing chunk")
    func rebuildPagesAreBoundedAndComplete() async throws {
        let path = temporaryStorePath("bounded-rebuild-pages")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let original = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "page-original", dimension: 4)
        )
        for index in 0..<7 {
            try await original.upsert(.init(
                id: String(format: "page-doc-%02d", index),
                type: "note",
                title: "Page \(index)",
                body: "bounded rebuild page \(index)"
            ))
        }

        let swapped = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "page-swapped", dimension: 4)
        )
        var cursor: IndexStore.RebuildCursor?
        var chunkIDs: [String] = []
        while true {
            let page = try await swapped.chunksMissingActiveEmbedding(after: cursor, limit: 2)
            #expect(page.count <= 2)
            guard let last = page.last else { break }
            chunkIDs.append(contentsOf: page.map(\.id))
            cursor = last.cursor
        }

        #expect(chunkIDs.count == 7)
        #expect(Set(chunkIDs).count == chunkIDs.count)
    }

    @Test("rebuild cursor uses chunk identity after equal document and ordinal keys")
    func rebuildCursorUsesChunkIDTieBreaker() async throws {
        let path = temporaryStorePath("rebuild-cursor-tie")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let original = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "tie-original", dimension: 4)
        )
        try await original.upsert(.init(
            id: "tie-document",
            type: "note",
            title: "Tie",
            body: [
                String(repeating: "alpha ", count: 300),
                String(repeating: "beta ", count: 300),
                String(repeating: "gamma ", count: 300),
            ].joined(separator: "\n\n")
        ))
        let db = try SQLite(path: path)
        try db.exec("UPDATE chunks SET ordinal = 0 WHERE document_id = 'tie-document'")
        let expectedCount = try await original.chunkSummaries(documentID: "tie-document").count
        #expect(expectedCount > 1)

        let swapped = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "tie-swapped", dimension: 4)
        )
        var cursor: IndexStore.RebuildCursor?
        var chunkIDs: [String] = []
        while true {
            let page = try await swapped.chunksMissingActiveEmbedding(after: cursor, limit: 1)
            guard let last = page.last else { break }
            chunkIDs.append(last.id)
            cursor = last.cursor
        }

        #expect(chunkIDs.count == expectedCount)
        #expect(Set(chunkIDs).count == expectedCount)
    }

    @Test("rebuild counts non-text chunks while paging past them")
    func rebuildCountsSkippedNonTextChunksAcrossPages() async throws {
        let path = temporaryStorePath("paged-nontext")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let original = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "modality-original", dimension: 4)
        )
        for index in 0..<3 {
            try await original.upsert(.init(
                id: "modality-\(index)",
                type: "note",
                title: "Modality \(index)",
                body: "modality fixture \(index)"
            ))
        }
        let db = try SQLite(path: path)
        try db.exec("""
        UPDATE embeddings
        SET modality = 'image'
        WHERE chunk_id IN (
          SELECT id FROM chunks WHERE document_id IN ('modality-0', 'modality-2')
        );
        UPDATE chunks
        SET embedding_modality = 'image'
        WHERE document_id IN ('modality-0', 'modality-2');
        """)

        let swapped = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "modality-swapped", dimension: 4)
        )
        let summary = try await swapped.rebuildActiveEmbeddingSpace(batchSize: 1)

        #expect(summary.rebuiltChunkCount == 1)
        #expect(summary.skippedNonTextChunkCount == 2)
        #expect(!summary.isComplete)
    }

    @Test("rebuild reports only non-text chunks that still need source re-ingestion")
    func rebuildRechecksSkippedNonTextChunksAtCompletion() async throws {
        let path = temporaryStorePath("rebuild-current-nontext")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let original = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "current-nontext-original", dimension: 4)
        )
        try await original.upsert(.init(
            id: "a-image",
            type: "note",
            title: "Image",
            body: "will be marked as image-derived"
        ))
        try await original.upsert(.init(
            id: "b-text",
            type: "note",
            title: "Text",
            body: "keeps the rebuild open for a concurrent deletion"
        ))
        let db = try SQLite(path: path)
        try db.exec("UPDATE chunks SET embedding_modality='image' WHERE document_id='a-image'")

        let gate = RebuildRaceGate()
        let rebuilding = try IndexStore(
            path: path,
            embedder: RebuildRaceEmbedder(gate: gate)
        )
        let deleter = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "current-nontext-deleter", dimension: 4)
        )
        let task = Task {
            try await rebuilding.rebuildActiveEmbeddingSpace(batchSize: 1)
        }

        await gate.waitForFirstBatch()
        #expect(try await deleter.delete(id: "a-image"))
        await gate.releaseFirstBatch()

        let summary = try await task.value
        #expect(summary.rebuiltChunkCount == 1)
        #expect(summary.skippedNonTextChunkCount == 0)
        #expect(summary.isComplete)
    }

    @Test("a paged rebuild resumes after a later batch fails")
    func rebuildResumesAfterPagedBatchFailure() async throws {
        let path = temporaryStorePath("paged-rebuild-resume")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let original = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "resume-original", dimension: 4)
        )
        for index in 0..<3 {
            try await original.upsert(.init(
                id: "resume-\(index)",
                type: "note",
                title: "Resume \(index)",
                body: "resume fixture \(index)"
            ))
        }

        let controller = FailOnceBatchController()
        let swapped = try IndexStore(
            path: path,
            embedder: FailOnceBatchEmbedder(controller: controller)
        )
        await #expect(throws: InjectedBatchFailure.self) {
            _ = try await swapped.rebuildActiveEmbeddingSpace(batchSize: 1)
        }
        #expect(try await swapped.embeddingSpaceCoverage().activeSpaceEmbeddingCount == 1)

        let resumed = try await swapped.rebuildActiveEmbeddingSpace(batchSize: 1)
        #expect(resumed.rebuiltChunkCount == 2)
        #expect(try await swapped.embeddingSpaceCoverage().activeSpaceEmbeddingCount == 3)
    }

    @Test("identical reingest reuses active-space embeddings and refreshes metadata")
    func identicalReingestReusesEmbeddingsAndRefreshesMetadata() async throws {
        let recorder = EmbeddingBatchRecorder()
        let store = try IndexStore(
            path: ":memory:",
            embedder: RecordingBatchEmbedder(recorder: recorder)
        )
        let body = "stable content whose vector should remain reusable"
        try await store.upsert(.init(
            id: "cached",
            type: "note",
            title: "Original",
            body: body,
            clusterID: "old-cluster",
            sourceURI: URL(filePath: "/tmp/original.md")
        ))
        #expect(await recorder.embeddedTextCount == 1)

        await recorder.reset()
        try await store.upsert(.init(
            id: "cached",
            type: "note",
            title: "Updated",
            body: body,
            clusterID: "new-cluster",
            sourceURI: URL(filePath: "/tmp/updated.md")
        ))

        #expect(await recorder.embeddedTextCount == 0)
        let hit = try #require(
            try await store.search("stable content", scope: .cluster("new-cluster", hard: true), limit: 1).first
        )
        #expect(hit.title == "Updated")
        #expect(hit.sourceURI == URL(filePath: "/tmp/updated.md"))
    }

    @Test("changed reingest embeds only chunk identities missing from the active space")
    func changedReingestEmbedsOnlyMissingChunks() async throws {
        let recorder = EmbeddingBatchRecorder()
        let store = try IndexStore(
            path: ":memory:",
            embedder: RecordingBatchEmbedder(recorder: recorder)
        )
        let originalBody = stablePrefixBody(tail: String(repeating: "gamma ", count: 300))
        try await store.upsert(.init(
            id: "partially-cached",
            type: "note",
            title: "Partial",
            body: originalBody
        ))
        let originalIDs = Set(
            try await store.chunkSummaries(documentID: "partially-cached").map(\.id.rawValue)
        )

        await recorder.reset()
        let changedBody = stablePrefixBody(tail: String(repeating: "delta ", count: 300))
        try await store.upsert(.init(
            id: "partially-cached",
            type: "note",
            title: "Partial",
            body: changedBody
        ))
        let changedIDs = Set(
            try await store.chunkSummaries(documentID: "partially-cached").map(\.id.rawValue)
        )
        let missingCount = changedIDs.subtracting(originalIDs).count

        #expect(missingCount > 0)
        #expect(missingCount < changedIDs.count)
        #expect(await recorder.embeddedTextCount == missingCount)
    }

    @Test("a policy change invalidates chunk-level embedding reuse")
    func policyChangeDoesNotReuseChunkEmbeddings() async throws {
        let recorder = EmbeddingBatchRecorder()
        let store = try IndexStore(
            path: ":memory:",
            embedder: RecordingBatchEmbedder(recorder: recorder)
        )
        let body = stablePrefixBody(tail: String(repeating: "policy ", count: 300))
        try await store.upsert(.init(
            id: "policy-scoped",
            type: "note",
            title: "Policy",
            body: body,
            policyID: "policy-a"
        ))
        let originalIDs = Set(
            try await store.chunkSummaries(documentID: "policy-scoped").map(\.id.rawValue)
        )

        await recorder.reset()
        try await store.upsert(.init(
            id: "policy-scoped",
            type: "note",
            title: "Policy",
            body: body,
            policyID: "policy-b"
        ))
        let changedIDs = Set(
            try await store.chunkSummaries(documentID: "policy-scoped").map(\.id.rawValue)
        )

        #expect(originalIDs.isDisjoint(with: changedIDs))
        #expect(await recorder.embeddedTextCount == changedIDs.count)
    }

    @Test("cached chunks do not bypass cross-store delete ordering")
    func cachedChunksDoNotBypassDeleteMutationToken() async throws {
        let path = temporaryStorePath("cached-delete-race")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let originalBody = stablePrefixBody(tail: String(repeating: "gamma ", count: 300))
        let seed = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "cache-race", dimension: 4)
        )
        try await seed.upsert(.init(
            id: "cached-race",
            type: "note",
            title: "Race",
            body: originalBody
        ))

        let gate = CacheRaceGate()
        let writer = try IndexStore(
            path: path,
            embedder: CacheRaceEmbedder(gate: gate)
        )
        let deleter = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "cache-race", dimension: 4)
        )
        let changedBody = stablePrefixBody(tail: String(repeating: "delta ", count: 300))
        let upsert = Task {
            try await writer.upsert(.init(
                id: "cached-race",
                type: "note",
                title: "Race",
                body: changedBody
            ))
        }

        await gate.waitForEmbedding()
        #expect(try await deleter.delete(id: "cached-race"))
        await gate.releaseEmbedding()
        try await upsert.value

        #expect(try await writer.chunkSummaries(documentID: "cached-race").isEmpty)
        #expect(try await writer.search("delta", limit: 5).isEmpty)
    }

    @Test("rebuild follows current chunk modality across mixed historical spaces")
    func rebuildUsesCurrentChunkModality() async throws {
        let path = temporaryStorePath("current-chunk-modality")
        let imageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("modality-\(UUID().uuidString).png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: imageURL)
        defer {
            try? FileManager.default.removeItem(atPath: path)
            try? FileManager.default.removeItem(at: imageURL)
        }
        let body = "one stable representation shared across modality changes"

        let imageSpace = try IndexStore(
            path: path,
            embedder: ModalityFixtureEmbedder(modelID: "modality-image-a", supportsImages: true)
        )
        try await imageSpace.upsert(.init(
            id: "mixed-modality",
            type: "image/png",
            title: "Mixed",
            body: body,
            sourceURI: imageURL
        ))

        let textSpace = try IndexStore(
            path: path,
            embedder: ModalityFixtureEmbedder(modelID: "modality-text-b", supportsImages: false)
        )
        try await textSpace.upsert(.init(
            id: "mixed-modality",
            type: "note",
            title: "Mixed",
            body: body
        ))

        let textRebuild = try IndexStore(
            path: path,
            embedder: ModalityFixtureEmbedder(modelID: "modality-text-c", supportsImages: false)
        )
        let rebuilt = try await textRebuild.rebuildActiveEmbeddingSpace(batchSize: 1)
        #expect(rebuilt.rebuiltChunkCount == 1)
        #expect(rebuilt.skippedNonTextChunkCount == 0)

        let currentImageSpace = try IndexStore(
            path: path,
            embedder: ModalityFixtureEmbedder(modelID: "modality-image-d", supportsImages: true)
        )
        try await currentImageSpace.upsert(.init(
            id: "mixed-modality",
            type: "image/png",
            title: "Mixed",
            body: body,
            sourceURI: imageURL
        ))

        let imageRebuild = try IndexStore(
            path: path,
            embedder: ModalityFixtureEmbedder(modelID: "modality-text-e", supportsImages: false)
        )
        let skipped = try await imageRebuild.rebuildActiveEmbeddingSpace(batchSize: 1)
        #expect(skipped.rebuiltChunkCount == 0)
        #expect(skipped.skippedNonTextChunkCount == 1)
    }

    @Test("rebuild discards a stale page and picks up the replacement chunk")
    func rebuildRechecksChunksAfterEmbedding() async throws {
        let path = temporaryStorePath("rebuild-replacement-race")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let original = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "rebuild-old", dimension: 4)
        )
        try await original.upsert(.init(
            id: "rebuild-race",
            type: "note",
            title: "Race",
            body: "old rebuild content"
        ))

        let gate = RebuildRaceGate()
        let rebuilding = try IndexStore(
            path: path,
            embedder: RebuildRaceEmbedder(gate: gate)
        )
        let replacement = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "replacement-space", dimension: 4)
        )
        let task = Task {
            try await rebuilding.rebuildActiveEmbeddingSpace(batchSize: 1)
        }

        await gate.waitForFirstBatch()
        try await replacement.upsert(.init(
            id: "rebuild-race",
            type: "note",
            title: "Race",
            body: "new replacement content"
        ))
        await gate.releaseFirstBatch()
        let summary = try await task.value

        #expect(summary.rebuiltChunkCount == 1)
        #expect(try await rebuilding.embeddingSpaceCoverage().activeSpaceEmbeddingCount == 1)
        #expect(
            try await rebuilding.chunkSummaries(documentID: "rebuild-race").map(\.text)
                == ["new replacement content"]
        )
    }

    private func stablePrefixBody(tail: String) -> String {
        [
            String(repeating: "alpha ", count: 300),
            String(repeating: "beta ", count: 300),
            tail,
        ].joined(separator: "\n\n")
    }
}

private struct ShortBatchEmbedder: FixtureEmbedder {
    let modelID = "short-batch"
    let dimension = 4

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        [1, 0, 0, 0]
    }

    func embed(_ texts: [String], kind: EmbedKind) async throws -> [[Float]] {
        Array(repeating: [1, 0, 0, 0], count: max(0, texts.count - 1))
    }
}

private enum InjectedBatchFailure: Error {
    case once
}

private actor FailOnceBatchController {
    private var callCount = 0
    private var hasFailed = false

    func vectors(for texts: [String]) throws -> [[Float]] {
        callCount += 1
        if callCount == 2, !hasFailed {
            hasFailed = true
            throw InjectedBatchFailure.once
        }
        return Array(repeating: [1, 0, 0, 0], count: texts.count)
    }
}

private struct FailOnceBatchEmbedder: FixtureEmbedder {
    let modelID = "resume-swapped"
    let dimension = 4
    let controller: FailOnceBatchController

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        [1, 0, 0, 0]
    }

    func embed(_ texts: [String], kind: EmbedKind) async throws -> [[Float]] {
        try await controller.vectors(for: texts)
    }
}

private actor EmbeddingBatchRecorder {
    private(set) var embeddedTextCount = 0

    func record(_ texts: [String]) {
        embeddedTextCount += texts.count
    }

    func reset() {
        embeddedTextCount = 0
    }
}

private struct RecordingBatchEmbedder: FixtureEmbedder {
    let modelID = "recording-cache"
    let dimension = 4
    let recorder: EmbeddingBatchRecorder

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        [1, 0, 0, 0]
    }

    func embed(_ texts: [String], kind: EmbedKind) async throws -> [[Float]] {
        await recorder.record(texts)
        return Array(repeating: [1, 0, 0, 0], count: texts.count)
    }
}

private actor CacheRaceGate {
    private var hasStarted = false
    private var startContinuation: CheckedContinuation<Void, Never>?
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    func suspendEmbedding() async {
        hasStarted = true
        startContinuation?.resume()
        startContinuation = nil
        await withCheckedContinuation { continuation in
            releaseContinuation = continuation
        }
    }

    func waitForEmbedding() async {
        if hasStarted { return }
        await withCheckedContinuation { continuation in
            startContinuation = continuation
        }
    }

    func releaseEmbedding() {
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}

private struct CacheRaceEmbedder: FixtureEmbedder {
    let modelID = "cache-race"
    let dimension = 4
    let gate: CacheRaceGate

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        [1, 0, 0, 0]
    }

    func embed(_ texts: [String], kind: EmbedKind) async throws -> [[Float]] {
        await gate.suspendEmbedding()
        return Array(repeating: [1, 0, 0, 0], count: texts.count)
    }
}

private struct ModalityFixtureEmbedder: FixtureEmbedder {
    let modelID: String
    let dimension = 4
    let supportsImages: Bool

    var supportsImageEmbedding: Bool { supportsImages }

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        [1, 0, 0, 0]
    }

    func embedImage(at url: URL) async throws -> [Float] {
        [0, 1, 0, 0]
    }
}

private actor RebuildRaceGate {
    private var firstBatchStarted = false
    private var startContinuation: CheckedContinuation<Void, Never>?
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    func suspendFirstBatch() async {
        guard !firstBatchStarted else { return }
        firstBatchStarted = true
        startContinuation?.resume()
        startContinuation = nil
        await withCheckedContinuation { continuation in
            releaseContinuation = continuation
        }
    }

    func waitForFirstBatch() async {
        if firstBatchStarted { return }
        await withCheckedContinuation { continuation in
            startContinuation = continuation
        }
    }

    func releaseFirstBatch() {
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}

private struct RebuildRaceEmbedder: FixtureEmbedder {
    let modelID = "rebuild-active"
    let dimension = 4
    let gate: RebuildRaceGate

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        [1, 0, 0, 0]
    }

    func embed(_ texts: [String], kind: EmbedKind) async throws -> [[Float]] {
        await gate.suspendFirstBatch()
        return Array(repeating: [1, 0, 0, 0], count: texts.count)
    }
}
