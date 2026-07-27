import ChartroomTestSupport
import Foundation
import Testing
@testable import IndexEngine

@Suite("IndexEngine facade")
struct IndexEngineFacadeTests {
    @Test("apps can ingest and search through the public facade")
    func ingestAndSearchThroughFacade() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 256))
        )
        let payload = SourcePayload(
            documentID: "doc-1",
            sourceID: "local-files",
            sourceURI: URL(filePath: "/tmp/retrieval-plan.md"),
            displayName: "Retrieval Plan",
            contentType: "net.daringfireball.markdown",
            body: .text("semantic retrieval with policy registry and vector search")
        )

        let summary = try await engine.ingest(.init(payloads: [payload]))
        #expect(summary.acceptedCount == 1)
        #expect(summary.failedCount == 0)

        let response = try await engine.search(.init(query: "policy registry", limit: 5))
        let firstResult = try #require(response.results.first)
        #expect(firstResult.documentID == "doc-1")
        #expect(firstResult.chunkID != "doc-1")
        #expect(firstResult.snippet?.contains("policy registry") == true)
        #expect(firstResult.rank == 1)
        #expect(firstResult.sourceID == "local-files")
        #expect(firstResult.sourceURI == URL(filePath: "/tmp/retrieval-plan.md"))
        #expect(firstResult.provenance.connectorID == "local-files")
        #expect(firstResult.provenance.policyID == "default")
        #expect(firstResult.provenance.representationID?.rawValue.hasPrefix("doc-1:representation:plainText:") == true)
        #expect(firstResult.provenance.embeddingSpaceID == "hashing-mock-v1:256")

        let snapshot = await engine.snapshot()
        #expect(snapshot.objectCount == 1)
        #expect(snapshot.documentCount == 1)
        #expect(snapshot.chunkCount == 1)
        #expect(snapshot.embeddingCount == 1)
        #expect(snapshot.embeddingDimension == 256)
        #expect(snapshot.modelID == "hashing-mock-v1")
        #expect(snapshot.embeddingSpaceID == "hashing-mock-v1:256")

        let health = await engine.health()
        #expect(health.objectCount == 1)
        #expect(health.documentCount == 1)
        #expect(health.chunkCount == 1)
        #expect(health.embeddingCount == 1)
        #expect(health.vectorBackendStatus?.backendID == .builtInSQLiteVectorBackend)
        #expect(health.vectorBackendStatus?.state == .ready)

        let modelStatus = await engine.modelStatus()
        #expect(modelStatus.modelID == "hashing-mock-v1")
        #expect(modelStatus.embeddingSpaceID == "hashing-mock-v1:256")
        #expect(modelStatus.dimension == 256)
        #expect(modelStatus.isAvailable)
    }

    @Test("model status reports observed availability, not an assumption")
    func modelStatusReflectsActualEmbedderHealth() async throws {
        // A provider that cannot produce a vector must surface as unavailable with its own reason,
        // never as a hardcoded "available". This is the GUI's only signal that search is broken.
        let failing = try await IndexEngine.openInMemory(
            configuration: .init(embedder: ThrowingEmbedder())
        )
        let failingStatus = await failing.modelStatus()
        #expect(failingStatus.isAvailable == false)
        #expect(failingStatus.message.contains("failed"))

        // A provider whose query output does not match the index's embedding space is equally
        // unusable: the wrong dimension silently breaks cosine search, so status must report it
        // as unavailable rather than claim a model that cannot answer a query.
        let mismatched = try await IndexEngine.openInMemory(
            configuration: .init(embedder: WrongQueryDimensionEmbedder())
        )
        let mismatchedStatus = await mismatched.modelStatus()
        #expect(mismatchedStatus.isAvailable == false)
        #expect(mismatchedStatus.message.contains("8"))
    }

    @Test("cluster filters are explicit and not inferred from policy IDs")
    func clusterFilteringRequiresExplicitCluster() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 256))
        )
        let payload = SourcePayload(
            documentID: "doc-1",
            displayName: "Policy Note",
            body: .text("policy registry vector search")
        )

        _ = try await engine.ingest(.init(payloads: [payload], policy: .init(id: "policy-a", version: 1)))

        let global = try await engine.search(.init(query: "policy registry", limit: 5))
        #expect(global.results.map(\.documentID).contains("doc-1"))

        let scoped = try await engine.search(
            .init(query: "policy registry", limit: 5, filters: .init(clusterID: "policy-a"))
        )
        #expect(scoped.results.isEmpty)
    }

    @Test("binary text references are extracted by the engine before indexing")
    func binaryTextReferencesAreSearchableByContent() async throws {
        let fileURL = FileManager.default.temporaryDirectory
            .appending(path: "engine-reference-\(UUID().uuidString).md")
        try "local folder ingest should search this needle phrase".write(
            to: fileURL,
            atomically: true,
            encoding: .utf8
        )
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 256))
        )
        let payload = SourcePayload(
            documentID: "binary-text-doc",
            sourceID: "local-files",
            sourceURI: fileURL,
            displayName: fileURL.lastPathComponent,
            contentType: "net.daringfireball.markdown",
            body: .binaryReference(fileURL)
        )

        let summary = try await engine.ingest(.init(payloads: [payload]))
        #expect(summary.acceptedCount == 1)

        let response = try await engine.search(.init(query: "needle phrase", limit: 5))
        #expect(response.results.first?.documentID == "binary-text-doc")
        #expect(response.results.first?.provenance.representationID?.rawValue.hasPrefix("binary-text-doc:representation:markdown:") == true)
    }

    @Test("facade search filters are honored before ranking")
    func facadeHonorsSearchFilters() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 256))
        )

        let local = SourcePayload(
            documentID: "doc-local",
            sourceID: "local-source",
            sourceURI: URL(filePath: "/tmp/local.md"),
            displayName: "Local Note",
            contentType: "net.daringfireball.markdown",
            body: .text("shared retrieval policy boundary")
        )
        let api = SourcePayload(
            documentID: "doc-api",
            sourceID: "api-source",
            sourceURI: try #require(URLComponents(string: "https://example.invalid/api/doc")?.url),
            displayName: "API Note",
            contentType: "application/json",
            body: .text("shared retrieval policy boundary")
        )

        _ = try await engine.ingest(.init(payloads: [local], policy: .init(id: "policy-a", version: 1)))
        _ = try await engine.ingest(.init(payloads: [api], policy: .init(id: "policy-b", version: 1)))

        let sourceFiltered = try await engine.search(
            .init(query: "shared retrieval", filters: .init(sourceIDs: ["api-source"]))
        )
        #expect(sourceFiltered.results.map(\.documentID) == ["doc-api"])

        let typeFiltered = try await engine.search(
            .init(query: "shared retrieval", filters: .init(contentTypes: ["net.daringfireball.markdown"]))
        )
        #expect(typeFiltered.results.map(\.documentID) == ["doc-local"])

        let policyFiltered = try await engine.search(
            .init(query: "shared retrieval", filters: .init(policyID: "policy-b"))
        )
        #expect(policyFiltered.results.map(\.documentID) == ["doc-api"])

        let spaceFiltered = try await engine.search(
            .init(query: "shared retrieval", filters: .init(embeddingSpaceID: "hashing-mock-v1:256"))
        )
        #expect(Set(spaceFiltered.results.map(\.documentID)) == ["doc-local", "doc-api"])

        let wrongSpace = try await engine.search(
            .init(query: "shared retrieval", filters: .init(embeddingSpaceID: "other-model:256"))
        )
        #expect(wrongSpace.results.isEmpty)
    }

    @Test("degraded search keeps FTS hits when the query vector channel is unavailable")
    func degradedSearchKeepsFTSHits() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: QueryFailingEmbedder())
        )
        _ = try await engine.ingest(.init(payloads: [
            SourcePayload(documentID: "doc-degraded", displayName: "Degraded", body: .text("thermal fallback search path"))
        ]))

        let degraded = try await engine.search(.init(query: "thermal fallback", limit: 5, allowDegradedResults: true))
        #expect(degraded.results.map(\.documentID) == ["doc-degraded"])
        #expect(degraded.diagnostics.degraded)
        #expect(degraded.diagnostics.missingChannels == [.vector])

        do {
            _ = try await engine.search(.init(query: "thermal fallback", limit: 5, allowDegradedResults: false))
            Issue.record("Expected non-degraded search to throw when query embedding fails")
        } catch let error as IndexEngineError {
            #expect(error.category == .embeddingProviderUnavailable)
            #expect(error.code == "index.search.embedding-unavailable")
        }
    }

    @Test("custom retrieval profile caps are applied by facade search")
    func retrievalProfileCapsSearchCandidates() async throws {
        let payload = SourcePayload(
            documentID: "doc-profile",
            displayName: "Profile Fixture",
            body: .text("body-only candidate marker")
        )

        let defaultEngine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 64))
        )
        _ = try await defaultEngine.ingest(.init(payloads: [payload]))
        let defaultResponse = try await defaultEngine.search(.init(query: "candidate marker", limit: 5))
        #expect(defaultResponse.results.map(\.documentID) == ["doc-profile"])

        let cappedProfile = RetrievalProfile(
            id: "no-candidates",
            version: 1,
            maxFTSCandidates: 0,
            maxVectorCandidates: 0,
            maxSnippets: 0
        )
        let cappedEngine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 64), retrievalProfile: cappedProfile)
        )
        _ = try await cappedEngine.ingest(.init(payloads: [payload]))
        let cappedResponse = try await cappedEngine.search(.init(query: "candidate marker", limit: 5))
        #expect(cappedResponse.results.isEmpty)
    }

    @Test("facade deletion removes documents and records jobs")
    func deleteThroughFacade() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 256))
        )
        let payload = SourcePayload(
            documentID: "doc-delete",
            sourceID: "local-source",
            displayName: "Delete Me",
            body: .text("temporary deletion target")
        )

        _ = try await engine.ingest(.init(payloads: [payload], jobID: "ingest-delete-fixture"))

        let summary = try await engine.delete(.init(documentIDs: ["doc-delete"], jobID: "delete-job"))
        #expect(summary.requestedCount == 1)
        #expect(summary.deletedCount == 1)
        #expect(summary.failedCount == 0)

        let response = try await engine.search(.init(query: "temporary deletion", limit: 5))
        #expect(response.results.isEmpty)

        let jobs = await engine.jobs(limit: 4)
        #expect(jobs.contains { $0.id == "delete-job" && $0.state == .succeeded })
    }

    @Test("deleting a nonexistent document reports a no-op instead of a successful delete")
    func deleteNonexistentDocumentIsNoOp() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 256))
        )

        let summary = try await engine.delete(.init(documentIDs: ["missing"], jobID: "delete-missing"))
        #expect(summary.requestedCount == 1)
        #expect(summary.deletedCount == 0)
        #expect(summary.failedCount == 0)

        let job = try #require(await engine.jobs(limit: 1).first)
        #expect(job.id == "delete-missing")
        #expect(job.completedUnitCount == 0)
        #expect(job.state == .succeeded)
    }

    @Test("persistent open storage failures use the typed facade error")
    func openStorageFailureIsTyped() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("index-open-directory-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        do {
            _ = try await IndexEngine.open(storeURL: directory)
            Issue.record("Expected opening a directory as a SQLite file to fail")
        } catch let error as IndexEngineError {
            #expect(error.category == .storageUnavailable)
            #expect(error.code == "index.open.storage-unavailable")
        }
    }

    @Test("persistent open marks interrupted jobs as failed")
    func openRecoversInterruptedJobs() async throws {
        let storeURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("interrupted-jobs-\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: storeURL) }

        do {
            let store = try IndexStore(path: storeURL.path, embedder: HashingEmbedder(dimension: 64))
            try await store.recordJob(.init(
                id: "interrupted",
                state: .running,
                completedUnitCount: 2,
                totalUnitCount: 5,
                message: "Ingesting source payloads"
            ))
        }

        let engine = try await IndexEngine.open(
            storeURL: storeURL,
            configuration: .init(embedder: HashingEmbedder(dimension: 64))
        )
        let job = try #require(await engine.jobs(limit: 1).first)
        #expect(job.id == "interrupted")
        #expect(job.state == .failed)
        #expect(job.completedUnitCount == 2)
        #expect(job.totalUnitCount == 5)
        #expect(job.message.contains("Interrupted before the index reopened"))
    }

    @Test("snapshot reports the on-disk store footprint; in-memory reports none")
    func snapshotReportsStoreFootprint() async throws {
        // An in-memory store has no on-disk files, so it reports no footprint.
        let memory = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 64))
        )
        #expect(await memory.snapshot().storeByteSize == nil)

        // A persistent store reports a positive byte size the harness can display
        // without knowing the SQLite/WAL file layout itself.
        let storeURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("store-footprint-\(UUID().uuidString).sqlite")
        defer {
            for suffix in ["", "-wal", "-shm"] {
                try? FileManager.default.removeItem(atPath: storeURL.path + suffix)
            }
        }
        let engine = try await IndexEngine.open(
            storeURL: storeURL,
            configuration: .init(embedder: HashingEmbedder(dimension: 64))
        )
        let size = try #require(await engine.snapshot().storeByteSize)
        #expect(size > 0)
    }

    @Test("the engine reports its retrieval pipeline with the live fusion constant")
    func engineReportsRetrievalPipeline() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 64))
        )
        let pipeline = engine.retrievalPipeline
        #expect(pipeline.filterStage == "SQL pre-filter")
        #expect(pipeline.candidateChannels.contains("FTS5 BM25"))
        #expect(pipeline.candidateChannels.contains("Vector cosine"))
        // The fusion string carries the same k the retrieval code actually uses, so the
        // description can't drift from IndexStoreRetrieval's implementation.
        #expect(pipeline.fusion.contains("Reciprocal Rank Fusion"))
        #expect(pipeline.fusion.contains("\(Int(IndexStore.reciprocalRankK))"))
    }

    @Test("failed ingestion is exposed through typed diagnostics")
    func failedIngestionRecordsDiagnostics() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: ThrowingEmbedder())
        )
        let payload = SourcePayload(
            documentID: "doc-fail",
            sourceID: "fixture-source",
            displayName: "Failing Payload",
            body: .text("this will fail at embedding time")
        )

        do {
            _ = try await engine.ingest(.init(payloads: [payload], jobID: "failed-ingest-job"))
            Issue.record("Expected ingestion to fail")
        } catch let error as IndexEngineError {
            #expect(error.category == .ingestionFailed)
            #expect(error.code == "index.ingest.failed")
            #expect(error.recoverability == .needsConfiguration)
        }

        let failures = await engine.failures(limit: 10)
        #expect(failures.first?.documentID == "doc-fail")
        #expect(failures.first?.sourceID == "fixture-source")
        #expect(failures.first?.category == .embeddingFailure)
        #expect(failures.first?.recoverability == .needsConfiguration)

        let jobs = await engine.jobs(limit: 4)
        #expect(jobs.first?.id == "failed-ingest-job")
        #expect(jobs.first?.state == .failed)
    }

    @Test("clearing failures prunes durable diagnostics, selectively and in full")
    func clearingFailuresPrunesDiagnostics() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: ThrowingEmbedder())
        )

        for index in 0..<3 {
            _ = try? await engine.ingest(.init(
                payloads: [SourcePayload(
                    documentID: DocumentID(rawValue: "doc-\(index)"),
                    sourceID: "fixture-source",
                    displayName: "Failing \(index)",
                    body: .text("fails at embedding time")
                )],
                jobID: JobID(rawValue: "job-\(index)")
            ))
        }

        let recorded = await engine.failures(limit: 10)
        #expect(recorded.count == 3)

        // Selective clear removes only the named failure.
        let victim = try #require(recorded.first { $0.documentID == "doc-1" })
        try await engine.clearFailures(ids: [victim.id])
        let afterSelective = await engine.failures(limit: 10)
        #expect(afterSelective.count == 2)
        #expect(!afterSelective.contains { $0.id == victim.id })
        #expect(afterSelective.contains { $0.documentID == "doc-0" })

        // Clearing all empties the diagnostics.
        try await engine.clearFailures(ids: nil)
        #expect(await engine.failures(limit: 10).isEmpty)
    }

    @Test("cancelled ingestion records a cancelled job without durable item failures")
    func cancelledIngestionIsNotRecordedAsEmbeddingFailures() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: SlowCancellableEmbedder())
        )
        let task = Task {
            try await engine.ingest(.init(
                payloads: [SourcePayload(documentID: "doc-cancel", displayName: "Cancel", body: .text("cancel me"))],
                jobID: "cancel-job"
            ))
        }

        try await Task.sleep(for: .milliseconds(20))
        task.cancel()

        do {
            _ = try await task.value
            Issue.record("Expected cancellation to throw through the typed facade")
        } catch let error as IndexEngineError {
            #expect(error.category == .cancelled)
            #expect(error.code == "index.ingest.cancelled")
        }

        #expect(await engine.failures(limit: 10).isEmpty)
        let job = try #require(await engine.jobs(limit: 1).first)
        #expect(job.id == "cancel-job")
        #expect(job.state == .cancelled)
    }

    @Test("ingest embedding-space failures keep typed recoverability in throws and durable rows")
    func ingestEmbeddingSpaceFailureRecoverability() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: WrongDocumentDimensionEmbedder())
        )
        let payload = SourcePayload(
            documentID: "doc-mismatch",
            sourceID: "fixture-source",
            displayName: "Wrong Vector",
            body: .text("this vector cannot fit the configured embedding space")
        )

        do {
            _ = try await engine.ingest(.init(payloads: [payload], jobID: "mismatched-ingest-job"))
            Issue.record("Expected embedding-space ingestion to fail")
        } catch let error as IndexEngineError {
            #expect(error.category == .ingestionFailed)
            #expect(error.recoverability == .needsConfiguration)
        }

        let failure = try #require(await engine.failures(limit: 1).first)
        #expect(failure.documentID == "doc-mismatch")
        #expect(failure.category == .embeddingFailure)
        #expect(failure.recoverability == .needsConfiguration)
        #expect(failure.isRecoverable)
    }

    @Test("one bad payload fails alone and does not poison the rest of the batch")
    func perItemFailureDoesNotPoisonTheBatch() async throws {
        // The spec is explicit: per-item failures must not fail a whole batch unless policy requires
        // fail-fast. A single poisoned payload must be recorded as a failure while its healthy
        // siblings are still ingested and become searchable.
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: SelectivelyFailingEmbedder())
        )

        let summary = try await engine.ingest(.init(payloads: [
            SourcePayload(documentID: "ok-1", displayName: "Good One", body: .text("retrieval policy registry")),
            SourcePayload(documentID: "bad", displayName: "Poisoned", body: .text("poison pill payload")),
            SourcePayload(documentID: "ok-2", displayName: "Good Two", body: .text("vector search fusion")),
        ]))

        #expect(summary.acceptedCount == 2)
        #expect(summary.failedCount == 1)
        #expect(summary.failures.map(\.documentID) == ["bad"])

        // The healthy siblings are durable and searchable — the batch was not aborted by the failure.
        let response = try await engine.search(.init(query: "policy registry", limit: 5))
        #expect(response.results.contains { $0.documentID == "ok-1" })

        let snapshot = await engine.snapshot()
        #expect(snapshot.documentCount == 2)
    }

    @Test("documents can be browsed as a typed projection, newest first, excluding tombstones")
    func documentsAreBrowsableAsTypedProjection() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 64))
        )
        _ = try await engine.ingest(.init(payloads: [
            SourcePayload(documentID: "doc-old", sourceID: "files", displayName: "Older Note",
                          contentType: "net.daringfireball.markdown", body: .text("alpha beta gamma")),
            SourcePayload(documentID: "doc-new", sourceID: "files", displayName: "Newer Note",
                          contentType: "net.daringfireball.markdown", body: .text("delta epsilon")),
        ]))

        let response = try await engine.browseDocuments(.init(limit: 50))
        #expect(response.totalMatching == 2)
        #expect(response.returnedCount == 2)
        let documents = response.documents

        // The projection carries identity and counts the GUI can list without touching SQLite.
        let newer = try #require(documents.first { $0.id == "doc-new" })
        #expect(newer.title == "Newer Note")
        #expect(newer.sourceID == "files")
        #expect(newer.contentType == "net.daringfireball.markdown")
        #expect(newer.policyID == "default")
        #expect(newer.chunkCount >= 1)

        // A tombstoned document drops out of the browse projection, matching what search can see.
        _ = try await engine.delete(.init(documentIDs: ["doc-old"]))
        let remaining = try await engine.browseDocuments(.init(limit: 50))
        #expect(remaining.documents.map(\.id) == ["doc-new"])

        // The limit is honored and a non-positive limit returns nothing.
        #expect(await engine.documents(limit: 0).isEmpty)
    }

    @Test("document browsing honors query, filters, sort, and paging")
    func documentBrowsingHonorsQueryFiltersSortAndPaging() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 64))
        )
        let apiPolicy = IngestionPolicy(id: "policy-api", version: 1)

        _ = try await engine.ingest(.init(payloads: [
            SourcePayload(
                documentID: "doc-local",
                sourceID: "files",
                sourceURI: URL(filePath: "/tmp/local-note.md"),
                displayName: "Local Note",
                contentType: "net.daringfireball.markdown",
                body: .text("local search fixture"),
                clusterID: "project-a"
            )
        ]))
        _ = try await engine.ingest(.init(payloads: [
            SourcePayload(
                documentID: "doc-api",
                sourceID: "api",
                sourceURI: try #require(URLComponents(string: "https://example.invalid/api/doc")?.url),
                displayName: "API Contract",
                contentType: "application/json",
                body: .text("typed connector fixture"),
                clusterID: "project-b"
            )
        ], policy: apiPolicy))
        _ = try await engine.ingest(.init(payloads: [
            SourcePayload(
                documentID: "doc-manual",
                sourceID: "files",
                sourceURI: URL(filePath: "/tmp/manual.txt"),
                displayName: "Manual",
                contentType: "public.plain-text",
                body: .text(String(repeating: "large document body ", count: 80)),
                clusterID: "project-a"
            )
        ]))

        let titleSorted = try await engine.browseDocuments(.init(sort: .titleAscending, limit: 10))
        #expect(titleSorted.documents.map(\.title) == ["API Contract", "Local Note", "Manual"])

        let truncated = try await engine.browseDocuments(.init(sort: .titleAscending, limit: 2))
        #expect(truncated.totalMatching == 3)
        #expect(truncated.returnedCount == 2)
        #expect(truncated.isTruncated)
        #expect(truncated.hasPreviousPage == false)
        #expect(truncated.hasNextPage)
        #expect(truncated.startIndex == 1)
        #expect(truncated.endIndex == 2)
        #expect(truncated.facets.sourceIDs == ["api", "files"])
        #expect(truncated.facets.contentTypes == ["application/json", "net.daringfireball.markdown", "public.plain-text"])

        let secondPage = try await engine.browseDocuments(.init(sort: .titleAscending, limit: 2, offset: 2))
        #expect(secondPage.documents.map(\.title) == ["Manual"])
        #expect(secondPage.totalMatching == 3)
        #expect(secondPage.hasPreviousPage)
        #expect(secondPage.hasNextPage == false)
        #expect(secondPage.startIndex == 3)
        #expect(secondPage.endIndex == 3)

        let facetOnly = try await engine.browseDocuments(.init(limit: 0))
        #expect(facetOnly.documents.isEmpty)
        #expect(facetOnly.totalMatching == 3)
        #expect(facetOnly.facets.sourceIDs == ["api", "files"])
        #expect(facetOnly.facets.contentTypes == ["application/json", "net.daringfireball.markdown", "public.plain-text"])

        let queryFiltered = try await engine.browseDocuments(.init(query: "api", sort: .titleAscending, limit: 10))
        #expect(queryFiltered.documents.map(\.id) == ["doc-api"])
        #expect(queryFiltered.facets.sourceIDs == ["api"])
        #expect(queryFiltered.facets.contentTypes == ["application/json"])

        let sourceFiltered = try await engine.browseDocuments(
            .init(filters: .init(sourceIDs: ["api"]), sort: .titleAscending, limit: 10)
        )
        #expect(sourceFiltered.documents.map(\.id) == ["doc-api"])

        let typeFiltered = try await engine.browseDocuments(
            .init(filters: .init(contentTypes: ["net.daringfireball.markdown"]), limit: 10)
        )
        #expect(typeFiltered.documents.map(\.id) == ["doc-local"])

        let policyFiltered = try await engine.browseDocuments(
            .init(filters: .init(policyID: "policy-api"), limit: 10)
        )
        let apiDocument = try #require(policyFiltered.documents.first)
        #expect(apiDocument.id == "doc-api")
        #expect(apiDocument.policyID == "policy-api")

        let clusterFiltered = try await engine.browseDocuments(
            .init(filters: .init(clusterID: "project-a"), sort: .titleAscending, limit: 10)
        )
        #expect(clusterFiltered.documents.map(\.id) == ["doc-local", "doc-manual"])

        let spaceFiltered = try await engine.browseDocuments(
            .init(filters: .init(embeddingSpaceID: "hashing-mock-v1:64"), limit: 10)
        )
        #expect(spaceFiltered.totalMatching == 3)
    }

    @Test("an unsatisfied policy is quarantined, visible in health, and refuses to ingest silently")
    func unsatisfiedPolicyIsQuarantinedAndVisible() async throws {
        // A policy naming a vector backend that this build never registered cannot run. The engine
        // must surface that as a quarantined resolution in health — never a silent `.satisfied` — and
        // must refuse to ingest under it with a typed configuration error rather than dropping data.
        let missingBackend: VectorBackendID = "indexengine.vector.sqlite-vec.absent"
        let policy = IngestionPolicy(id: "needs-sqlite-vec", version: 1, vectorBackendID: missingBackend)
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 64), defaultPolicy: policy)
        )

        let resolution = try #require(await engine.health().policyStates.first { $0.policyID == "needs-sqlite-vec" })
        #expect(resolution.state == .quarantined)
        #expect(resolution.missingComponents.contains(missingBackend))

        do {
            _ = try await engine.ingest(.init(
                payloads: [SourcePayload(documentID: "doc-1", displayName: "Doc", body: .text("body text"))],
                policy: policy
            ))
            Issue.record("Expected ingestion under a quarantined policy to throw, not silently drop the payload")
        } catch let error as IndexEngineError {
            #expect(error.category == .policyQuarantined)
            #expect(error.recoverability == .needsConfiguration)
            #expect(error.relatedIDs.contains("needs-sqlite-vec"))
        }
    }

    @Test("a query embedding-space mismatch surfaces as embeddingSpaceUnavailable, not a provider fault")
    func searchSpaceMismatchIsTypedDistinctly() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: WrongQueryDimensionEmbedder())
        )
        _ = try await engine.ingest(.init(payloads: [
            SourcePayload(documentID: "doc-a", displayName: "Thermal", body: .text("thermal capture"))
        ]))

        do {
            _ = try await engine.search(.init(query: "thermal", limit: 5, allowDegradedResults: false))
            Issue.record("Expected a typed search error")
        } catch let error as IndexEngineError {
            #expect(error.category == .embeddingSpaceUnavailable)
            #expect(error.code == "index.search.embedding-space-mismatch")
            #expect(error.recoverability == .needsConfiguration)
        }
    }
}

private enum FixtureEmbeddingError: Error {
    case failed
}

private struct ThrowingEmbedder: FixtureEmbedder {
    let modelID = "throwing-fixture"
    let dimension = 256

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        throw FixtureEmbeddingError.failed
    }
}

private struct SlowCancellableEmbedder: FixtureEmbedder {
    let modelID = "slow-cancellable"
    let dimension = 8

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        try await Task.sleep(for: .seconds(5))
        return [Float](repeating: 1, count: dimension)
    }
}

private struct WrongQueryDimensionEmbedder: FixtureEmbedder {
    let modelID = "wrong-query-dimension"
    let dimension = 8

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        [Float](repeating: 1, count: kind == .document ? dimension : 3)
    }
}

private struct WrongDocumentDimensionEmbedder: FixtureEmbedder {
    let modelID = "wrong-document-dimension"
    let dimension = 8

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        [Float](repeating: 1, count: kind == .document ? 3 : dimension)
    }
}

private struct QueryFailingEmbedder: FixtureEmbedder {
    let modelID = "query-failing-fixture"
    let dimension = 64
    private let healthy = HashingEmbedder(modelID: "query-failing-fixture", dimension: 64)

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        if kind == .query { throw FixtureEmbeddingError.failed }
        return try await healthy.embed(text, kind: kind)
    }
}

/// Fails only for payloads whose text contains "poison", and embeds everything else normally —
/// so a single bad item can be placed in an otherwise healthy batch.
private struct SelectivelyFailingEmbedder: FixtureEmbedder {
    let modelID = "selective-fixture"
    let dimension = 64
    private let healthy = HashingEmbedder(modelID: "selective-fixture", dimension: 64)

    func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
        if text.lowercased().contains("poison") { throw FixtureEmbeddingError.failed }
        return try await healthy.embed(text, kind: kind)
    }
}
