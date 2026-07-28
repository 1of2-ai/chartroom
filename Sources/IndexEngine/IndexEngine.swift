import Foundation

/// Current concrete engine facade.
///
/// This is intentionally thin today: it wraps the existing SQLite-backed
/// `IndexStore` while the durable document, representation, chunk, and backend
/// layers are built out behind the same public boundary.
public actor IndexEngine: IndexEngineClient {
    public nonisolated let storeURL: URL?
    public nonisolated let configuration: IndexEngineConfiguration

    private let store: IndexStore
    private var lastIngestedAt: Date?
    private var recentFailures: [FailureSnapshot] = []
    private var recentJobs: [JobSnapshot] = []
    private let maxRecentDiagnostics = 1_000

    private init(store: IndexStore, storeURL: URL?, configuration: IndexEngineConfiguration) {
        self.store = store
        self.storeURL = storeURL
        self.configuration = configuration
    }

    /// The engine's real retrieval pipeline, including the live reciprocal-rank-fusion constant
    /// so the description can't drift from the implementation in `IndexStoreRetrieval`.
    public nonisolated var retrievalPipeline: RetrievalPipelineDescriptor {
        RetrievalPipelineDescriptor(
            filterStage: "SQL pre-filter",
            candidateChannels: ["Exact match", "FTS5 BM25", "Vector cosine"],
            fusion: "Reciprocal Rank Fusion (k=\(Int(IndexStore.reciprocalRankK)))"
        )
    }

    /// Open a persistent local index.
    public static func open(
        storeURL: URL,
        configuration: IndexEngineConfiguration = .init()
    ) async throws -> IndexEngine {
        do {
            let store = try IndexStore(path: storeURL.path, embedder: configuration.embedder)
            let resolvedConfiguration = configuration.resolvedForOpen()
            try await store.persistPolicy(
                resolvedConfiguration.defaultPolicy,
                resolutions: resolvedConfiguration.registry.policyStates
            )
            try await store.recoverInterruptedJobs()
            return IndexEngine(store: store, storeURL: storeURL, configuration: resolvedConfiguration)
        } catch let error as IndexEngineError {
            throw error
        } catch let error as IndexStoreSchemaError {
            throw schemaOpenError(error)
        } catch {
            throw storageUnavailableError(
                error,
                code: "index.open.storage-unavailable",
                summary: "The index store could not be opened."
            )
        }
    }

    /// Open an in-memory index for tests, previews, and GUI fixture work.
    public static func openInMemory(
        configuration: IndexEngineConfiguration = .init()
    ) async throws -> IndexEngine {
        do {
            let store = try IndexStore(path: ":memory:", embedder: configuration.embedder)
            let resolvedConfiguration = configuration.resolvedForOpen()
            try await store.persistPolicy(
                resolvedConfiguration.defaultPolicy,
                resolutions: resolvedConfiguration.registry.policyStates
            )
            try await store.recoverInterruptedJobs()
            return IndexEngine(store: store, storeURL: nil, configuration: resolvedConfiguration)
        } catch let error as IndexEngineError {
            throw error
        } catch let error as IndexStoreSchemaError {
            throw schemaOpenError(error)
        } catch {
            throw storageUnavailableError(
                error,
                code: "index.open.storage-unavailable",
                summary: "The in-memory index store could not be opened."
            )
        }
    }

    /// Ingest normalized source payloads through the current text path.
    ///
    /// This already writes to `IndexStore`. Rich extraction, versioned
    /// representations, and chunk lineage will replace the direct one-payload to
    /// one-object mapping without changing the GUI-facing method shape.
    public func ingest(_ request: IngestRequest) async throws -> IngestionSummary {
        let startedAt = Date.now
        let policyResolution = configuration.registry.resolve(policy: request.policy)
        do {
            try Task.checkCancellation()
            try await store.persistPolicy(request.policy, resolutions: [policyResolution])
        } catch is CancellationError {
            await recordJob(.init(
                id: request.jobID,
                state: .cancelled,
                completedUnitCount: 0,
                totalUnitCount: request.payloads.count,
                message: "Ingestion cancelled"
            ))
            throw IndexEngineError(
                .cancelled,
                code: "index.ingest.cancelled",
                recoverability: .retryable,
                summary: "Ingestion was cancelled."
            )
        } catch {
            throw Self.storageUnavailableError(
                error,
                code: "index.ingest.policy-storage-unavailable",
                summary: "The index store could not persist ingestion policy state."
            )
        }
        guard policyResolution.state == .satisfied || policyResolution.state == .degraded else {
            throw IndexEngineError(
                .policyQuarantined,
                code: "index.policy.unsatisfied",
                recoverability: .needsConfiguration,
                summary: "The ingestion policy cannot run with the current registry.",
                detail: policyResolution.message,
                relatedIDs: [request.policy.id]
            )
        }

        await recordJob(.init(
            id: request.jobID,
            state: .running,
            completedUnitCount: 0,
            totalUnitCount: request.payloads.count,
            message: "Ingesting source payloads"
        ))
        var accepted = 0
        var failures: [FailureSnapshot] = []

        do {
            for payload in request.payloads {
                try Task.checkCancellation()
                do {
                    let resolved = try await extractedPayload(for: payload)
                    try Task.checkCancellation()
                    try await store.upsert(try resolved.indexedObject(policy: request.policy))
                    try Task.checkCancellation()
                    accepted += 1
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    failures.append(
                        FailureSnapshot(
                            id: EngineID(rawValue: UUID().uuidString),
                            category: FailureSnapshot.Category(error),
                            message: "Could not ingest \(payload.displayName)",
                            detail: String(describing: error),
                            sourceID: payload.sourceID,
                            documentID: payload.documentID,
                            sourceURI: payload.sourceURI,
                            recoverability: IndexEngineError.Recoverability(error),
                            occurredAt: Date.now
                        )
                    )
                }
            }
        } catch is CancellationError {
            await recordJob(.init(
                id: request.jobID,
                state: .cancelled,
                completedUnitCount: accepted,
                totalUnitCount: request.payloads.count,
                message: "Ingestion cancelled"
            ))
            throw IndexEngineError(
                .cancelled,
                code: "index.ingest.cancelled",
                recoverability: .retryable,
                summary: "Ingestion was cancelled."
            )
        }

        let finishedAt = Date.now
        if accepted > 0 {
            lastIngestedAt = finishedAt
        }
        await recordFailures(failures)
        await recordJob(.init(
            id: request.jobID,
            state: accepted == 0 && !request.payloads.isEmpty && !failures.isEmpty ? .failed : .succeeded,
            completedUnitCount: accepted,
            totalUnitCount: request.payloads.count,
            message: "\(accepted) accepted, \(failures.count) failed"
        ))

        if accepted == 0, !request.payloads.isEmpty, let first = failures.first {
            throw IndexEngineError(
                .ingestionFailed,
                code: "index.ingest.failed",
                recoverability: first.recoverability,
                summary: first.message,
                detail: first.detail,
                relatedIDs: [first.id]
            )
        }

        return IngestionSummary(
            jobID: request.jobID,
            acceptedCount: accepted,
            failedCount: failures.count,
            failures: failures,
            startedAt: startedAt,
            finishedAt: finishedAt
        )
    }

    /// Run a registered `ContentExtractor` for binary payloads whose content type it
    /// supports, rewriting the payload to carry the extracted text so it flows through
    /// the existing text → chunk → embed path. Payloads with no matching extractor (or a
    /// non-binary body) pass through unchanged. Extractor errors propagate to the caller
    /// and become durable failure records.
    private func extractedPayload(for payload: SourcePayload) async throws -> SourcePayload {
        guard case .binaryReference = payload.body,
              let extractor = configuration.extractors.first(where: {
                  $0.supportedContentTypes.contains(payload.contentType)
              })
        else { return payload }

        let representations = try await extractor.extract(payload, options: ExtractionOptions())
        guard let representation = representations.first else { return payload }

        var rewritten = payload
        rewritten.body = .preExtracted(kind: representation.kind, text: representation.text)
        rewritten.metadata.merge(representation.metadata) { _, new in new }
        return rewritten
    }

    /// Remove documents through the public engine boundary.
    ///
    /// Connectors may report deletes, but the app should apply them here so the
    /// engine remains the owner of all durable index mutations.
    public func delete(_ request: DeleteRequest) async throws -> DeletionSummary {
        let startedAt = Date.now
        await recordJob(.init(
            id: request.jobID,
            state: .running,
            kind: .delete,
            completedUnitCount: 0,
            totalUnitCount: request.documentIDs.count,
            message: "Deleting indexed documents"
        ))
        var deleted = 0
        var failures: [FailureSnapshot] = []

        for documentID in request.documentIDs {
            do {
                let didDelete = try await store.delete(id: documentID.rawValue)
                if didDelete {
                    deleted += 1
                }
            } catch {
                failures.append(
                    FailureSnapshot(
                        id: EngineID(rawValue: UUID().uuidString),
                        category: .storageFailure,
                        message: "Could not delete \(documentID.rawValue)",
                        detail: String(describing: error),
                        documentID: documentID,
                        recoverability: IndexEngineError.Recoverability(error),
                        occurredAt: Date.now
                    )
                )
            }
        }

        let finishedAt = Date.now
        await recordFailures(failures)
        await recordJob(.init(
            id: request.jobID,
            state: deleted == 0 && !request.documentIDs.isEmpty && !failures.isEmpty ? .failed : .succeeded,
            kind: .delete,
            completedUnitCount: deleted,
            totalUnitCount: request.documentIDs.count,
            message: "\(deleted) deleted, \(failures.count) failed"
        ))

        if deleted == 0, !request.documentIDs.isEmpty, let first = failures.first {
            throw IndexEngineError(
                .deletionFailed,
                code: "index.delete.failed",
                recoverability: first.recoverability,
                summary: first.message,
                detail: first.detail,
                relatedIDs: [first.id]
            )
        }

        return DeletionSummary(
            jobID: request.jobID,
            requestedCount: request.documentIDs.count,
            deletedCount: deleted,
            failedCount: failures.count,
            failures: failures,
            startedAt: startedAt,
            finishedAt: finishedAt
        )
    }

    private static func storageUnavailableError(_ error: Error, code: String, summary: String) -> IndexEngineError {
        if let engineError = error as? IndexEngineError {
            return engineError
        }
        return IndexEngineError(
            .storageUnavailable,
            code: code,
            recoverability: .retryable,
            summary: summary,
            detail: String(describing: error)
        )
    }

    private static func schemaOpenError(_ error: IndexStoreSchemaError) -> IndexEngineError {
        switch error {
        case .storeFromNewerBuild:
            return IndexEngineError(
                .migrationRequired,
                code: "index.open.newer-schema",
                recoverability: .needsUserAction,
                summary: "This index requires a newer version of Chartroom.",
                detail: String(describing: error)
            )
        case .referentialIntegrityCheckFailed:
            return IndexEngineError(
                .migrationFailed,
                code: "index.open.migration-failed",
                recoverability: .needsUserAction,
                summary: "The index store could not be migrated safely.",
                detail: String(describing: error)
            )
        }
    }

    /// Search the active index.
    ///
    /// The current implementation maps to the first hybrid `IndexStore` path:
    /// FTS5 plus vector similarity with reciprocal rank fusion. The response
    /// type already carries the diagnostics the GUI needs as richer backends are
    /// added.
    public func search(_ request: SearchRequest) async throws -> SearchResponse {
        let execution: (hits: [SearchHit], diagnostics: SearchDiagnostics)
        do {
            execution = try await store.searchDetailed(
                request.query,
                scope: request.filters.clusterID.map { Scope.cluster($0.rawValue, hard: true) } ?? .global,
                filters: request.filters,
                limit: request.limit,
                allowDegradedResults: request.allowDegradedResults,
                profile: configuration.retrievalProfile.resolved(for: request.mode)
            )
        } catch {
            throw searchError(error)
        }

        let hits = execution.hits
        let results = hits.map { hit in
            SearchResultSnapshot(
                id: EngineID(rawValue: hit.id),
                documentID: EngineID(rawValue: hit.documentID),
                chunkID: EngineID(rawValue: hit.chunkID),
                sourceID: hit.sourceID.map(EngineID.init(rawValue:)),
                title: hit.title,
                snippet: hit.snippet,
                sourceURI: hit.sourceURI,
                contentType: hit.type,
                score: hit.score,
                rank: 0,
                similarity: hit.similarity,
                isWeak: hit.isWeak,
                lineStart: hit.lineStart,
                lineEnd: hit.lineEnd,
                diagnostics: SearchResultDiagnostics(
                    ftsRank: hit.keywordRank,
                    vectorRank: hit.vectorRank,
                    exactRank: hit.exactRank,
                    keywordScore: hit.keywordScore,
                    graphReason: nil,
                    appliedBoosts: []
                ),
                provenance: ResultProvenance(
                    connectorID: hit.sourceID.map(EngineID.init(rawValue:)),
                    policyID: hit.policyID.map(EngineID.init(rawValue:)),
                    representationID: hit.representationID.map(EngineID.init(rawValue:)),
                    embeddingSpaceID: hit.embeddingSpaceID.map(EngineID.init(rawValue:))
                )
            )
        }
        .enumerated()
        .map { index, result in
            var ranked = result
            ranked.rank = index + 1
            return ranked
        }

        return SearchResponse(
            query: request.query,
            mode: request.mode,
            results: results,
            diagnostics: execution.diagnostics
        )
    }

    /// Classify a thrown search failure into the typed error contract.
    ///
    /// The GUI chooses its recovery affordance from the category and
    /// recoverability, so a storage fault must not masquerade as an embedding
    /// fault. A query/vector dimension mismatch is an embedding-space problem,
    /// a SQLite fault is a storage problem, and anything else is treated as the
    /// embedding provider failing to produce a query vector.
    private func searchError(_ error: Error) -> IndexEngineError {
        switch error {
        case let engineError as IndexEngineError:
            return engineError
        case let storeError as IndexStoreError:
            return IndexEngineError(
                .embeddingSpaceUnavailable,
                code: "index.search.embedding-space-mismatch",
                recoverability: .needsConfiguration,
                summary: "The query embedding is not comparable to the stored embedding space.",
                detail: String(describing: storeError)
            )
        case let storageError as SQLiteError:
            return IndexEngineError(
                .storageUnavailable,
                code: "index.search.storage-unavailable",
                recoverability: .retryable,
                summary: "The index store could not complete the search query.",
                detail: String(describing: storageError)
            )
        default:
            return IndexEngineError(
                .embeddingProviderUnavailable,
                code: "index.search.embedding-unavailable",
                recoverability: .needsConfiguration,
                summary: "The embedding provider could not produce a query vector.",
                detail: String(describing: error)
            )
        }
    }

    public func health() async -> IndexHealthSnapshot {
        let snapshot = await snapshot()
        return IndexHealthSnapshot(
            objectCount: snapshot.objectCount,
            documentCount: snapshot.documentCount,
            chunkCount: snapshot.chunkCount,
            embeddingCount: snapshot.embeddingCount,
            policyStates: snapshot.policyStates,
            vectorBackendStatus: await store.vectorBackendStatus()
        )
    }

    /// Browse documents currently visible to search through a typed projection.
    ///
    /// This is a GUI-facing query contract, not a storage escape hatch. It supports the same
    /// source, type, policy, cluster, and embedding-space filters used by search, and throws typed
    /// storage errors instead of making the document browser silently look empty.
    public func browseDocuments(_ request: DocumentBrowseRequest = .init()) async throws -> DocumentBrowseResponse {
        do {
            return try await store.documentSummaries(request: request)
        } catch {
            throw IndexEngineError(
                .storageUnavailable,
                code: "index.documents.storage-unavailable",
                recoverability: .retryable,
                summary: "The index store could not browse documents.",
                detail: String(describing: error)
            )
        }
    }

    public func chunks(forDocument documentID: DocumentID) async throws -> [ChunkSummary] {
        do {
            return try await store.chunkSummaries(documentID: documentID.rawValue)
        } catch {
            throw IndexEngineError(
                .storageUnavailable,
                code: "index.chunks.storage-unavailable",
                recoverability: .retryable,
                summary: "The index store could not read the document's chunks.",
                detail: String(describing: error)
            )
        }
    }

    public func failures(limit: Int = 50) async -> [FailureSnapshot] {
        guard limit > 0 else { return [] }
        let durableFailures = (try? await store.failureSnapshots(limit: limit + recentFailures.count)) ?? []
        return Self.mergedFailures(durableFailures, recentFailures: recentFailures, limit: limit)
    }

    public func jobs(limit: Int = 50) async -> [JobSnapshot] {
        guard limit > 0 else { return [] }
        let durableJobs = (try? await store.jobSnapshots(limit: limit + recentJobs.count)) ?? []
        return Self.mergedJobs(durableJobs, recentJobs: recentJobs, limit: limit)
    }

    /// Report the embedding provider's health *and* what kind of provider it is.
    ///
    /// A hashing fallback probes as perfectly available, so availability alone told hosts that
    /// semantic retrieval was working when the index had no semantic channel at all. The message
    /// leads with the degradation whenever the provider is not model-backed, because a host that
    /// renders only the message must still convey it.
    public func modelStatus() async -> ModelStatusSnapshot {
        let status = await store.embeddingProviderStatus()
        let isModelBacked = store.isModelBacked
        let message: String
        if !isModelBacked {
            message = """
            Not model-backed: '\(store.modelID)' produces vector-shaped output without semantic \
            meaning, so retrieval is lexical only. \(status.message)
            """
        } else {
            message = status.message
        }
        return ModelStatusSnapshot(
            modelID: store.modelID,
            embeddingSpaceID: EngineID(rawValue: store.embeddingSpaceID),
            dimension: store.dimension,
            isAvailable: status.isAvailable,
            isModelBacked: isModelBacked,
            message: message
        )
    }

    public func rebuildEmbeddings() async throws -> EmbeddingRebuildSummary {
        do {
            return try await store.rebuildActiveEmbeddingSpace()
        } catch {
            throw rebuildError(error)
        }
    }

    /// A rebuild fails for the same underlying reasons as a search, but on the *document* side of
    /// the embedder. Reusing the search messages here would report a failed query vector for what
    /// is actually a failed document embedding.
    private func rebuildError(_ error: Error) -> IndexEngineError {
        switch error {
        case let engineError as IndexEngineError:
            return engineError
        case let storeError as IndexStoreError:
            return IndexEngineError(
                .embeddingSpaceUnavailable,
                code: "index.rebuild.embedding-space-mismatch",
                recoverability: .needsConfiguration,
                summary: "The rebuilt vectors do not match the store's embedding dimension.",
                detail: String(describing: storeError)
            )
        case let storageError as SQLiteError:
            return IndexEngineError(
                .storageUnavailable,
                code: "index.rebuild.storage-unavailable",
                recoverability: .retryable,
                summary: "The index store could not write the rebuilt embeddings.",
                detail: String(describing: storageError)
            )
        default:
            return IndexEngineError(
                .embeddingProviderUnavailable,
                code: "index.rebuild.embedding-unavailable",
                recoverability: .needsConfiguration,
                summary: "The embedding provider could not embed the stored content.",
                detail: String(describing: error)
            )
        }
    }

    public func snapshot() async -> IndexEngineSnapshot {
        let counts: IndexStoreCounts
        do {
            counts = try await store.counts()
        } catch {
            appendInMemoryDiagnosticFailure(
                code: "index.snapshot.counts-unavailable",
                message: "The index store could not read snapshot counts.",
                error: error
            )
            counts = IndexStoreCounts(documentCount: 0, chunkCount: 0, embeddingCount: 0)
        }
        return IndexEngineSnapshot(
            storeURL: storeURL,
            storeByteSize: await store.storeByteSize(),
            objectCount: counts.documentCount,
            documentCount: counts.documentCount,
            chunkCount: counts.chunkCount,
            embeddingCount: counts.embeddingCount,
            modelID: store.modelID,
            embeddingDimension: store.dimension,
            embeddingSpaceID: EngineID(rawValue: store.embeddingSpaceID),
            embeddingSpaceCoverage: try? await store.embeddingSpaceCoverage(),
            lastIngestedAt: lastIngestedAt,
            policyStates: configuration.registry.policyStates
        )
    }

    /// Clear failure diagnostics from both the durable store and the in-memory recent buffer.
    /// `ids == nil` clears every recorded failure.
    public func clearFailures(ids: Set<EngineID>? = nil) async throws {
        if let ids {
            let rawIDs = Set(ids.map(\.rawValue))
            recentFailures.removeAll { rawIDs.contains($0.id.rawValue) }
            try await store.deleteFailures(ids: rawIDs)
        } else {
            recentFailures.removeAll()
            try await store.deleteFailures(ids: nil)
        }
    }

    private func recordFailures(_ failures: [FailureSnapshot]) async {
        guard !failures.isEmpty else { return }
        for failure in failures {
            recentFailures.append(failure)
            do {
                try await store.recordFailure(failure)
            } catch {
                appendInMemoryDiagnosticFailure(
                    code: "index.diagnostics.failure-write-unavailable",
                    message: "The index store could not persist a failure diagnostic.",
                    error: error
                )
            }
        }
        trimDiagnostics()
    }

    private func recordJob(_ job: JobSnapshot) async {
        recentJobs.append(job)
        do {
            try await store.recordJob(job)
        } catch {
            appendInMemoryDiagnosticFailure(
                code: "index.diagnostics.job-write-unavailable",
                message: "The index store could not persist a job diagnostic.",
                error: error
            )
        }
        trimDiagnostics()
    }

    private func appendInMemoryDiagnosticFailure(code: String, message: String, error: Error) {
        recentFailures.append(
            FailureSnapshot(
                id: EngineID(rawValue: code),
                category: .storageFailure,
                message: message,
                detail: String(describing: error),
                recoverability: .retryable,
                occurredAt: Date.now
            )
        )
        trimDiagnostics()
    }

    private func trimDiagnostics() {
        if recentFailures.count > maxRecentDiagnostics {
            recentFailures.removeFirst(recentFailures.count - maxRecentDiagnostics)
        }
        if recentJobs.count > maxRecentDiagnostics {
            recentJobs.removeFirst(recentJobs.count - maxRecentDiagnostics)
        }
    }

    private static func mergedFailures(
        _ durableFailures: [FailureSnapshot],
        recentFailures: [FailureSnapshot],
        limit: Int
    ) -> [FailureSnapshot] {
        var byID: [EngineID: FailureSnapshot] = [:]
        for failure in durableFailures + recentFailures {
            if let existing = byID[failure.id], existing.occurredAt >= failure.occurredAt {
                continue
            }
            byID[failure.id] = failure
        }
        return Array(byID.values)
            .sorted { lhs, rhs in
                if lhs.occurredAt == rhs.occurredAt {
                    return lhs.id.rawValue < rhs.id.rawValue
                }
                return lhs.occurredAt > rhs.occurredAt
            }
            .prefix(limit)
            .map { $0 }
    }

    private static func mergedJobs(
        _ durableJobs: [JobSnapshot],
        recentJobs: [JobSnapshot],
        limit: Int
    ) -> [JobSnapshot] {
        var seen = Set<EngineID>()
        var merged: [JobSnapshot] = []
        for job in Array(recentJobs.reversed()) + durableJobs {
            guard !seen.contains(job.id) else { continue }
            seen.insert(job.id)
            merged.append(job)
            if merged.count == limit { break }
        }
        return merged
    }
}
