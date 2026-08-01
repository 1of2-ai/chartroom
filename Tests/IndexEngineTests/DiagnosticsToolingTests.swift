import Foundation
import Testing
@testable import IndexEngine

@Suite("SearchBenchmark")
struct SearchBenchmarkTests {
    @Test("nearest-rank percentile handles empty, single, and multi-sample sets")
    func percentileNearestRank() {
        #expect(SearchBenchmark.percentile([], 0.50) == nil)
        #expect(SearchBenchmark.percentile([0.2], 0.50) == 0.2)
        #expect(SearchBenchmark.percentile([0.2], 0.95) == 0.2)

        let values: [TimeInterval] = (1...100).map { TimeInterval($0) / 1000 }
        #expect(SearchBenchmark.percentile(values, 0.50) == 0.050)
        #expect(SearchBenchmark.percentile(values, 0.95) == 0.095)
    }

    @Test("a report aggregates per-layer stats and skips layers without samples")
    func reportAggregatesLayers() {
        let samples = [
            SearchBenchmark.Sample(diagnostics: SearchDiagnostics(ftsLatency: 0.010, totalLatency: 0.030)),
            SearchBenchmark.Sample(diagnostics: SearchDiagnostics(ftsLatency: 0.020, totalLatency: 0.010)),
        ]
        let report = SearchBenchmark.Report(queries: ["a", "b"], iterations: 1, samples: samples)

        #expect(report.totalRuns == 2)
        let total = report.layers.first { $0.id == "Total" }
        #expect(total?.p50 == 0.010)
        #expect(total?.p95 == 0.030)
        #expect(total?.sampleCount == 2)
        let vector = report.layers.first { $0.id == "Vector" }
        #expect(vector?.sampleCount == 0)
        #expect(vector?.p50 == nil)
    }

    @Test("run executes queries × iterations searches in diagnostic mode")
    func runExecutesQueryMatrix() async throws {
        let engine = BenchmarkScriptedEngine()
        let report = try await SearchBenchmark.run(engine: engine, queries: ["one", "two"], iterations: 3, limit: 7)

        #expect(report.totalRuns == 6)
        let requests = await engine.recordedRequests()
        #expect(requests.count == 6)
        #expect(requests.allSatisfy { $0.mode == .diagnostic && $0.limit == 7 })
        #expect(Set(requests.map(\.query)) == ["one", "two"])
    }
}

@Suite("DiagnosticsBundle")
struct DiagnosticsBundleTests {
    @Test("SearchResponse round-trips through Codable")
    func searchResponseRoundTrips() throws {
        let response = SearchResponse(
            query: "boundary",
            mode: .diagnostic,
            results: [],
            diagnostics: SearchDiagnostics(ftsLatency: 0.012, totalLatency: 0.034)
        )
        let data = try JSONEncoder().encode(response)
        let decoded = try JSONDecoder().decode(SearchResponse.self, from: data)
        #expect(decoded == response)
    }

    @Test("legacy failure snapshots decode isRecoverable without recoverability")
    func legacyFailureSnapshotDecodes() throws {
        let data = Data("""
        {
          "id": "legacy-failure",
          "category": "storageFailure",
          "message": "legacy",
          "detail": "legacy detail",
          "isRecoverable": false,
          "occurredAt": 0
        }
        """.utf8)

        let decoded = try JSONDecoder().decode(FailureSnapshot.self, from: data)
        #expect(decoded.recoverability == .unrecoverable)
        #expect(!decoded.isRecoverable)
    }

    @Test("legacy job snapshots decode a missing kind as ingest")
    func legacyJobSnapshotDecodes() throws {
        let data = Data("""
        {
          "id": "legacy-job",
          "state": "succeeded",
          "completedUnitCount": 2,
          "totalUnitCount": 2,
          "message": "done"
        }
        """.utf8)

        let decoded = try JSONDecoder().decode(JobSnapshot.self, from: data)
        #expect(decoded.kind == .ingest)
    }

    @Test("default export removes user content and stale last-search output")
    func defaultExportIsSafeForSharing() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "diagnostics-bundle-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let bundle = sensitiveBundle()
        try bundle.write(to: directory, mode: .includeSensitiveData)
        #expect(FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("last-search.json").path
        ))
        try bundle.write(to: directory)

        let written = try Set(FileManager.default.contentsOfDirectory(atPath: directory.path))
        #expect(written == [
            "snapshot.json", "health.json", "model-status.json",
            "jobs.json", "failures.json", "history-availability.json",
        ])
        let exportedText = try written
            .map { try String(contentsOf: directory.appendingPathComponent($0), encoding: .utf8) }
            .joined(separator: "\n")
        for sentinel in [
            "/Users/private/Chartroom.sqlite",
            "secret query",
            "secret title",
            "secret snippet",
            "secret-source",
            "secret-document",
            "secret failure detail",
            "secret failure message",
            "secret job message",
            "secret model message",
            "secret policy message",
            "secret backend message",
        ] {
            #expect(!exportedText.contains(sentinel), "Safe export leaked: \(sentinel)")
        }

        // Export is a projection. The in-memory diagnostic object remains useful to the app.
        #expect(bundle.snapshot.storeURL?.path == "/Users/private/Chartroom.sqlite")
        #expect(bundle.lastSearch?.query == "secret query")
        #expect(bundle.failures.first?.detail == "secret failure detail")
    }

    @Test("a failed safe rewrite removes prior sensitive export artifacts first")
    func failedSafeRewriteDoesNotLeaveSensitiveArtifacts() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "diagnostics-failed-safe-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let unrelatedURL = directory.appendingPathComponent("keep-me.txt")
        try Data("unrelated".utf8).write(to: unrelatedURL)
        let bundle = sensitiveBundle()
        try bundle.write(to: directory, mode: .includeSensitiveData)

        #expect(throws: InjectedDiagnosticsWriteFailure.self) {
            try bundle.write(
                to: directory,
                mode: .safeForSharing,
                fileWriter: { _, _ in throw InjectedDiagnosticsWriteFailure() }
            )
        }

        for name in [
            "snapshot.json", "health.json", "model-status.json",
            "jobs.json", "failures.json", "history-availability.json", "last-search.json",
        ] {
            #expect(!FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(name).path
            ))
        }
        #expect(FileManager.default.fileExists(atPath: unrelatedURL.path))
    }

    @Test("safe cleanup refuses artifact-named directories and still clears later sensitive files")
    func safeCleanupDoesNotRecursivelyDeleteDirectories() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "diagnostics-blocked-cleanup-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let bundle = sensitiveBundle()
        try bundle.write(to: directory, mode: .includeSensitiveData)
        let blockedURL = directory.appendingPathComponent("snapshot.json")
        try FileManager.default.removeItem(at: blockedURL)
        try FileManager.default.createDirectory(at: blockedURL, withIntermediateDirectories: false)
        let markerURL = blockedURL.appendingPathComponent("do-not-delete.txt")
        try Data("unrelated".utf8).write(to: markerURL)

        #expect(throws: (any Error).self) {
            try bundle.write(to: directory)
        }

        #expect(FileManager.default.fileExists(atPath: markerURL.path))
        for name in [
            "health.json", "model-status.json", "jobs.json",
            "failures.json", "history-availability.json", "last-search.json",
        ] {
            #expect(!FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(name).path
            ))
        }
    }

    @Test("sensitive export explicitly preserves the complete diagnostic payload")
    func sensitiveExportPreservesPayload() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "diagnostics-full-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let bundle = sensitiveBundle()
        try bundle.write(to: directory, mode: .includeSensitiveData)

        let lastSearch = try Data(contentsOf: directory.appendingPathComponent("last-search.json"))
        let decoded = try JSONDecoder().decode(SearchResponse.self, from: lastSearch)
        #expect(decoded.query == "secret query")
        let failures = try Data(contentsOf: directory.appendingPathComponent("failures.json"))
        #expect(String(decoding: failures, as: UTF8.self).contains("secret failure detail"))
    }

    @Test("default client history reports that durable completeness is unsupported")
    func defaultClientHistoryIsNotSupported() async {
        let history = await BenchmarkScriptedEngine().diagnosticHistory(limit: 10)
        #expect(history.jobsAvailability == .notSupported)
        #expect(history.failuresAvailability == .notSupported)
    }

    private func sensitiveBundle() -> DiagnosticsBundle {
        let policy = PolicyResolution(
            policyID: "private-policy",
            state: .degraded,
            missingComponents: ["private-component"],
            message: "secret policy message"
        )
        return DiagnosticsBundle(
            snapshot: IndexEngineSnapshot(
                storeURL: URL(filePath: "/Users/private/Chartroom.sqlite"),
                objectCount: 2,
                modelID: "test",
                embeddingDimension: 4,
                embeddingSpaceID: "test:4",
                lastIngestedAt: nil,
                policyStates: [policy]
            ),
            health: IndexHealthSnapshot(
                objectCount: 2,
                policyStates: [policy],
                vectorBackendStatus: VectorStorageStatus(
                    backendID: "private-backend",
                    state: .degraded,
                    message: "secret backend message"
                )
            ),
            modelStatus: ModelStatusSnapshot(
                modelID: "test",
                embeddingSpaceID: "test:4",
                dimension: 4,
                isAvailable: false,
                isModelBacked: true,
                message: "secret model message"
            ),
            jobs: [
                JobSnapshot(
                    id: "secret-job",
                    state: .failed,
                    completedUnitCount: 1,
                    totalUnitCount: 2,
                    message: "secret job message"
                ),
            ],
            failures: [
                FailureSnapshot(
                    id: "secret-failure",
                    category: .storageFailure,
                    message: "secret failure message",
                    detail: "secret failure detail",
                    sourceID: "secret-source",
                    documentID: "secret-document",
                    sourceURI: URL(filePath: "/Users/private/source.md"),
                    recoverability: .retryable,
                    occurredAt: Date(timeIntervalSince1970: 1)
                ),
            ],
            lastSearch: SearchResponse(
                query: "secret query",
                mode: .diagnostic,
                results: [
                    SearchResultSnapshot(
                        id: "secret-result",
                        documentID: "secret-document",
                        chunkID: "secret-chunk",
                        sourceID: "secret-source",
                        title: "secret title",
                        snippet: "secret snippet",
                        sourceURI: URL(filePath: "/Users/private/source.md"),
                        contentType: "note",
                        score: 1,
                        rank: 1,
                        diagnostics: .init(),
                        provenance: .init()
                    ),
                ],
                diagnostics: SearchDiagnostics()
            )
        )
    }
}

private struct InjectedDiagnosticsWriteFailure: Error {}

/// Returns empty diagnostic-mode responses and records every search request.
private actor BenchmarkScriptedEngine: IndexEngineClient {
    private var requests: [SearchRequest] = []

    func recordedRequests() -> [SearchRequest] {
        requests
    }

    func search(_ request: SearchRequest) async throws -> SearchResponse {
        requests.append(request)
        return SearchResponse(
            query: request.query,
            mode: request.mode,
            results: [],
            diagnostics: SearchDiagnostics(totalLatency: 0.001)
        )
    }

    func ingest(_ request: IngestRequest) async throws -> IngestionSummary {
        IngestionSummary(jobID: request.jobID, acceptedCount: 0, failedCount: 0, failures: [], startedAt: .now, finishedAt: .now)
    }

    func delete(_ request: DeleteRequest) async throws -> DeletionSummary {
        DeletionSummary(jobID: request.jobID, requestedCount: 0, deletedCount: 0, failedCount: 0, failures: [], startedAt: .now, finishedAt: .now)
    }

    func browseDocuments(_ request: DocumentBrowseRequest) async throws -> DocumentBrowseResponse {
        DocumentBrowseResponse(request: request, documents: [], totalMatching: 0)
    }

    func rebuildEmbeddings() async throws -> EmbeddingRebuildSummary {
        EmbeddingRebuildSummary()
    }

    func health() async -> IndexHealthSnapshot {
        IndexHealthSnapshot(objectCount: 0, policyStates: [])
    }

    func failures(limit: Int) async -> [FailureSnapshot] {
        []
    }

    func jobs(limit: Int) async -> [JobSnapshot] {
        []
    }

    func modelStatus() async -> ModelStatusSnapshot {
        ModelStatusSnapshot(modelID: "benchmark", embeddingSpaceID: nil, dimension: 0, isAvailable: true, isModelBacked: false)
    }

    func snapshot() async -> IndexEngineSnapshot {
        IndexEngineSnapshot(
            storeURL: nil,
            objectCount: 0,
            modelID: "benchmark",
            embeddingDimension: 0,
            embeddingSpaceID: "benchmark:0",
            lastIngestedAt: nil,
            policyStates: []
        )
    }
}
