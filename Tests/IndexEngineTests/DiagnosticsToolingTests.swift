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

    @Test("writes one JSON file per snapshot, plus last-search when present")
    func writesBundleFiles() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "diagnostics-bundle-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let bundle = DiagnosticsBundle(
            snapshot: IndexEngineSnapshot(
                storeURL: nil,
                objectCount: 2,
                modelID: "test",
                embeddingDimension: 0,
                embeddingSpaceID: "test:0",
                lastIngestedAt: nil,
                policyStates: []
            ),
            health: IndexHealthSnapshot(objectCount: 2, policyStates: []),
            modelStatus: ModelStatusSnapshot(modelID: "test", embeddingSpaceID: nil, dimension: 0, isAvailable: true, isModelBacked: false),
            jobs: [],
            failures: [],
            lastSearch: SearchResponse(query: "q", mode: .fast, results: [], diagnostics: SearchDiagnostics())
        )
        try bundle.write(to: directory)

        let written = try Set(FileManager.default.contentsOfDirectory(atPath: directory.path))
        #expect(written == [
            "snapshot.json", "health.json", "model-status.json",
            "jobs.json", "failures.json", "last-search.json",
        ])

        let lastSearch = try Data(contentsOf: directory.appendingPathComponent("last-search.json"))
        let decoded = try JSONDecoder().decode(SearchResponse.self, from: lastSearch)
        #expect(decoded.query == "q")
    }
}

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
