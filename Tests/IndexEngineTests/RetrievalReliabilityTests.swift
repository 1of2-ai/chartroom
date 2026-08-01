import Foundation
import Testing
@testable import IndexEngine

@Suite("Retrieval reliability")
struct RetrievalReliabilityTests {
    private enum CoverageReadFailure: Error {
        case injected
    }

    private func temporaryStorePath(_ label: String) -> String {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("\(label)-\(UUID().uuidString).sqlite")
            .path
    }

    @Test("degraded search surfaces a failed coverage read as unavailable")
    func degradedCoverageFailureIsUnavailable() async throws {
        let path = temporaryStorePath("degraded-coverage")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let original = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "coverage-original", dimension: 4)
        )
        try await original.upsert(.init(
            id: "coverage-document",
            type: "note",
            title: "Coverage",
            body: "lexical coverage marker"
        ))
        let swapped = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "coverage-swapped", dimension: 4)
        )
        await swapped.replaceEmbeddingSpaceCoverageReader {
            throw CoverageReadFailure.injected
        }

        let response = try await swapped.searchDetailed(
            "lexical coverage",
            limit: 5,
            allowDegradedResults: true
        )

        #expect(response.hits.map(\.documentID) == ["coverage-document"])
        #expect(response.diagnostics.embeddingSpaceCoverageState == .unavailable)
        #expect(response.diagnostics.degraded)
        #expect(!response.diagnostics.embeddingSpaceMismatch)
    }

    @Test("strict search propagates a failed coverage read")
    func strictCoverageFailurePropagates() async throws {
        let path = temporaryStorePath("strict-coverage")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let original = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "strict-original", dimension: 4)
        )
        try await original.upsert(.init(
            id: "strict-document",
            type: "note",
            title: "Strict",
            body: "strict lexical marker"
        ))
        let swapped = try IndexStore(
            path: path,
            embedder: HashingEmbedder(modelID: "strict-swapped", dimension: 4)
        )
        await swapped.replaceEmbeddingSpaceCoverageReader {
            throw CoverageReadFailure.injected
        }

        await #expect(throws: CoverageReadFailure.self) {
            _ = try await swapped.searchDetailed(
                "strict lexical",
                limit: 5,
                allowDegradedResults: false
            )
        }
    }

    @Test("cluster lookup maps candidates across bounded SQL batches")
    func clusterLookupUsesBoundedBatches() async throws {
        let store = try IndexStore(path: ":memory:", embedder: HashingEmbedder(dimension: 4))
        var expected: [String: String] = [:]
        for index in 0..<600 {
            let documentID = "cluster-map-\(index)"
            let clusterID = "cluster-\(index % 2)"
            try await store.upsert(.init(
                id: documentID,
                type: "note",
                title: "Candidate \(index)",
                body: "candidate text \(index)",
                clusterID: clusterID
            ))
            let chunk = try #require(
                try await store.chunkSummaries(documentID: documentID).first
            )
            expected[chunk.id.rawValue] = clusterID
        }

        let preparations = StatementPreparationRecorder()
        await store.observeSQLiteStatementPreparations { sql in
            if sql.contains("SELECT chunks.id, documents.cluster_id FROM chunks") {
                preparations.record()
            }
        }
        let actual = try await store.clusterMap(Set(expected.keys))
        await store.observeSQLiteStatementPreparations(nil)

        #expect(actual == expected)
        #expect(preparations.count == 2)
    }
}

private final class StatementPreparationRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = 0

    var count: Int {
        lock.withLock { storage }
    }

    func record() {
        lock.withLock {
            storage += 1
        }
    }
}
