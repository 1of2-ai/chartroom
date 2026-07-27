import ConnectorEngine
import Foundation
import IndexEngine
import SyncEngine
import Testing
@testable import ChartroomControl

/// The workspace is where cross-index relevance is decided, so it is where the weak tail has to be
/// separated and where the ordering claim has to be honest. Doing either in a view would leave the
/// merge, the result counts, and every non-GUI client still treating rank artifacts as relevance.
@Suite("Workspace relevance")
struct WorkspaceRelevanceTests {
    private func makeRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "chartroom-workspace-relevance-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func makeWorkspace(root: URL) -> ChartroomWorkspace {
        ChartroomWorkspace(
            catalogURL: root.appending(path: "Indexes.json"),
            storesDirectory: root.appending(path: "Indexes", directoryHint: .isDirectory),
            engineFactory: { _ in (try await IndexEngine.openInMemory(), nil) },
            cursorStore: RelevanceTestCursorStore()
        )
    }

    private func seed(
        _ workspace: ChartroomWorkspace,
        indexNamed name: String,
        root: URL,
        files: [String: String]
    ) async throws -> ChartroomIndex {
        let index = try await workspace.createIndex(named: name)
        let source = root.appending(path: name, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        for (fileName, body) in files {
            try body.write(to: source.appending(path: fileName), atomically: true, encoding: .utf8)
        }
        let session = try await workspace.session(for: index.id)
        _ = try await session.open()
        _ = try await session.ingestLocalSource(source)
        return index
    }

    /// The C1 reproduction at the workspace layer: nonsense against a populated workspace used to
    /// fill `results` with the corpus.
    @Test("a query matching nothing yields no results, only a weak tail")
    func nonsenseQueryProducesNoResults() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)

        _ = try await seed(workspace, indexNamed: "Personal", root: root, files: [
            "Atlas.md": "atlas routing tables converge on the shortest path",
            "Ledger.md": "ledger synchronization reconciles pending transactions",
        ])

        let response = try await workspace.search(.init(query: "zzqqxx nonexistent term", limit: 10))

        #expect(response.results.isEmpty, "nothing matched, so nothing may be presented as an answer")
        #expect(!response.weakResults.isEmpty, "top-N vector search still produces the tail")
        #expect(response.weakResults.allSatisfy { $0.result.isWeak })
        #expect(response.allResults.count == response.weakResults.count)
    }

    @Test("a real query yields results and no weak tail displacing them")
    func realQueryProducesResults() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)

        _ = try await seed(workspace, indexNamed: "Personal", root: root, files: [
            "Atlas.md": "atlas routing tables converge on the shortest path",
            "Ledger.md": "ledger synchronization reconciles pending transactions",
        ])

        let response = try await workspace.search(.init(query: "atlas routing", limit: 10))

        let best = try #require(response.results.first)
        #expect(best.result.documentID.rawValue.hasSuffix("Atlas.md"))
        #expect(best.result.isWeak == false)
        #expect(response.results.allSatisfy { !$0.result.isWeak })
    }

    /// Ranks are assigned per partition, so the weak tail does not consume the numbering the real
    /// answers use.
    @Test("both partitions are ranked from one")
    func partitionsAreRankedIndependently() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)

        _ = try await seed(workspace, indexNamed: "Personal", root: root, files: [
            "Atlas.md": "atlas routing tables converge on the shortest path",
            "Ledger.md": "ledger synchronization reconciles pending transactions",
            "Mel.md": "mel spectrogram frontend windows the audio signal",
        ])

        let response = try await workspace.search(.init(query: "atlas routing", limit: 10))

        #expect(response.results.map(\.rank) == Array(1...response.results.count))
        if !response.weakResults.isEmpty {
            #expect(response.weakResults.map(\.rank) == Array(1...response.weakResults.count))
        }
    }

    /// One shared embedding space and a similarity on every result means the merge can order on an
    /// absolute quantity. That is the only condition under which cross-index ordering is sound.
    @Test("a single-embedding-space merge orders by similarity, descending")
    func mergeOrdersBySimilarityWhenComparable() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)

        _ = try await seed(workspace, indexNamed: "Personal", root: root, files: [
            "Atlas.md": "atlas routing tables converge on the shortest path",
        ])
        _ = try await seed(workspace, indexNamed: "Work", root: root, files: [
            "Routing.md": "atlas routing tables converge on the shortest path",
            "Unrelated.md": "quarterly budget forecast approval",
        ])

        let response = try await workspace.search(.init(query: "atlas routing", limit: 10))

        #expect(response.ordering == .similarity)
        let similarities = response.results.compactMap { $0.result.similarity }
        #expect(similarities.count == response.results.count)
        #expect(similarities == similarities.sorted(by: >), "merged order must follow the absolute signal")
    }

    /// Two indexes both contributing a rank-1 hit is exactly where fusion scores stop being
    /// comparable, so the merged order must not be reported as similarity-ordered unless it is.
    @Test("the response states which quantity it ordered on")
    func orderingIsDeclaredNotAssumed() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)

        _ = try await seed(workspace, indexNamed: "Personal", root: root, files: [
            "Atlas.md": "atlas routing marker",
        ])
        _ = try await seed(workspace, indexNamed: "Work", root: root, files: [
            "Atlas.md": "atlas routing marker",
        ])

        let response = try await workspace.search(.init(query: "atlas routing", limit: 10))

        #expect(response.results.count == 2)
        // Whichever branch was taken, the claim has to match the data it was derived from.
        switch response.ordering {
        case .similarity:
            #expect(response.results.allSatisfy { $0.result.similarity != nil })
            #expect(Set(response.results.compactMap { $0.result.provenance.embeddingSpaceID }).count == 1)
        case .fusionRank:
            let missingSimilarity = response.results.contains { $0.result.similarity == nil }
            let mixedSpaces = Set(response.results.compactMap { $0.result.provenance.embeddingSpaceID }).count > 1
            #expect(missingSimilarity || mixedSpaces)
        }
    }

    @Test("an empty workspace search reports fusion ordering rather than claiming similarity")
    func emptyMergeDoesNotClaimSimilarity() async throws {
        let root = try makeRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)
        _ = try await workspace.createIndex(named: "Empty")

        let response = try await workspace.search(.init(query: "anything", limit: 10))

        #expect(response.results.isEmpty)
        #expect(response.ordering == .fusionRank)
    }
}

private final class RelevanceTestCursorStore: CursorStore, @unchecked Sendable {
    private let lock = NSLock()
    private var cursors: [String: SourceCursor] = [:]

    func cursor(forKey key: String) -> SourceCursor? {
        lock.withLock { cursors[key] }
    }

    func setCursor(_ cursor: SourceCursor, forKey key: String) {
        lock.withLock { cursors[key] = cursor }
    }
}
