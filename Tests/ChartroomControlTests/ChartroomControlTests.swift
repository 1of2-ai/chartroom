import ConnectorEngine
import Foundation
import IndexEngine
import SyncEngine
import Testing
@testable import ChartroomControl

@Suite("Chartroom session")
struct ChartroomSessionTests {
    @Test("opens an engine and reports status through the product command surface")
    func opensAndReportsStatus() async throws {
        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: InMemoryCursorStore()
        )

        let status = try await session.open()

        #expect(status.state == .ready)
        #expect(status.snapshot?.documentCount == 0)
        #expect(status.health?.documentCount == 0)
        #expect(status.modelStatus?.isAvailable == true)
    }

    @Test("ingests a local source, searches it, browses it, inspects chunks, and deletes it")
    func localSourceFeatureParityCommands() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "chartroom-session-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let noteURL = root.appending(path: "Needle.md")
        try "headless mcp parity search needle".write(to: noteURL, atomically: true, encoding: .utf8)

        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: InMemoryCursorStore()
        )
        _ = try await session.open()

        let sync = try await session.ingestLocalSource(root)
        #expect(sync.accepted == 1)
        #expect(sync.totalFailed == 0)

        let search = try await session.search(.init(query: "parity needle", mode: .diagnostic, limit: 5))
        let result = try #require(search.results.first)
        #expect(result.documentID == "local-files:Needle.md")

        let browse = try await session.browseDocuments(.init(limit: 10))
        #expect(browse.documents.map(\.id) == ["local-files:Needle.md"])

        let chunks = try await session.chunks(forDocument: result.documentID)
        #expect(chunks.count == 1)
        #expect(chunks.first?.text.contains("headless mcp parity") == true)

        let deletion = try await session.deleteDocuments([result.documentID])
        #expect(deletion.deletedCount == 1)

        let empty = try await session.search(.init(query: "parity needle", mode: .diagnostic, limit: 5))
        #expect(empty.results.isEmpty)
    }

    @Test("workspace persists index inclusion and keeps connector cursors separate per index")
    func workspacePersistsInclusionAndNamespacesCursors() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "chartroom-workspace-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let source = root.appending(path: "Source", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try "cursor isolation marker".write(
            to: source.appending(path: "Marker.md"),
            atomically: true,
            encoding: .utf8
        )

        let cursors = InMemoryCursorStore()
        let workspace = ChartroomWorkspace(
            catalogURL: root.appending(path: "Indexes.json"),
            storesDirectory: root.appending(path: "Indexes", directoryHint: .isDirectory),
            engineFactory: { _ in (try await IndexEngine.openInMemory(), nil) },
            cursorStore: cursors
        )

        let personal = try await workspace.createIndex(named: "Personal")
        let work = try await workspace.createIndex(named: "Work")
        try await workspace.setSearchEnabled(false, for: work.id)

        let personalSession = try await workspace.session(for: personal.id)
        let workSession = try await workspace.session(for: work.id)
        _ = try await personalSession.open()
        _ = try await workSession.open()

        #expect(try await personalSession.ingestLocalSource(source).accepted == 1)
        #expect(try await workSession.ingestLocalSource(source).accepted == 1)

        let reopened = ChartroomWorkspace(
            catalogURL: root.appending(path: "Indexes.json"),
            storesDirectory: root.appending(path: "Indexes", directoryHint: .isDirectory),
            engineFactory: { _ in (try await IndexEngine.openInMemory(), nil) },
            cursorStore: cursors
        )
        let indexes = try await reopened.indexes()

        #expect(indexes.map(\.name) == ["Personal", "Work"])
        #expect(indexes.first { $0.id == personal.id }?.isSearchEnabled == true)
        #expect(indexes.first { $0.id == work.id }?.isSearchEnabled == false)
    }

    @Test("workspace search scopes result identity and continues after an index failure")
    func workspaceSearchScopesResultsAndContinuesAfterIndexFailure() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "chartroom-workspace-search-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let workspace = ChartroomWorkspace(
            catalogURL: root.appending(path: "Indexes.json"),
            storesDirectory: root.appending(path: "Indexes", directoryHint: .isDirectory),
            engineFactory: { index in
                if index.name == "Unavailable" {
                    throw IndexEngineError(
                        .storageUnavailable,
                        code: "test.index.unavailable",
                        recoverability: .retryable,
                        summary: "Unavailable for test"
                    )
                }
                return (try await IndexEngine.openInMemory(), nil)
            },
            cursorStore: InMemoryCursorStore()
        )

        let personal = try await workspace.createIndex(named: "Personal")
        let work = try await workspace.createIndex(named: "Work")
        _ = try await workspace.createIndex(named: "Unavailable")

        let personalSource = root.appending(path: "Personal", directoryHint: .isDirectory)
        let workSource = root.appending(path: "Work", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: personalSource, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: workSource, withIntermediateDirectories: true)
        try "atlas retrieval marker".write(
            to: personalSource.appending(path: "Personal.md"),
            atomically: true,
            encoding: .utf8
        )
        try "atlas retrieval marker".write(
            to: workSource.appending(path: "Work.md"),
            atomically: true,
            encoding: .utf8
        )

        let personalSession = try await workspace.session(for: personal.id)
        let workSession = try await workspace.session(for: work.id)
        _ = try await personalSession.open()
        _ = try await workSession.open()

        #expect(try await personalSession.ingestLocalSource(personalSource).accepted == 1)
        #expect(try await workSession.ingestLocalSource(workSource).accepted == 1)

        let response = try await workspace.search(.init(query: "atlas retrieval", limit: 10))

        #expect(response.results.count == 2)
        #expect(Set(response.results.map(\.id)).count == 2)
        #expect(Set(response.results.map(\.index.id)) == Set([personal.id, work.id]))
        #expect(response.statuses.first { $0.index.name == "Unavailable" }?.errorMessage == "Unavailable for test")
    }
}

private final class InMemoryCursorStore: CursorStore, @unchecked Sendable {
    private let lock = NSLock()
    private var cursors: [String: SourceCursor] = [:]

    func cursor(forKey key: String) -> SourceCursor? {
        lock.withLock { cursors[key] }
    }

    func setCursor(_ cursor: SourceCursor, forKey key: String) {
        lock.withLock {
            cursors[key] = cursor
        }
    }
}
