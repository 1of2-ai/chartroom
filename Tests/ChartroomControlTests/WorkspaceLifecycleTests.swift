import ChartroomControl
import ConnectorEngine
import Foundation
import IndexEngine
import SyncEngine
import Testing

/// H4: indexes could be created and toggled, never renamed, deleted, or rebuilt. Deletion is the
/// dangerous one — an adopted store lives beside the catalog rather than under the workspace's own
/// stores directory, so removing "the store's folder" would take the whole workspace with it.
@Suite("Workspace lifecycle")
struct WorkspaceLifecycleTests {
    private func temporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "chartroom-lifecycle-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func makeWorkspace(root: URL, bootstrapIndexes: [ChartroomIndex] = []) -> ChartroomWorkspace {
        ChartroomWorkspace(
            catalogURL: root.appending(path: "Indexes.json"),
            storesDirectory: root.appending(path: "Indexes", directoryHint: .isDirectory),
            bootstrapIndexes: bootstrapIndexes,
            engineFactory: { _ in (try await IndexEngine.openInMemory(), nil) },
            cursorStore: LifecycleTestCursorStore()
        )
    }

    @Test("deleting an index removes it from the catalog and deletes its store directory")
    func deleteRemovesCatalogEntryAndStore() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)

        let index = try await workspace.createIndex(named: "Field Notes")
        let directory = index.storeURL.deletingLastPathComponent()
        FileManager.default.createFile(atPath: index.storeURL.path, contents: Data("x".utf8))
        #expect(FileManager.default.fileExists(atPath: directory.path))

        try await workspace.deleteIndex(index.id)

        #expect(try await workspace.indexes().isEmpty)
        #expect(FileManager.default.fileExists(atPath: directory.path) == false)

        // The catalog on disk agrees, not just the in-memory copy.
        let reloaded = makeWorkspace(root: root)
        #expect(try await reloaded.indexes().isEmpty)
    }

    /// The guard that matters: an adopted store sits next to `Indexes.json`, so its parent is the
    /// workspace root. Deleting that parent would destroy the catalog and every other index.
    @Test("deleting an adopted index never removes the directory it shares with the catalog")
    func deleteSparesAdoptedStoreParent() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let legacyStore = root.appending(path: "IndexEngine.sqlite")
        let legacy = ChartroomIndex(name: "Legacy", storeURL: legacyStore)
        let workspace = makeWorkspace(root: root, bootstrapIndexes: [legacy])
        _ = try await workspace.indexes()

        FileManager.default.createFile(atPath: legacyStore.path, contents: Data("x".utf8))
        let survivor = try await workspace.createIndex(named: "Keeper")

        try await workspace.deleteIndex(legacy.id)

        #expect(FileManager.default.fileExists(atPath: legacyStore.path) == false, "the store itself goes")
        #expect(FileManager.default.fileExists(atPath: root.path), "the workspace root must survive")
        #expect(
            FileManager.default.fileExists(atPath: survivor.storeURL.deletingLastPathComponent().path),
            "an unrelated index must survive"
        )
        #expect(try await workspace.indexes().map(\.id) == [survivor.id])
    }

    @Test("deleting an index also removes its SQLite sidecars")
    func deleteRemovesSidecars() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let legacyStore = root.appending(path: "IndexEngine.sqlite")
        let legacy = ChartroomIndex(name: "Legacy", storeURL: legacyStore)
        let workspace = makeWorkspace(root: root, bootstrapIndexes: [legacy])
        _ = try await workspace.indexes()
        for suffix in ["", "-wal", "-shm"] {
            FileManager.default.createFile(atPath: legacyStore.path + suffix, contents: Data("x".utf8))
        }

        try await workspace.deleteIndex(legacy.id)

        for suffix in ["", "-wal", "-shm"] {
            #expect(FileManager.default.fileExists(atPath: legacyStore.path + suffix) == false)
        }
    }

    @Test("deleting an unknown index reports rather than silently succeeding")
    func deleteUnknownIndexThrows() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)

        await #expect(throws: (any Error).self) {
            try await workspace.deleteIndex(UUID())
        }
    }

    @Test("renaming rejects a name already in use and keeps the original")
    func renameRejectsDuplicates() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)

        let first = try await workspace.createIndex(named: "Alpha")
        _ = try await workspace.createIndex(named: "Beta")

        await #expect(throws: (any Error).self) {
            try await workspace.renameIndex(first.id, to: "Beta")
        }
        #expect(try await workspace.indexes().first { $0.id == first.id }?.name == "Alpha")
    }

    @Test("rebuilding an index reports what it restored")
    func rebuildReportsSummary() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)

        let index = try await workspace.createIndex(named: "Rebuildable")

        // An in-memory engine starts empty, so there is nothing orphaned to restore. The contract
        // under test is that the call reaches the engine and reports honestly rather than throwing.
        let summary = try await workspace.rebuildIndex(index.id)
        #expect(summary.rebuiltChunkCount == 0)
        #expect(summary.isComplete)
    }
}

private final class LifecycleTestCursorStore: CursorStore, @unchecked Sendable {
    private var storage: [String: SourceCursor] = [:]

    func cursor(forKey key: String) -> SourceCursor? { storage[key] }
    func setCursor(_ cursor: SourceCursor, forKey key: String) { storage[key] = cursor }
}
