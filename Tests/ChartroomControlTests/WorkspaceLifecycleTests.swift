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

    private func makeWorkspace(
        root: URL,
        bootstrapIndexes: [ChartroomIndex] = [],
        engineFactory: @escaping ChartroomWorkspace.EngineFactory = { _ in
            (try await IndexEngine.openInMemory(), nil)
        }
    ) -> ChartroomWorkspace {
        ChartroomWorkspace(
            catalogURL: root.appending(path: "Indexes.json"),
            storesDirectory: root.appending(path: "Indexes", directoryHint: .isDirectory),
            bootstrapIndexes: bootstrapIndexes,
            engineFactory: engineFactory,
            cursorStore: LifecycleTestCursorStore()
        )
    }

    private func waitUntilCatalogIsEmpty(_ workspace: ChartroomWorkspace) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(2))
        while clock.now < deadline {
            if try await workspace.indexes().isEmpty {
                return
            }
            try await Task.sleep(for: .milliseconds(5))
        }
        throw LifecycleTestError.timedOutWaitingForCatalogRemoval
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

    @Test("deleting a managed index refuses a replaced directory symlink without traversing it")
    func deleteManagedDirectorySymlinkDoesNotEscape() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)
        let index = try await workspace.createIndex(named: "Redirected")
        let managedDirectory = index.storeURL.deletingLastPathComponent()

        try FileManager.default.removeItem(at: managedDirectory)
        let externalDirectory = root.appending(path: "External", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: externalDirectory, withIntermediateDirectories: true)
        let externalStore = externalDirectory.appending(path: index.storeURL.lastPathComponent)
        let externalContents = Data("must survive managed cleanup".utf8)
        try externalContents.write(to: externalStore)
        try FileManager.default.createSymbolicLink(
            at: managedDirectory,
            withDestinationURL: externalDirectory
        )

        do {
            _ = try await workspace.deleteIndex(index.id)
            Issue.record("Expected cleanup to refuse a replacement managed-directory symlink")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.workspace.store-delete-failed")
        }

        #expect(try await workspace.indexes().isEmpty)
        let reloaded = makeWorkspace(root: root)
        #expect(try await reloaded.indexes().isEmpty, "Catalog-first deletion must remain durable")
        #expect(
            try FileManager.default.attributesOfItem(atPath: managedDirectory.path)[.type]
                as? FileAttributeType == .typeSymbolicLink,
            "Cleanup must leave a replacement entry untouched"
        )
        let externalStoreSurvived = FileManager.default.fileExists(atPath: externalStore.path)
        #expect(externalStoreSurvived, "Cleanup must not traverse the managed directory symlink")
        if externalStoreSurvived {
            #expect(try Data(contentsOf: externalStore) == externalContents)
        }
    }

    @Test("deleting a managed index refuses an ordinary replacement directory")
    func deleteManagedDirectoryReplacementDoesNotDeleteIt() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)
        let index = try await workspace.createIndex(named: "Replaced Leaf")
        let managedDirectory = index.storeURL.deletingLastPathComponent()
        try Data("original store".utf8).write(to: index.storeURL)

        let displacedDirectory = managedDirectory
            .deletingLastPathComponent()
            .appending(path: "Displaced-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.moveItem(at: managedDirectory, to: displacedDirectory)
        try FileManager.default.createDirectory(at: managedDirectory, withIntermediateDirectories: false)
        let replacementFile = managedDirectory.appending(path: "must-survive.txt")
        let replacementContents = Data("replacement directory is not the index".utf8)
        try replacementContents.write(to: replacementFile)

        do {
            _ = try await workspace.deleteIndex(index.id)
            Issue.record("Expected cleanup to refuse a replacement managed directory")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.workspace.store-delete-failed")
        }

        #expect(try Data(contentsOf: replacementFile) == replacementContents)
        #expect(
            FileManager.default.fileExists(atPath: displacedDirectory.path),
            "Refusing the replacement must also leave the originally bound directory visible"
        )
    }

    @Test("deleting a managed index stays bound when the stores directory is replaced")
    func deleteManagedIndexUsesBoundStoresDirectory() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)
        let index = try await workspace.createIndex(named: "Bound Managed")
        let managedDirectory = index.storeURL.deletingLastPathComponent()
        let storesDirectory = managedDirectory.deletingLastPathComponent()
        try Data("original managed store".utf8).write(to: index.storeURL)

        let boundStoresDirectory = root.appending(path: "BoundIndexes", directoryHint: .isDirectory)
        try FileManager.default.moveItem(at: storesDirectory, to: boundStoresDirectory)
        let boundManagedDirectory = boundStoresDirectory
            .appending(path: managedDirectory.lastPathComponent, directoryHint: .isDirectory)

        let redirectedStoresDirectory = root
            .appending(path: "RedirectedIndexes", directoryHint: .isDirectory)
        let redirectedManagedDirectory = redirectedStoresDirectory
            .appending(path: managedDirectory.lastPathComponent, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: redirectedManagedDirectory,
            withIntermediateDirectories: true
        )
        let redirectedStore = redirectedManagedDirectory.appending(path: index.storeURL.lastPathComponent)
        let redirectedContents = Data("redirected managed store must survive".utf8)
        try redirectedContents.write(to: redirectedStore)
        try FileManager.default.createSymbolicLink(
            at: storesDirectory,
            withDestinationURL: redirectedStoresDirectory
        )

        _ = try await workspace.deleteIndex(index.id)

        #expect(
            !FileManager.default.fileExists(atPath: boundManagedDirectory.path),
            "Cleanup must remove the directory from the stores root captured when the index was created"
        )
        let redirectedStoreSurvived = FileManager.default.fileExists(atPath: redirectedStore.path)
        #expect(redirectedStoreSurvived, "Cleanup must not follow a replacement stores-directory symlink")
        if redirectedStoreSurvived {
            #expect(try Data(contentsOf: redirectedStore) == redirectedContents)
        }
    }

    @Test("opening after a stores-root replacement stabilizes before rebinding cleanup")
    func lazyOpenRebindsManagedDeletionTarget() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(
            root: root,
            engineFactory: { index in
                try FileManager.default.createDirectory(
                    at: index.storeURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try Data("store opened at the catalog path".utf8).write(to: index.storeURL)
                return (try await IndexEngine.openInMemory(), index.storeURL)
            }
        )
        let index = try await workspace.createIndex(named: "Lazy Rebind")
        let originalStoresDirectory = index.storeURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let boundStoresDirectory = root.appending(path: "InitiallyBound", directoryHint: .isDirectory)
        try FileManager.default.moveItem(
            at: originalStoresDirectory,
            to: boundStoresDirectory
        )

        let session = try await workspace.session(for: index.id)
        do {
            _ = try await session.open()
            Issue.record("Expected the first open to refuse an unbound replacement path")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.workspace.store-binding-changed")
        }

        _ = try await session.open()
        #expect(FileManager.default.fileExists(atPath: index.storeURL.path))

        _ = try await workspace.deleteIndex(index.id)

        #expect(
            !FileManager.default.fileExists(
                atPath: index.storeURL.deletingLastPathComponent().path
            ),
            "Deletion must remove the store path the host factory actually opened"
        )
        #expect(
            FileManager.default.fileExists(atPath: boundStoresDirectory.path),
            "Rebinding an opened replacement must not delete the historical stores root"
        )
    }

    @Test("opening refuses a managed store replaced while its engine factory runs")
    func lazyOpenRejectsManagedStoreReplacementDuringFactory() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let displacedDirectory = root
            .appending(path: "DisplacedDuringOpen", directoryHint: .isDirectory)
        let replacementContents = Data("replacement store must survive".utf8)
        let workspace = makeWorkspace(
            root: root,
            engineFactory: { index in
                let managedDirectory = index.storeURL.deletingLastPathComponent()
                try FileManager.default.moveItem(
                    at: managedDirectory,
                    to: displacedDirectory
                )
                try FileManager.default.createDirectory(
                    at: managedDirectory,
                    withIntermediateDirectories: false
                )
                try replacementContents.write(to: index.storeURL)
                return (try await IndexEngine.openInMemory(), index.storeURL)
            }
        )
        let index = try await workspace.createIndex(named: "Raced Open")
        let session = try await workspace.session(for: index.id)

        do {
            _ = try await session.open()
            Issue.record("Expected open to refuse a store identity changed by its factory")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.workspace.store-binding-changed")
        }

        do {
            _ = try await workspace.deleteIndex(index.id)
            Issue.record("Expected cleanup to refuse the replacement directory")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.workspace.store-delete-failed")
        }

        #expect(try Data(contentsOf: index.storeURL) == replacementContents)
        #expect(FileManager.default.fileExists(atPath: displacedDirectory.path))
    }

    @Test("opening refuses a managed store symlink present before its factory runs")
    func lazyOpenRejectsManagedStoreSymlink() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(
            root: root,
            engineFactory: { index in
                (try await IndexEngine.openInMemory(), index.storeURL)
            }
        )
        let index = try await workspace.createIndex(named: "Linked Open")
        let managedDirectory = index.storeURL.deletingLastPathComponent()
        let displacedDirectory = root
            .appending(path: "DisplacedLinkedOpen", directoryHint: .isDirectory)
        try FileManager.default.moveItem(at: managedDirectory, to: displacedDirectory)

        let externalDirectory = root
            .appending(path: "ExternalLinkedOpen", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: externalDirectory,
            withIntermediateDirectories: true
        )
        let externalStore = externalDirectory.appending(path: index.storeURL.lastPathComponent)
        let externalContents = Data("external store must survive".utf8)
        try externalContents.write(to: externalStore)
        try FileManager.default.createSymbolicLink(
            at: managedDirectory,
            withDestinationURL: externalDirectory
        )

        let session = try await workspace.session(for: index.id)
        do {
            _ = try await session.open()
            Issue.record("Expected open to refuse a managed-store symlink")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.workspace.store-binding-changed")
        }

        do {
            _ = try await workspace.deleteIndex(index.id)
            Issue.record("Expected cleanup to retain the unverified symlink")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.workspace.store-delete-failed")
        }

        #expect(try Data(contentsOf: externalStore) == externalContents)
        #expect(
            try FileManager.default.attributesOfItem(atPath: managedDirectory.path)[.type]
                as? FileAttributeType == .typeSymbolicLink
        )
    }

    @Test("creating after a stores-root replacement binds the new root")
    func createAfterStoresRootReplacementUsesCurrentDirectory() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)
        let first = try await workspace.createIndex(named: "First")
        let storesDirectory = first.storeURL
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let historicalStoresDirectory = root
            .appending(path: "HistoricalIndexes", directoryHint: .isDirectory)
        try FileManager.default.moveItem(
            at: storesDirectory,
            to: historicalStoresDirectory
        )
        try FileManager.default.createDirectory(
            at: storesDirectory,
            withIntermediateDirectories: false
        )

        let second = try await workspace.createIndex(named: "Second")
        let secondDirectory = second.storeURL.deletingLastPathComponent()
        #expect(FileManager.default.fileExists(atPath: secondDirectory.path))

        _ = try await workspace.deleteIndex(second.id)

        #expect(
            !FileManager.default.fileExists(atPath: secondDirectory.path),
            "The deletion binding must follow the current stores-directory identity"
        )
        #expect(
            FileManager.default.fileExists(atPath: historicalStoresDirectory.path)
        )
    }

    /// The guard that matters: an adopted store sits next to `Indexes.json`, so its parent is the
    /// workspace root. Deleting that parent would destroy the catalog and every other index.
    @Test("deleting an adopted index never removes the directory it shares with the catalog")
    func deleteSparesAdoptedStoreParent() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let legacyStore = root.appending(path: "IndexEngine.sqlite")
        let legacy = ChartroomIndex(name: "Legacy", storeURL: legacyStore)
        FileManager.default.createFile(atPath: legacyStore.path, contents: Data("x".utf8))
        let workspace = makeWorkspace(root: root, bootstrapIndexes: [legacy])
        _ = try await workspace.indexes()

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
        for suffix in ["", "-wal", "-shm"] {
            FileManager.default.createFile(atPath: legacyStore.path + suffix, contents: Data("x".utf8))
        }
        let workspace = makeWorkspace(root: root, bootstrapIndexes: [legacy])
        _ = try await workspace.indexes()

        try await workspace.deleteIndex(legacy.id)

        for suffix in ["", "-wal", "-shm"] {
            #expect(FileManager.default.fileExists(atPath: legacyStore.path + suffix) == false)
        }
    }

    @Test("deleting an adopted index refuses a replacement store file")
    func deleteAdoptedStoreReplacementDoesNotDeleteIt() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let adoptedStore = root.appending(path: "IndexEngine.sqlite")
        try Data("original adopted store".utf8).write(to: adoptedStore)
        let adopted = ChartroomIndex(name: "Adopted Replacement", storeURL: adoptedStore)
        let workspace = makeWorkspace(root: root, bootstrapIndexes: [adopted])
        _ = try await workspace.indexes()

        let displacedStore = root.appending(path: "Displaced.sqlite")
        try FileManager.default.moveItem(at: adoptedStore, to: displacedStore)
        let replacementContents = Data("replacement store must survive".utf8)
        try replacementContents.write(to: adoptedStore)
        try Data("replacement wal must survive".utf8)
            .write(to: URL(filePath: adoptedStore.path + "-wal"))
        try Data("replacement shm must survive".utf8)
            .write(to: URL(filePath: adoptedStore.path + "-shm"))

        do {
            _ = try await workspace.deleteIndex(adopted.id)
            Issue.record("Expected cleanup to refuse a replacement adopted store")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.workspace.store-delete-failed")
        }

        #expect(try Data(contentsOf: adoptedStore) == replacementContents)
        #expect(FileManager.default.fileExists(atPath: adoptedStore.path + "-wal"))
        #expect(FileManager.default.fileExists(atPath: adoptedStore.path + "-shm"))
        #expect(
            FileManager.default.fileExists(atPath: displacedStore.path),
            "The originally bound store remains visible after its catalog path is replaced"
        )
    }

    @Test("deleting an adopted index stays bound when its parent is replaced")
    func deleteAdoptedIndexUsesBoundParentDirectory() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let adoptedParent = root.appending(path: "Adopted", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: adoptedParent, withIntermediateDirectories: true)
        let adoptedStore = adoptedParent.appending(path: "IndexEngine.sqlite")
        for suffix in ["", "-wal", "-shm"] {
            try Data("original adopted \(suffix)".utf8).write(
                to: suffix.isEmpty
                    ? adoptedStore
                    : adoptedParent.appending(path: adoptedStore.lastPathComponent + suffix)
            )
        }

        let adopted = ChartroomIndex(name: "Bound Adopted", storeURL: adoptedStore)
        let workspace = makeWorkspace(root: root, bootstrapIndexes: [adopted])
        _ = try await workspace.indexes()

        let boundParent = root.appending(path: "BoundAdopted", directoryHint: .isDirectory)
        try FileManager.default.moveItem(at: adoptedParent, to: boundParent)
        let redirectedParent = root.appending(path: "RedirectedAdopted", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: redirectedParent, withIntermediateDirectories: true)
        let redirectedStore = redirectedParent.appending(path: adoptedStore.lastPathComponent)
        let redirectedContents = Data("redirected adopted store must survive".utf8)
        try redirectedContents.write(to: redirectedStore)
        try Data("redirected wal must survive".utf8)
            .write(to: redirectedParent.appending(path: adoptedStore.lastPathComponent + "-wal"))
        try Data("redirected shm must survive".utf8)
            .write(to: redirectedParent.appending(path: adoptedStore.lastPathComponent + "-shm"))
        try FileManager.default.createSymbolicLink(
            at: adoptedParent,
            withDestinationURL: redirectedParent
        )

        _ = try await workspace.deleteIndex(adopted.id)

        for suffix in ["", "-wal", "-shm"] {
            let boundFile = suffix.isEmpty
                ? boundParent.appending(path: adoptedStore.lastPathComponent)
                : boundParent.appending(path: adoptedStore.lastPathComponent + suffix)
            #expect(
                !FileManager.default.fileExists(atPath: boundFile.path),
                "Cleanup must remove the originally bound adopted store and sidecars"
            )
        }
        let redirectedStoreSurvived = FileManager.default.fileExists(atPath: redirectedStore.path)
        #expect(redirectedStoreSurvived, "Cleanup must not follow a replacement adopted-parent symlink")
        if redirectedStoreSurvived {
            #expect(try Data(contentsOf: redirectedStore) == redirectedContents)
        }
    }

    @Test("deleting an adopted index reports an unavailable bound parent")
    func deleteAdoptedIndexReportsUnavailableParent() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let adoptedParent = root.appending(path: "Mounted", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: adoptedParent,
            withIntermediateDirectories: true
        )
        let adoptedStore = adoptedParent.appending(path: "IndexEngine.sqlite")
        try Data("bound store".utf8).write(to: adoptedStore)
        let adopted = ChartroomIndex(name: "Unavailable Parent", storeURL: adoptedStore)

        let initialWorkspace = makeWorkspace(root: root, bootstrapIndexes: [adopted])
        _ = try await initialWorkspace.indexes()

        let displacedParent = root.appending(path: "Unmounted", directoryHint: .isDirectory)
        try FileManager.default.moveItem(at: adoptedParent, to: displacedParent)
        let reloadedWorkspace = makeWorkspace(root: root)
        _ = try await reloadedWorkspace.indexes()

        do {
            _ = try await reloadedWorkspace.deleteIndex(adopted.id)
            Issue.record("Expected cleanup to report the unavailable adopted-store parent")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.workspace.store-delete-failed")
        }

        #expect(try await reloadedWorkspace.indexes().isEmpty)
        #expect(FileManager.default.fileExists(
            atPath: displacedParent.appending(path: adoptedStore.lastPathComponent).path
        ))
    }

    @Test("a nil factory store URL cannot bypass adopted-store identity validation")
    func nilFactoryStoreURLStillValidatesAdoptedStore() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let adoptedStore = root.appending(path: "IndexEngine.sqlite")
        let adopted = ChartroomIndex(name: "Nil Store URL", storeURL: adoptedStore)
        let workspace = makeWorkspace(
            root: root,
            bootstrapIndexes: [adopted],
            engineFactory: { index in
                if !FileManager.default.fileExists(atPath: index.storeURL.path) {
                    try Data("factory-created store".utf8).write(to: index.storeURL)
                }
                return (try await IndexEngine.openInMemory(), nil)
            }
        )
        _ = try await workspace.indexes()
        let session = try await workspace.session(for: adopted.id)

        do {
            _ = try await session.open()
            Issue.record("Expected the first open to reject a newly materialized store identity")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.workspace.store-binding-changed")
        }

        #expect(try await session.open().state == .ready)
        _ = try await workspace.deleteIndex(adopted.id)
        #expect(!FileManager.default.fileExists(atPath: adoptedStore.path))
    }

    @Test("deleting an index permanently invalidates a retained session")
    func deleteInvalidatesRetainedSession() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)
        let index = try await workspace.createIndex(named: "Retained")
        let session = try await workspace.session(for: index.id)
        _ = try await session.open()

        try await workspace.deleteIndex(index.id)

        do {
            _ = try await session.open()
            Issue.record("Expected a retained session to reject reopening after its index was deleted")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.session.invalidated")
        }

        do {
            _ = try await session.search(.init(query: "deleted index", limit: 1))
            Issue.record("Expected a retained session to reject commands after its index was deleted")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.session.invalidated")
        }
    }

    @Test("closing while an open is suspended prevents its stale engine from becoming ready")
    func closeWinsOverSuspendedOpen() async throws {
        let factoryGate = LifecycleGate()
        let session = ChartroomSession(
            engineFactory: {
                await factoryGate.suspend()
                return (try await IndexEngine.openInMemory(), nil)
            },
            cursorStore: LifecycleTestCursorStore()
        )

        let openTask = Task { try await session.open() }
        try await factoryGate.waitUntilEntered()
        await session.close()
        #expect(await session.status().state == .unopened)

        await factoryGate.release()
        do {
            _ = try await openTask.value
            Issue.record("Expected close() to supersede the suspended open")
        } catch {
            // A close deliberately supersedes this open attempt.
        }

        #expect(await session.status().state == .unopened)
        #expect(await session.status().snapshot == nil)

        #expect(try await session.open().state == .ready, "A fresh open after close must still work")
    }

    @Test("concurrent opens share one engine-factory invocation")
    func concurrentOpensAreSingleFlight() async throws {
        let factoryGate = LifecycleGate()
        let session = ChartroomSession(
            engineFactory: {
                await factoryGate.suspend()
                return (try await IndexEngine.openInMemory(), nil)
            },
            cursorStore: LifecycleTestCursorStore()
        )

        let firstOpen = Task { try await session.open() }
        try await factoryGate.waitUntilEntered()
        let secondOpen = Task { try await session.open() }
        let releaseTask = Task {
            let clock = ContinuousClock()
            let deadline = clock.now.advanced(by: .milliseconds(100))
            while await factoryGate.entryCount() < 2, clock.now < deadline {
                await Task.yield()
            }
            await factoryGate.release()
        }

        #expect(try await firstOpen.value.state == .ready)
        #expect(try await secondOpen.value.state == .ready)
        await releaseTask.value
        #expect(await factoryGate.entryCount() == 1)
    }

    @Test("deletion leaves the store in place until an opening factory has quiesced")
    func deleteWaitsForOpeningFactory() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let factoryGate = LifecycleGate()
        let workspace = makeWorkspace(
            root: root,
            engineFactory: { index in
                await factoryGate.suspend()
                return (try await IndexEngine.openInMemory(), index.storeURL)
            }
        )
        let index = try await workspace.createIndex(named: "Opening")
        FileManager.default.createFile(atPath: index.storeURL.path, contents: Data("store".utf8))
        let session = try await workspace.session(for: index.id)

        let openTask = Task { try await session.open() }
        try await factoryGate.waitUntilEntered()
        let deleteTask = Task { try await workspace.deleteIndex(index.id) }
        try await waitUntilCatalogIsEmpty(workspace)

        #expect(
            FileManager.default.fileExists(atPath: index.storeURL.path),
            "Catalog removal may lead, but store removal must wait for the opening factory"
        )

        await factoryGate.release()
        _ = try? await openTask.value
        _ = try await deleteTask.value
        #expect(!FileManager.default.fileExists(atPath: index.storeURL.path))
    }

    @Test("deletion leaves the store in place until an already-started search has quiesced")
    func deleteWaitsForStartedSearch() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let queryGate = LifecycleGate()
        let engine = LifecycleGateEngine(searchGate: queryGate)
        let workspace = makeWorkspace(
            root: root,
            engineFactory: { index in
                return (engine, index.storeURL)
            }
        )
        let index = try await workspace.createIndex(named: "Busy")
        FileManager.default.createFile(atPath: index.storeURL.path, contents: Data("store".utf8))
        let session = try await workspace.session(for: index.id)
        _ = try await session.open()

        let searchTask = Task {
            try await session.search(.init(query: "blocked query", limit: 1))
        }
        try await queryGate.waitUntilEntered()

        let deleteTask = Task { try await workspace.deleteIndex(index.id) }
        try await waitUntilCatalogIsEmpty(workspace)
        #expect(FileManager.default.fileExists(atPath: index.storeURL.path))

        await queryGate.release()
        _ = try await searchTask.value
        _ = try await deleteTask.value
        #expect(!FileManager.default.fileExists(atPath: index.storeURL.path))
    }

    @Test("deletion leaves the store in place until an already-started sync has quiesced")
    func deleteWaitsForStartedSync() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let documentGate = LifecycleGate()
        let engine = LifecycleGateEngine(ingestGate: documentGate)
        let workspace = makeWorkspace(
            root: root,
            engineFactory: { index in
                return (engine, index.storeURL)
            }
        )
        let index = try await workspace.createIndex(named: "Syncing")
        FileManager.default.createFile(atPath: index.storeURL.path, contents: Data("store".utf8))
        let session = try await workspace.session(for: index.id)
        _ = try await session.open()

        let syncTask = Task {
            try await session.sync(
                connector: LifecycleSourceConnector(),
                cursorKey: "busy-source"
            )
        }
        try await documentGate.waitUntilEntered()

        let deleteTask = Task { try await workspace.deleteIndex(index.id) }
        try await waitUntilCatalogIsEmpty(workspace)
        #expect(FileManager.default.fileExists(atPath: index.storeURL.path))

        await documentGate.release()
        #expect(try await syncTask.value.accepted == 1)
        _ = try await deleteTask.value
        #expect(!FileManager.default.fileExists(atPath: index.storeURL.path))
    }

    @Test("deleting from a progress callback fails instead of waiting on its own sync")
    func deleteFromProgressCallbackDoesNotDeadlock() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = makeWorkspace(root: root)
        let index = try await workspace.createIndex(named: "Reentrant")
        let session = try await workspace.session(for: index.id)
        _ = try await session.open()
        let probe = ReentrantDeletionProbe()

        let syncTask = Task {
            try await session.sync(
                connector: LifecycleSourceConnector(),
                cursorKey: "reentrant-delete",
                onProgress: { _ in
                    guard await probe.claimAttempt() else { return }
                    do {
                        _ = try await workspace.deleteIndex(index.id)
                        await probe.record(code: "unexpected-success")
                    } catch {
                        await probe.record(
                            code: (error as? IndexEngineError)?.code
                                ?? String(describing: error)
                        )
                    }
                }
            )
        }

        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(2))
        while await probe.resultCode() == nil, clock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        guard let code = await probe.resultCode() else {
            syncTask.cancel()
            Issue.record("Deletion waited on the sync operation that invoked it")
            return
        }

        #expect(code == "chartroom.workspace.delete-from-active-operation")
        #expect(try await syncTask.value.accepted == 1)
        #expect(try await workspace.indexes().map(\.id) == [index.id])
    }

    @Test("store cleanup failure is typed after the catalog entry is removed")
    func deleteReportsStoreCleanupFailure() async throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let protectedDirectory = root.appending(path: "Protected", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: protectedDirectory, withIntermediateDirectories: true)
        let storeURL = protectedDirectory.appending(path: "IndexEngine.sqlite")
        try Data("store".utf8).write(to: storeURL)
        let index = ChartroomIndex(name: "Protected", storeURL: storeURL)
        let workspace = makeWorkspace(root: root, bootstrapIndexes: [index])
        _ = try await workspace.indexes()

        try FileManager.default.setAttributes(
            [.posixPermissions: NSNumber(value: Int16(0o500))],
            ofItemAtPath: protectedDirectory.path
        )
        defer {
            try? FileManager.default.setAttributes(
                [.posixPermissions: NSNumber(value: Int16(0o700))],
                ofItemAtPath: protectedDirectory.path
            )
        }

        do {
            _ = try await workspace.deleteIndex(index.id)
            Issue.record("Expected a typed error when the store file could not be removed")
        } catch let error as IndexEngineError {
            #expect(error.code == "chartroom.workspace.store-delete-failed")
        }

        #expect(try await workspace.indexes().isEmpty, "The catalog-first orphan policy must remain intact")
        let reloaded = makeWorkspace(root: root)
        #expect(try await reloaded.indexes().isEmpty, "The catalog on disk must remain authoritative")
        #expect(FileManager.default.fileExists(atPath: storeURL.path), "The failed cleanup remains visible on disk")
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

private enum LifecycleTestError: Error {
    case timedOutWaitingForCatalogRemoval
    case timedOutWaitingForOperationStart
}

private actor LifecycleGate {
    private var entries = 0
    private var released = false
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    func suspend() async {
        entries += 1
        await withCheckedContinuation { continuation in
            if released {
                continuation.resume()
            } else {
                releaseWaiters.append(continuation)
            }
        }
    }

    func waitUntilEntered() async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(2))
        while clock.now < deadline {
            if entries > 0 {
                return
            }
            try await Task.sleep(for: .milliseconds(5))
        }
        throw LifecycleTestError.timedOutWaitingForOperationStart
    }

    func entryCount() -> Int {
        entries
    }

    func release() {
        released = true
        let waitingForRelease = releaseWaiters
        releaseWaiters.removeAll()
        for waiter in waitingForRelease {
            waiter.resume()
        }
    }
}

private actor ReentrantDeletionProbe {
    private var attempted = false
    private var code: String?

    func claimAttempt() -> Bool {
        guard !attempted else { return false }
        attempted = true
        return true
    }

    func record(code: String) {
        self.code = code
    }

    func resultCode() -> String? {
        code
    }
}

private actor LifecycleGateEngine: IndexEngineClient {
    let searchGate: LifecycleGate?
    let ingestGate: LifecycleGate?

    init(searchGate: LifecycleGate? = nil, ingestGate: LifecycleGate? = nil) {
        self.searchGate = searchGate
        self.ingestGate = ingestGate
    }

    func ingest(_ request: IngestRequest) async throws -> IngestionSummary {
        await ingestGate?.suspend()
        return IngestionSummary(
            jobID: request.jobID,
            acceptedCount: request.payloads.count,
            failedCount: 0,
            failures: [],
            startedAt: .now,
            finishedAt: .now
        )
    }

    func delete(_ request: DeleteRequest) async throws -> DeletionSummary {
        DeletionSummary(
            jobID: request.jobID,
            requestedCount: request.documentIDs.count,
            deletedCount: request.documentIDs.count,
            failedCount: 0,
            failures: [],
            startedAt: .now,
            finishedAt: .now
        )
    }

    func search(_ request: SearchRequest) async throws -> SearchResponse {
        await searchGate?.suspend()
        return SearchResponse(
            query: request.query,
            mode: request.mode,
            results: [],
            diagnostics: SearchDiagnostics()
        )
    }

    func browseDocuments(_ request: DocumentBrowseRequest) async throws -> DocumentBrowseResponse {
        DocumentBrowseResponse(request: request, documents: [], totalMatching: 0)
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
        ModelStatusSnapshot(
            modelID: "lifecycle-gate",
            embeddingSpaceID: nil,
            dimension: 0,
            isAvailable: true,
            isModelBacked: false
        )
    }

    func snapshot() async -> IndexEngineSnapshot {
        IndexEngineSnapshot(
            storeURL: nil,
            objectCount: 0,
            modelID: "lifecycle-gate",
            embeddingDimension: 0,
            embeddingSpaceID: "lifecycle-gate:0",
            lastIngestedAt: nil,
            policyStates: []
        )
    }

    func rebuildEmbeddings() async throws -> EmbeddingRebuildSummary {
        EmbeddingRebuildSummary()
    }
}

private struct LifecycleSourceConnector: SourceConnector {
    let id: ConnectorID = "lifecycle-source"
    let capabilities = ConnectorCapabilities(supportsIncrementalSync: true, supportsRuntimeTools: false)

    func validate() async throws {}

    func changes(since cursor: SourceCursor?) async throws -> AsyncThrowingStream<SourceEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(.upsert(SourcePayload(
                documentID: "lifecycle-document",
                sourceID: id,
                displayName: "Lifecycle",
                body: .text("blocked document")
            )))
            continuation.yield(.checkpoint("lifecycle-cursor"))
            continuation.finish()
        }
    }

    func fetch(_ reference: SourceReference) async throws -> SourcePayload {
        SourcePayload(
            documentID: "lifecycle-document",
            sourceID: id,
            sourceURI: reference.uri,
            displayName: "Lifecycle",
            body: .text("blocked document")
        )
    }
}

private final class LifecycleTestCursorStore: CursorStore, @unchecked Sendable {
    private var storage: [String: SourceCursor] = [:]

    func cursor(forKey key: String) -> SourceCursor? { storage[key] }
    func setCursor(_ cursor: SourceCursor, forKey key: String) { storage[key] = cursor }
}
