import ConnectorEngine
import Foundation
import SyncEngine
import Testing
@testable import ChartroomControl
@testable import IndexEngine

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
        #expect(status.diagnosticHistory?.jobsAvailability == .available)
        #expect(status.diagnosticHistory?.failuresAvailability == .available)
    }

    @Test("status and session history propagate a durable channel failure")
    func propagatesUnavailableDiagnosticHistory() async throws {
        let storeURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("chartroom-history-\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: storeURL) }
        let engine = try await IndexEngine.open(
            storeURL: storeURL,
            configuration: .init(embedder: HashingEmbedder(dimension: 4))
        )
        let db = try SQLite(path: storeURL.path)
        try db.exec("DROP TABLE failures")
        let session = ChartroomSession(
            engineFactory: { (engine, storeURL) },
            cursorStore: InMemoryCursorStore()
        )

        let status = try await session.open()
        #expect(status.diagnosticHistory?.failuresAvailability == .unavailable)
        #expect(status.diagnosticHistory?.jobsAvailability == .available)

        let history = try await session.diagnosticHistory(limit: 10)
        #expect(history.failuresAvailability == .unavailable)
        #expect(history.jobsAvailability == .available)
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

    @Test("syncs sharing a cursor key run serially and the follower reads the leader cursor")
    func sameKeySyncsRunSerially() async throws {
        let cursors = InMemoryCursorStore()
        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: cursors
        )
        _ = try await session.open()

        let firstGate = ConnectorEntryGate()
        let secondGate = ConnectorEntryGate(released: true)
        let firstConnector = GatedCheckpointConnector(
            id: "first",
            checkpoint: "cursor-first",
            gate: firstGate
        )
        let secondConnector = GatedCheckpointConnector(
            id: "second",
            checkpoint: "cursor-second",
            gate: secondGate
        )

        let firstTask = Task {
            try await session.sync(connector: firstConnector, cursorKey: "shared")
        }
        await firstGate.waitUntilEntered()

        let secondTask = Task {
            try await session.sync(connector: secondConnector, cursorKey: "shared")
        }
        let clock = ContinuousClock()
        let overlapDeadline = clock.now.advanced(by: .milliseconds(100))
        while !(await secondGate.hasEntered), clock.now < overlapDeadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        let secondOverlappedFirst = await secondGate.hasEntered

        await firstGate.release()
        _ = try await firstTask.value
        _ = try await secondTask.value

        #expect(!secondOverlappedFirst)
        #expect(firstConnector.receivedCursor == nil)
        #expect(secondConnector.receivedCursor == "cursor-first")
        #expect(cursors.cursor(forKey: "shared") == "cursor-second")
    }

    @Test("syncs with different cursor keys remain concurrent")
    func differentKeySyncsRemainConcurrent() async throws {
        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: InMemoryCursorStore()
        )
        _ = try await session.open()

        let firstGate = ConnectorEntryGate()
        let secondGate = ConnectorEntryGate(released: true)
        let firstConnector = GatedCheckpointConnector(
            id: "first",
            checkpoint: "cursor-first",
            gate: firstGate
        )
        let secondConnector = GatedCheckpointConnector(
            id: "second",
            checkpoint: "cursor-second",
            gate: secondGate
        )

        let firstTask = Task {
            try await session.sync(connector: firstConnector, cursorKey: "first-key")
        }
        await firstGate.waitUntilEntered()

        let secondTask = Task {
            try await session.sync(connector: secondConnector, cursorKey: "second-key")
        }
        let clock = ContinuousClock()
        let startDeadline = clock.now.advanced(by: .seconds(1))
        while !(await secondGate.hasEntered), clock.now < startDeadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        let secondStartedBeforeFirstFinished = await secondGate.hasEntered

        await firstGate.release()
        _ = try await firstTask.value
        _ = try await secondTask.value

        #expect(secondStartedBeforeFirstFinished)
        #expect(secondConnector.receivedCursor == nil)
    }

    @Test("cancelling a queued same-key sync removes it without waiting for the leader")
    func cancellingQueuedSameKeySyncRemovesIt() async throws {
        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: InMemoryCursorStore()
        )
        _ = try await session.open()

        let firstGate = ConnectorEntryGate()
        let secondGate = ConnectorEntryGate(released: true)
        let firstTask = Task {
            try await session.sync(
                connector: GatedCheckpointConnector(
                    id: "first",
                    checkpoint: "cursor-first",
                    gate: firstGate
                ),
                cursorKey: "shared"
            )
        }
        await firstGate.waitUntilEntered()

        let completion = SessionTaskCompletion()
        let secondTask = Task {
            do {
                let outcome = try await session.sync(
                    connector: GatedCheckpointConnector(
                        id: "second",
                        checkpoint: "cursor-second",
                        gate: secondGate
                    ),
                    cursorKey: "shared"
                )
                await completion.markCompleted()
                return outcome
            } catch {
                await completion.markCompleted()
                throw error
            }
        }
        try await Task.sleep(for: .milliseconds(20))
        secondTask.cancel()

        let clock = ContinuousClock()
        let cancellationDeadline = clock.now.advanced(by: .seconds(1))
        while !(await completion.hasCompleted), clock.now < cancellationDeadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        let completedBeforeLeader = await completion.hasCompleted
        if !completedBeforeLeader {
            await firstGate.release()
        }
        let secondOutcome = try await secondTask.value
        let secondConnectorStarted = await secondGate.hasEntered

        await firstGate.release()
        _ = try await firstTask.value

        #expect(completedBeforeLeader)
        #expect(secondOutcome.stopped)
        #expect(!secondConnectorStarted)
    }

    @Test("close() stops in-flight syncs and drains before returning")
    func closeStopsInFlightSyncsAndDrains() async throws {
        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: InMemoryCursorStore()
        )
        _ = try await session.open()

        // The gate is never released: only close() stopping the sync can finish it.
        let gate = ConnectorEntryGate()
        let completion = SessionTaskCompletion()
        let syncTask = Task {
            defer { Task { await completion.markCompleted() } }
            return try await session.sync(
                connector: GatedCheckpointConnector(id: "gated", checkpoint: "cursor", gate: gate),
                cursorKey: "close-drain"
            )
        }
        await gate.waitUntilEntered()

        try await session.close()

        // close() must not return while the sync is still running; the stop it issued
        // lets the orchestrator finish without the gate ever being released.
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(2))
        while !(await completion.hasCompleted), clock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(await completion.hasCompleted, "close() returned while the sync was still in flight")

        // Released only for cleanup; a drained sync has already finished without it.
        await gate.release()
        let outcome = try await syncTask.value
        #expect(outcome.stopped)

        let status = await session.status()
        #expect(status.state == .unopened)
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

private actor ConnectorEntryGate {
    private var entered = false
    private var released: Bool
    private var entryWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    init(released: Bool = false) {
        self.released = released
    }

    var hasEntered: Bool { entered }

    func enterAndWait() async {
        entered = true
        let entryWaiters = self.entryWaiters
        self.entryWaiters.removeAll()
        for waiter in entryWaiters {
            waiter.resume()
        }

        guard !released else { return }
        await withCheckedContinuation { continuation in
            releaseWaiters.append(continuation)
        }
    }

    func waitUntilEntered() async {
        guard !entered else { return }
        await withCheckedContinuation { continuation in
            entryWaiters.append(continuation)
        }
    }

    func release() {
        released = true
        let releaseWaiters = self.releaseWaiters
        self.releaseWaiters.removeAll()
        for waiter in releaseWaiters {
            waiter.resume()
        }
    }
}

private actor SessionTaskCompletion {
    private var completed = false

    var hasCompleted: Bool { completed }

    func markCompleted() {
        completed = true
    }
}

private final class GatedCheckpointConnector: SourceConnector, @unchecked Sendable {
    let id: ConnectorID
    let capabilities = ConnectorCapabilities(supportsIncrementalSync: true, supportsRuntimeTools: false)

    private let checkpoint: SourceCursor
    private let gate: ConnectorEntryGate
    private let lock = NSLock()
    private var _receivedCursor: SourceCursor?

    init(id: ConnectorID, checkpoint: SourceCursor, gate: ConnectorEntryGate) {
        self.id = id
        self.checkpoint = checkpoint
        self.gate = gate
    }

    var receivedCursor: SourceCursor? {
        lock.withLock { _receivedCursor }
    }

    func validate() async throws {}

    func changes(since cursor: SourceCursor?) async throws -> AsyncThrowingStream<SourceEvent, Error> {
        lock.withLock {
            _receivedCursor = cursor
        }
        await gate.enterAndWait()
        return AsyncThrowingStream { continuation in
            continuation.yield(.checkpoint(checkpoint))
            continuation.finish()
        }
    }

    func fetch(_ reference: SourceReference) async throws -> SourcePayload {
        SourcePayload(
            documentID: EngineID(rawValue: reference.externalID ?? reference.uri.lastPathComponent),
            sourceID: id,
            sourceURI: reference.uri,
            displayName: reference.uri.lastPathComponent,
            body: .binaryReference(reference.uri)
        )
    }
}
