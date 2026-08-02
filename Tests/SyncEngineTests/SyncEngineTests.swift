import ConnectorEngine
import Foundation
import IndexEngine
import Testing
@testable import SyncEngine

@Suite("SyncOrchestrator cursor policy")
struct SyncOrchestratorTests {
    @Test("the cursor advances only after engine acceptance")
    func cursorAdvancesAfterAcceptance() async throws {
        let engine = ScriptedEngine()
        let store = InMemoryCursorStore()
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)

        let outcome = try await orchestrator.sync(
            connector: ScriptedConnector(events: [
                .upsert(Self.payload("doc-1")),
                .checkpoint("cursor-accepted"),
            ]),
            cursorKey: "key"
        )

        #expect(outcome.accepted == 1)
        #expect(outcome.newCursor == "cursor-accepted")
        #expect(outcome.hadChanges)
        #expect(store.cursor(forKey: "key") == "cursor-accepted")
    }

    @Test("the cursor does not advance when engine ingestion throws")
    func cursorHeldWhenIngestThrows() async throws {
        let engine = ScriptedEngine(ingestBehavior: .throwError)
        let store = InMemoryCursorStore()
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)

        let outcome = try await orchestrator.sync(
            connector: ScriptedConnector(events: [
                .upsert(Self.payload("doc-1")),
                .checkpoint("cursor-rejected"),
            ]),
            cursorKey: "key"
        )

        #expect(outcome.ingestFailed == 1)
        #expect(outcome.newCursor == nil)
        #expect(store.cursor(forKey: "key") == nil)
    }

    @Test("a partial ingestion failure holds the cursor back")
    func cursorHeldOnPartialFailure() async throws {
        // Failed files must remain behind the old cursor so a later sync can retry
        // them even if the file did not change on disk.
        let engine = ScriptedEngine(ingestBehavior: .failDocumentsSuffixed("-1"))
        let store = InMemoryCursorStore()
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)

        let outcome = try await orchestrator.sync(
            connector: ScriptedConnector(events: [
                .upsert(Self.payload("doc-0")),
                .upsert(Self.payload("doc-1")),
                .checkpoint("cursor-partial"),
            ]),
            cursorKey: "key"
        )

        #expect(outcome.accepted == 1)
        #expect(outcome.ingestFailed == 1)
        #expect(outcome.newCursor == nil)
        #expect(store.cursor(forKey: "key") == nil)
    }

    @Test("a checkpoint-only sync advances the cursor and reports no changes")
    func checkpointOnlySyncAdvancesCursor() async throws {
        let engine = ScriptedEngine()
        let store = InMemoryCursorStore()
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)

        let outcome = try await orchestrator.sync(
            connector: ScriptedConnector(events: [.checkpoint("cursor-noop")]),
            cursorKey: "key"
        )

        #expect(!outcome.hadChanges)
        #expect(outcome.newCursor == "cursor-noop")
        #expect(store.cursor(forKey: "key") == "cursor-noop")
        #expect(await engine.recordedCalls().isEmpty)
    }

    @Test("the stored cursor is passed to the connector on the next sync")
    func storedCursorReachesConnector() async throws {
        let engine = ScriptedEngine()
        let store = InMemoryCursorStore()
        store.setCursor("cursor-previous", forKey: "key")
        let connector = ScriptedConnector(events: [.checkpoint("cursor-previous")])
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)

        _ = try await orchestrator.sync(connector: connector, cursorKey: "key")

        #expect(connector.receivedCursor() == "cursor-previous")
    }

    @Test("deletes are deduplicated and applied before any ingest")
    func deletesAreDedupedAndOrderedBeforeIngest() async throws {
        let engine = ScriptedEngine()
        let store = InMemoryCursorStore()
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)

        let outcome = try await orchestrator.sync(
            connector: ScriptedConnector(events: [
                .upsert(Self.payload("doc-new")),
                .delete(documentID: "doc-gone"),
                .move(documentID: "doc-moved", newURI: URL(filePath: "/tmp/moved.md")),
                .delete(documentID: "doc-gone"),
                .checkpoint("cursor-deletes"),
            ]),
            cursorKey: "key"
        )

        let calls = await engine.recordedCalls()
        #expect(calls.first == .delete(["doc-gone", "doc-moved"]))
        #expect(calls.dropFirst().allSatisfy { call in
            if case .ingest = call { return true } else { return false }
        })
        #expect(outcome.deletedCount == 2)
        #expect(outcome.newCursor == "cursor-deletes")
    }

    @Test("unreadable paths and kept documents are surfaced without blocking the cursor")
    func unreadablePathsAreSurfaced() async throws {
        let engine = ScriptedEngine()
        let store = InMemoryCursorStore()
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)

        let outcome = try await orchestrator.sync(
            connector: ScriptedConnector(events: [
                .pathUnavailable(path: "/b/locked.md", reason: "permission denied"),
                .pathUnavailable(path: "/a/locked.md", reason: "permission denied"),
                .pathUnavailable(path: "/b/locked.md", reason: "permission denied"),
                .permissionChanged(documentID: "doc-kept"),
                .checkpoint("cursor-unreadable"),
            ]),
            cursorKey: "key"
        )

        #expect(outcome.unreadablePaths == ["/a/locked.md", "/b/locked.md"])
        #expect(outcome.unreadableKeptCount == 1)
        #expect(outcome.newCursor == "cursor-unreadable")
    }

    @Test("stopping a paused run wakes it, halts ingestion, and holds the cursor")
    func stoppingPausedRunWakesAndHaltsIt() async throws {
        let engine = ScriptedEngine(ingestDelay: .milliseconds(60))
        let store = InMemoryCursorStore()
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)
        let control = SyncControl()
        let progress = ProgressRecorder()

        let syncTask = Task {
            try await orchestrator.sync(
                connector: ScriptedConnector(events: [
                    .upsert(Self.payload("doc-0")),
                    .upsert(Self.payload("doc-1")),
                    .upsert(Self.payload("doc-2")),
                    .checkpoint("cursor-stopped"),
                ]),
                cursorKey: "key",
                control: control,
                onProgress: { await progress.record($0) }
            )
        }

        while await progress.updates().isEmpty {
            try await Task.sleep(for: .milliseconds(5))
        }
        control.pause()
        try await Task.sleep(for: .milliseconds(120))
        #expect(control.state == .paused)

        control.stop()
        let outcome = try await syncTask.value

        #expect(outcome.stopped)
        #expect(outcome.accepted < 3)
        #expect(outcome.newCursor == nil)
        #expect(store.cursor(forKey: "key") == nil)
    }

    @Test("cancelling a paused sync wakes it and holds the cursor")
    func cancellingPausedRunWakesAndHaltsIt() async throws {
        let engine = ScriptedEngine()
        let store = InMemoryCursorStore()
        store.setCursor("cursor-before", forKey: "key")
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)
        let control = SyncControl()
        control.pause()
        let connector = ScriptedConnector(events: [
            .upsert(Self.payload("doc-0")),
            .checkpoint("cursor-after-cancel"),
        ])
        let completion = CompletionRecorder()

        let syncTask = Task {
            do {
                let outcome = try await orchestrator.sync(
                    connector: connector,
                    cursorKey: "key",
                    control: control
                )
                await completion.markCompleted()
                return outcome
            } catch {
                await completion.markCompleted()
                throw error
            }
        }

        let clock = ContinuousClock()
        let startDeadline = clock.now.advanced(by: .seconds(15))
        while connector.receivedCursor() != "cursor-before", clock.now < startDeadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(connector.receivedCursor() == "cursor-before")
        try await Task.sleep(for: .milliseconds(20))
        #expect(await engine.recordedCalls().isEmpty)

        syncTask.cancel()
        let cancellationDeadline = clock.now.advanced(by: .seconds(15))
        while !(await completion.hasCompleted()), clock.now < cancellationDeadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        let completedWithoutRescue = await completion.hasCompleted()
        if !completedWithoutRescue {
            control.stop()
        }
        let outcome = try await syncTask.value

        #expect(completedWithoutRescue, "Cancellation must resume a task parked by pause")
        #expect(outcome.stopped)
        #expect(outcome.accepted == 0)
        #expect(outcome.newCursor == nil)
        #expect(store.cursor(forKey: "key") == "cursor-before")
    }

    @Test("stopping while connector collection is blocked cancels collection and holds the cursor")
    func stoppingBlockedCollectionCancelsIt() async throws {
        let engine = ScriptedEngine()
        let store = InMemoryCursorStore()
        store.setCursor("cursor-before", forKey: "key")
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)
        let connector = BlockingConnector()
        let control = SyncControl()
        let completion = CompletionRecorder()

        let syncTask = Task {
            let outcome = try await orchestrator.sync(
                connector: connector,
                cursorKey: "key",
                control: control
            )
            await completion.markCompleted()
            return outcome
        }

        let clock = ContinuousClock()
        let startDeadline = clock.now.advanced(by: .seconds(15))
        while !connector.hasStarted, clock.now < startDeadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(connector.hasStarted)

        control.stop()
        let stopDeadline = clock.now.advanced(by: .seconds(15))
        while !(await completion.hasCompleted()), clock.now < stopDeadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        let completedWithoutRescue = await completion.hasCompleted()
        if !completedWithoutRescue {
            connector.finish()
        }
        let outcome = try await syncTask.value

        #expect(completedWithoutRescue, "Stop must cancel connector collection instead of waiting for its next event")
        #expect(connector.wasCancelled)
        #expect(outcome.stopped)
        #expect(outcome.newCursor == nil)
        #expect(store.cursor(forKey: "key") == "cursor-before")
        #expect(await engine.recordedCalls().isEmpty)
    }

    @Test("cancelling while connector collection is blocked returns a stopped outcome")
    func cancellingBlockedCollectionStopsRun() async throws {
        let engine = ScriptedEngine()
        let store = InMemoryCursorStore()
        store.setCursor("cursor-before", forKey: "key")
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)
        let connector = BlockingConnector()
        let completion = CompletionRecorder()

        let syncTask = Task {
            let outcome = try await orchestrator.sync(
                connector: connector,
                cursorKey: "key"
            )
            await completion.markCompleted()
            return outcome
        }

        let clock = ContinuousClock()
        let startDeadline = clock.now.advanced(by: .seconds(15))
        while !connector.hasStarted, clock.now < startDeadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        #expect(connector.hasStarted)

        syncTask.cancel()
        let cancellationDeadline = clock.now.advanced(by: .seconds(15))
        while !(await completion.hasCompleted()), clock.now < cancellationDeadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        let completedWithoutRescue = await completion.hasCompleted()
        if !completedWithoutRescue {
            connector.finish()
        }
        let outcome = try await syncTask.value

        #expect(completedWithoutRescue, "Cancellation must terminate connector collection")
        #expect(connector.wasCancelled)
        #expect(outcome.stopped)
        #expect(outcome.newCursor == nil)
        #expect(store.cursor(forKey: "key") == "cursor-before")
        #expect(await engine.recordedCalls().isEmpty)
    }

    @Test("pause gates deletion and cursor advancement after connector collection")
    func pauseGatesDeletionAndCursorAdvancement() async throws {
        let engine = ScriptedEngine()
        let store = InMemoryCursorStore()
        store.setCursor("cursor-before", forKey: "key")
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)
        let connector = ScriptedConnector(events: [
            .delete(documentID: "doc-gone"),
            .checkpoint("cursor-after"),
        ])
        let control = SyncControl()
        control.pause()

        let syncTask = Task {
            try await orchestrator.sync(
                connector: connector,
                cursorKey: "key",
                control: control
            )
        }

        let clock = ContinuousClock()
        let collectionDeadline = clock.now.advanced(by: .seconds(15))
        while connector.receivedCursor() != "cursor-before", clock.now < collectionDeadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        try await Task.sleep(for: .milliseconds(20))
        let callsWhilePaused = await engine.recordedCalls()
        let cursorWhilePaused = store.cursor(forKey: "key")

        control.resume()
        let outcome = try await syncTask.value

        #expect(callsWhilePaused.isEmpty)
        #expect(cursorWhilePaused == "cursor-before")
        #expect(outcome.deletedCount == 1)
        #expect(outcome.newCursor == "cursor-after")
        #expect(store.cursor(forKey: "key") == "cursor-after")
    }

    @Test("progress reports the payload total and the item being ingested")
    func progressReportsTotalsAndCurrentItem() async throws {
        let engine = ScriptedEngine()
        let store = InMemoryCursorStore()
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: store)
        let progress = ProgressRecorder()

        _ = try await orchestrator.sync(
            connector: ScriptedConnector(events: [
                .upsert(Self.payload("doc-0")),
                .upsert(Self.payload("doc-1")),
                .checkpoint("cursor-progress"),
            ]),
            cursorKey: "key",
            onProgress: { await progress.record($0) }
        )

        let updates = await progress.updates()
        #expect(updates.count == 4)
        #expect(updates.allSatisfy { $0.totalPayloads == 2 })
        #expect(updates.first?.currentItem == "Payload doc-0")
        #expect(updates.last?.completedPayloads == 2)
        #expect(updates.last?.accepted == 2)
    }

    static func payload(_ id: String) -> SourcePayload {
        SourcePayload(
            documentID: EngineID(rawValue: id),
            sourceID: "scripted",
            sourceURI: URL(filePath: "/tmp/\(id).md"),
            displayName: "Payload \(id)",
            body: .text("fixture body for \(id)")
        )
    }
}

@Suite("UserDefaultsCursorStore")
struct UserDefaultsCursorStoreTests {
    @Test("cursors persist across store instances sharing a defaults suite")
    func cursorsPersistAcrossInstances() throws {
        let suiteName = "SyncEngineTests-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let first = UserDefaultsCursorStore(defaults: defaults, storageKey: "cursors")
        first.setCursor("cursor-a", forKey: "source-a")
        first.setCursor("cursor-b", forKey: "source-b")

        let second = UserDefaultsCursorStore(defaults: defaults, storageKey: "cursors")
        #expect(second.cursor(forKey: "source-a") == "cursor-a")
        #expect(second.cursor(forKey: "source-b") == "cursor-b")
        #expect(second.cursor(forKey: "source-c") == nil)
    }

    @Test("concurrent writes to different cursor keys cannot overwrite one another")
    func concurrentWritesPreserveBothKeys() throws {
        let suiteName = "SyncEngineTests-\(UUID().uuidString)"
        let defaults = try #require(CoordinatedDictionaryReadUserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }
        let store = UserDefaultsCursorStore(defaults: defaults, storageKey: "cursors")
        let writes = DispatchGroup()
        let queue = DispatchQueue(label: "SyncEngineTests.cursor-writes", attributes: .concurrent)

        writes.enter()
        queue.async {
            store.setCursor("cursor-a", forKey: "source-a")
            writes.leave()
        }
        writes.enter()
        queue.async {
            store.setCursor("cursor-b", forKey: "source-b")
            writes.leave()
        }

        let completedWithoutRescue = writes.wait(timeout: .now() + 15) == .success
        if !completedWithoutRescue {
            defaults.releaseBlockedReads()
            writes.wait()
        }

        #expect(completedWithoutRescue)
        #expect(store.cursor(forKey: "source-a") == "cursor-a")
        #expect(store.cursor(forKey: "source-b") == "cursor-b")
    }

    @Test("per-key cursor storage still reads legacy dictionary entries")
    func perKeyStorageReadsLegacyDictionary() throws {
        let suiteName = "SyncEngineTests-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }
        defaults.set(["source": "cursor-legacy"], forKey: "cursors")
        let store = UserDefaultsCursorStore(defaults: defaults, storageKey: "cursors")

        #expect(store.cursor(forKey: "source") == "cursor-legacy")

        store.setCursor("cursor-current", forKey: "source")
        #expect(store.cursor(forKey: "source") == "cursor-current")
    }
}

@Suite("LocalFileRetry")
struct LocalFileRetryTests {
    @Test("retry re-applies the connector's file policy, not just existence")
    func retryReappliesFilePolicy() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "sync-engine-retry-policy-\(UUID().uuidString)", directoryHint: .isDirectory)
        let outside = FileManager.default.temporaryDirectory
            .appending(path: "sync-engine-retry-outside-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }

        // The file recorded at sync time has since been replaced by a symlink to a
        // secret outside the tree — the exact swap the connector's scan rejects.
        let secretURL = outside.appending(path: "secret.md")
        try "credential material".write(to: secretURL, atomically: true, encoding: .utf8)
        let swappedURL = root.appending(path: "Swapped.md")
        try FileManager.default.createSymbolicLink(at: swappedURL, withDestinationURL: secretURL)

        let grownURL = root.appending(path: "Grown.md")
        try String(repeating: "x", count: 4096).write(to: grownURL, atomically: true, encoding: .utf8)

        let wrongTypeURL = root.appending(path: "Binary.bin")
        try Data([0x00, 0x01]).write(to: wrongTypeURL)

        let validURL = root.appending(path: "Valid.md")
        try "still fine".write(to: validURL, atomically: true, encoding: .utf8)

        let failures = [
            failure(documentID: "local-files:Swapped.md", sourceID: "local-files", sourceURI: swappedURL),
            failure(documentID: "local-files:Grown.md", sourceID: "local-files", sourceURI: grownURL),
            failure(documentID: "local-files:Binary.bin", sourceID: "local-files", sourceURI: wrongTypeURL),
            failure(documentID: "local-files:Valid.md", sourceID: "local-files", sourceURI: validURL),
        ]
        let payloads = LocalFileRetry.payloads(
            for: failures,
            options: .init(allowedPathExtensions: ["md"], maxFileSizeBytes: 1024)
        )

        #expect(payloads.map(\.documentID) == ["local-files:Valid.md"])
    }

    @Test("rebuilds payloads only for local-file failures whose file still exists")
    func rebuildsPayloadsForExistingLocalFiles() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "sync-engine-retry-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let existingURL = root.appending(path: "Existing.md")
        try "retry me".write(to: existingURL, atomically: true, encoding: .utf8)
        let missingURL = root.appending(path: "Missing.md")

        let failures = [
            failure(documentID: "local-files:Existing.md", sourceID: "local-files", sourceURI: existingURL),
            failure(documentID: "local-files:Missing.md", sourceID: "local-files", sourceURI: missingURL),
            failure(documentID: "other:Existing.md", sourceID: "other-source", sourceURI: existingURL),
        ]

        let payloads = LocalFileRetry.payloads(for: failures)

        #expect(payloads.count == 1)
        let payload = try #require(payloads.first)
        #expect(payload.documentID == "local-files:Existing.md")
        #expect(payload.sourceID == LocalFileRetry.localFilesSourceID)
        #expect(payload.displayName == "Existing.md")
    }

    private func failure(documentID: String, sourceID: SourceID, sourceURI: URL) -> FailureSnapshot {
        FailureSnapshot(
            id: EngineID(rawValue: "failure-\(documentID)"),
            category: .embeddingFailure,
            message: "Could not ingest \(documentID)",
            detail: "test failure",
            sourceID: sourceID,
            documentID: DocumentID(rawValue: documentID),
            sourceURI: sourceURI,
            recoverability: .retryable,
            occurredAt: .now
        )
    }
}

// MARK: - Test doubles

/// Emits a fixed event script and records the cursor it was asked to resume from.
private final class ScriptedConnector: SourceConnector, @unchecked Sendable {
    let id: ConnectorID = "scripted"
    let capabilities = ConnectorCapabilities(supportsIncrementalSync: true, supportsRuntimeTools: false)

    private let events: [SourceEvent]
    private let lock = NSLock()
    private var _receivedCursor: SourceCursor?

    init(events: [SourceEvent]) {
        self.events = events
    }

    func receivedCursor() -> SourceCursor? {
        lock.withLock { _receivedCursor }
    }

    func validate() async throws {}

    func changes(since cursor: SourceCursor?) async throws -> AsyncThrowingStream<SourceEvent, Error> {
        lock.withLock { _receivedCursor = cursor }
        return AsyncThrowingStream { continuation in
            for event in events {
                continuation.yield(event)
            }
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

private final class BlockingConnector: SourceConnector, @unchecked Sendable {
    let id: ConnectorID = "blocking"
    let capabilities = ConnectorCapabilities(supportsIncrementalSync: true, supportsRuntimeTools: false)

    private let lock = NSLock()
    private var continuation: AsyncThrowingStream<SourceEvent, Error>.Continuation?
    private var started = false
    private var cancelled = false

    var hasStarted: Bool {
        lock.withLock { started }
    }

    var wasCancelled: Bool {
        lock.withLock { cancelled }
    }

    func validate() async throws {}

    func changes(since cursor: SourceCursor?) async throws -> AsyncThrowingStream<SourceEvent, Error> {
        AsyncThrowingStream { continuation in
            lock.withLock {
                self.continuation = continuation
                started = true
            }
            continuation.onTermination = { [weak self] termination in
                guard case .cancelled = termination else { return }
                self?.lock.withLock {
                    self?.cancelled = true
                    self?.continuation = nil
                }
            }
        }
    }

    func finish() {
        let continuation = lock.withLock {
            defer { self.continuation = nil }
            return self.continuation
        }
        continuation?.finish()
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

private final class InMemoryCursorStore: CursorStore, @unchecked Sendable {
    private let lock = NSLock()
    private var cursors: [String: SourceCursor] = [:]

    func cursor(forKey key: String) -> SourceCursor? {
        lock.withLock { cursors[key] }
    }

    func setCursor(_ cursor: SourceCursor, forKey key: String) {
        lock.withLock { cursors[key] = cursor }
    }
}

private final class CoordinatedDictionaryReadUserDefaults: UserDefaults, @unchecked Sendable {
    private let condition = NSCondition()
    private var readCount = 0
    private var readsReleased = false

    override func dictionary(forKey defaultName: String) -> [String: Any]? {
        condition.lock()
        readCount += 1
        if readCount == 2 {
            readsReleased = true
            condition.broadcast()
        }
        while !readsReleased {
            condition.wait()
        }
        condition.unlock()
        return super.dictionary(forKey: defaultName)
    }

    func releaseBlockedReads() {
        condition.lock()
        readsReleased = true
        condition.broadcast()
        condition.unlock()
    }
}

private actor ProgressRecorder {
    private var recorded: [SyncProgressUpdate] = []

    func record(_ update: SyncProgressUpdate) {
        recorded.append(update)
    }

    func updates() -> [SyncProgressUpdate] {
        recorded
    }
}

private actor CompletionRecorder {
    private var completed = false

    func markCompleted() {
        completed = true
    }

    func hasCompleted() -> Bool {
        completed
    }
}

/// An engine whose ingest behavior is scripted per test and which records the order of
/// mutating calls so ordering guarantees can be asserted.
private actor ScriptedEngine: IndexEngineClient {
    enum IngestBehavior {
        case acceptAll
        case throwError
        case failDocumentsSuffixed(String)
    }

    enum Call: Equatable {
        case ingest([DocumentID])
        case delete([DocumentID])
    }

    private let ingestBehavior: IngestBehavior
    private let ingestDelay: Duration?
    private var calls: [Call] = []

    init(ingestBehavior: IngestBehavior = .acceptAll, ingestDelay: Duration? = nil) {
        self.ingestBehavior = ingestBehavior
        self.ingestDelay = ingestDelay
    }

    func recordedCalls() -> [Call] {
        calls
    }

    func ingest(_ request: IngestRequest) async throws -> IngestionSummary {
        calls.append(.ingest(request.payloads.map(\.documentID)))
        if let ingestDelay {
            try? await Task.sleep(for: ingestDelay)
        }

        switch ingestBehavior {
        case .acceptAll:
            return IngestionSummary(
                jobID: request.jobID,
                acceptedCount: request.payloads.count,
                failedCount: 0,
                failures: [],
                startedAt: .now,
                finishedAt: .now
            )
        case .throwError:
            throw IndexEngineError(
                .ingestionFailed,
                code: "test.ingest.rejected",
                recoverability: .retryable,
                summary: "Rejected by test engine"
            )
        case let .failDocumentsSuffixed(suffix):
            let failures = request.payloads.filter { $0.documentID.rawValue.hasSuffix(suffix) }.map { payload in
                FailureSnapshot(
                    id: EngineID(rawValue: "failure-\(payload.documentID.rawValue)"),
                    category: .embeddingFailure,
                    message: "Could not ingest \(payload.displayName)",
                    detail: "test partial failure",
                    sourceID: payload.sourceID,
                    documentID: payload.documentID,
                    isRecoverable: true,
                    occurredAt: .now
                )
            }
            return IngestionSummary(
                jobID: request.jobID,
                acceptedCount: request.payloads.count - failures.count,
                failedCount: failures.count,
                failures: failures,
                startedAt: .now,
                finishedAt: .now
            )
        }
    }

    func delete(_ request: DeleteRequest) async throws -> DeletionSummary {
        calls.append(.delete(request.documentIDs))
        return DeletionSummary(
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
        SearchResponse(query: request.query, mode: request.mode, results: [], diagnostics: SearchDiagnostics())
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
        ModelStatusSnapshot(modelID: "scripted", embeddingSpaceID: nil, dimension: 0, isAvailable: true, isModelBacked: false)
    }

    func snapshot() async -> IndexEngineSnapshot {
        IndexEngineSnapshot(
            storeURL: nil,
            objectCount: 0,
            modelID: "scripted",
            embeddingDimension: 0,
            embeddingSpaceID: "scripted:0",
            lastIngestedAt: nil,
            policyStates: []
        )
    }
}
