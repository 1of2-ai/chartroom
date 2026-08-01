import ConnectorEngine
import Foundation
import IndexEngine
import SyncEngine

private enum ChartroomSessionOperationContext {
    @TaskLocal static var sessionID: UUID?
}

private final class CursorKeySyncCoordinator: @unchecked Sendable {
    private struct Waiter {
        var id: UUID
        var continuation: CheckedContinuation<Bool, Never>
    }

    private let lock = NSLock()
    private var activeKeys: Set<String> = []
    private var waiters: [String: [Waiter]] = [:]

    func acquire(_ key: String) async -> Bool {
        guard !Task.isCancelled else { return false }
        let waiterID = UUID()

        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                let immediateResult: Bool? = lock.withLock {
                    guard !Task.isCancelled else { return false }
                    if activeKeys.insert(key).inserted {
                        return true
                    }
                    waiters[key, default: []].append(
                        Waiter(id: waiterID, continuation: continuation)
                    )
                    return nil
                }
                if let immediateResult {
                    continuation.resume(returning: immediateResult)
                }
            }
        } onCancel: {
            let continuation: CheckedContinuation<Bool, Never>? = lock.withLock {
                guard var queued = waiters[key],
                      let index = queued.firstIndex(where: { $0.id == waiterID })
                else {
                    return nil
                }
                let continuation = queued.remove(at: index).continuation
                if queued.isEmpty {
                    waiters.removeValue(forKey: key)
                } else {
                    waiters[key] = queued
                }
                return continuation
            }
            continuation?.resume(returning: false)
        }
    }

    func release(_ key: String) {
        let continuation: CheckedContinuation<Bool, Never>? = lock.withLock {
            guard var queued = waiters[key], !queued.isEmpty else {
                activeKeys.remove(key)
                return nil
            }
            let continuation = queued.removeFirst().continuation
            if queued.isEmpty {
                waiters.removeValue(forKey: key)
            } else {
                waiters[key] = queued
            }
            return continuation
        }
        continuation?.resume(returning: true)
    }
}

public enum ChartroomSessionState: String, Codable, Hashable, Sendable {
    case unopened
    case opening
    case ready
    case failed
    case invalidated
}

public struct ChartroomStatus: Codable, Sendable {
    public var state: ChartroomSessionState
    public var storeURL: URL?
    public var snapshot: IndexEngineSnapshot?
    public var health: IndexHealthSnapshot?
    public var modelStatus: ModelStatusSnapshot?
    public var diagnosticHistory: DiagnosticHistory?
    public var jobs: [JobSnapshot]
    public var failures: [FailureSnapshot]
    public var lastError: String?

    public init(
        state: ChartroomSessionState,
        storeURL: URL? = nil,
        snapshot: IndexEngineSnapshot? = nil,
        health: IndexHealthSnapshot? = nil,
        modelStatus: ModelStatusSnapshot? = nil,
        jobs: [JobSnapshot] = [],
        failures: [FailureSnapshot] = [],
        lastError: String? = nil
    ) {
        self.state = state
        self.storeURL = storeURL
        self.snapshot = snapshot
        self.health = health
        self.modelStatus = modelStatus
        self.diagnosticHistory = nil
        self.jobs = jobs
        self.failures = failures
        self.lastError = lastError
    }
}

public actor ChartroomSession {
    public typealias EngineFactory = @Sendable () async throws -> (any IndexEngineClient, URL?)

    private struct PendingOpen {
        var id: UUID
        var generation: UInt64
        var task: Task<(any IndexEngineClient, URL?), Error>
    }

    private let engineFactory: EngineFactory
    private let cursorStore: any CursorStore
    private let syncCoordinator = CursorKeySyncCoordinator()
    private var state: ChartroomSessionState = .unopened
    private var engine: (any IndexEngineClient)?
    private var storeURL: URL?
    private var lastSearch: SearchResponse?
    private var lastError: String?
    private struct QuiescenceWaiter {
        /// Whether a pending `open()` keeps this waiter suspended. `invalidateAndWait`
        /// waits for everything; `close()` must not — an open never touches the engine
        /// being retired, and a factory that ignores cancellation would deadlock it.
        var includesOpens: Bool
        var continuation: CheckedContinuation<Void, Never>
    }

    private var isInvalidated = false
    private var activeOperationCount = 0
    private var activeOpenCount = 0
    private var quiescenceWaiters: [QuiescenceWaiter] = []
    private var lifecycleGeneration: UInt64 = 0
    private var pendingOpen: PendingOpen?
    private var activeSyncControls: [UUID: SyncControl] = [:]
    private nonisolated let operationContextID = UUID()

    public init(engineFactory: @escaping EngineFactory, cursorStore: any CursorStore) {
        self.engineFactory = engineFactory
        self.cursorStore = cursorStore
    }

    public func open() async throws -> ChartroomStatus {
        try beginOperation()
        activeOpenCount += 1
        defer { endOperation(isOpen: true) }

        return try await withOperationContext {
            if state == .ready {
                return try await currentStatus(
                    validatingGeneration: lifecycleGeneration
                )
            }

            let opening: PendingOpen
            if let pendingOpen {
                opening = pendingOpen
            } else {
                state = .opening
                lastError = nil
                let factory = engineFactory
                opening = PendingOpen(
                    id: UUID(),
                    generation: lifecycleGeneration,
                    task: Task {
                        try await factory()
                    }
                )
                pendingOpen = opening
            }

            do {
                let opened = try await opening.task.value
                guard !isInvalidated else {
                    throw invalidatedError()
                }
                guard opening.generation == lifecycleGeneration else {
                    throw CancellationError()
                }
                if pendingOpen?.id == opening.id {
                    pendingOpen = nil
                    engine = opened.0
                    storeURL = opened.1
                    state = .ready
                }
                return try await currentStatus(
                    validatingGeneration: opening.generation
                )
            } catch {
                if isInvalidated {
                    throw invalidatedError()
                }
                guard opening.generation == lifecycleGeneration else {
                    throw CancellationError()
                }
                if pendingOpen?.id == opening.id {
                    pendingOpen = nil
                    state = .failed
                    lastError = String(describing: error)
                }
                throw error
            }
        }
    }

    /// Releases the current engine instance. The next `open()` rebuilds it through this
    /// session's factory, which lets a host apply a changed model configuration without
    /// constructing an engine outside the product command surface.
    ///
    /// Cancel-then-drain: in-flight syncs are stopped (including ones running under a
    /// caller-supplied `SyncControl`) and every active operation finishes before this
    /// returns, so no work started under the old configuration can write after a reopen.
    /// The non-terminal sibling of `invalidateAndWait`.
    public func close() async throws {
        guard !isInvalidated else { return }
        guard !isOperationActiveOnCurrentTask() else {
            throw IndexEngineError(
                .configurationInvalid,
                code: "chartroom.session.close-inside-operation",
                recoverability: .needsConfiguration,
                summary: "The session cannot be closed from inside one of its own operations."
            )
        }
        lifecycleGeneration &+= 1
        pendingOpen?.task.cancel()
        pendingOpen = nil
        engine = nil
        storeURL = nil
        lastSearch = nil
        lastError = nil
        state = .unopened

        for control in activeSyncControls.values {
            control.stop()
        }
        await awaitQuiescence(includingOpens: false)
    }

    public func status(limit: Int = 1_000) async -> ChartroomStatus {
        guard !isInvalidated else {
            return ChartroomStatus(state: .invalidated, lastError: lastError)
        }
        activeOperationCount += 1
        defer { endOperation() }
        return await withOperationContext {
            await currentStatus(limit: limit)
        }
    }

    /// Permanently rejects new work and waits for every engine operation that crossed the
    /// session boundary before invalidation to finish.
    func invalidateAndWait() async {
        isInvalidated = true
        pendingOpen?.task.cancel()
        pendingOpen = nil
        state = .invalidated
        lastError = invalidatedError().summary

        await awaitQuiescence(includingOpens: true)

        engine = nil
        storeURL = nil
        lastSearch = nil
    }

    public func search(_ request: SearchRequest) async throws -> SearchResponse {
        try beginOperation()
        defer { endOperation() }
        return try await withOperationContext {
            let response = try await requiredEngine().search(request)
            lastSearch = response
            return response
        }
    }

    public func browseDocuments(_ request: DocumentBrowseRequest) async throws -> DocumentBrowseResponse {
        try beginOperation()
        defer { endOperation() }
        return try await withOperationContext {
            try await requiredEngine().browseDocuments(request)
        }
    }

    public func chunks(forDocument documentID: DocumentID) async throws -> [ChunkSummary] {
        try beginOperation()
        defer { endOperation() }
        return try await withOperationContext {
            try await requiredEngine().chunks(forDocument: documentID)
        }
    }

    public func ingestLocalSource(
        _ rootURL: URL,
        connectorID: ConnectorID = "local-files",
        cursorKey: String? = nil,
        options: LocalFileConnectorOptions = .init(),
        control: SyncControl? = nil,
        onProgress: SyncOrchestrator.ProgressHandler? = nil
    ) async throws -> SyncOutcome {
        try beginOperation()
        defer { endOperation() }
        return try await withOperationContext {
            let connector = LocalFileConnector(rootURL: rootURL, id: connectorID, options: options)
            let effectiveCursorKey = cursorKey ?? "\(connector.id.rawValue)|\(rootURL.standardizedFileURL.path)"
            return try await runSync(
                connector: connector,
                cursorKey: effectiveCursorKey,
                control: control,
                onProgress: onProgress
            )
        }
    }

    /// Runs a connector through Chartroom's cursor and ingestion policy. Hosts provide
    /// sources and presentation only; event ordering, cursor advancement, and cancellation
    /// remain owned by the core library.
    public func sync(
        connector: any SourceConnector,
        cursorKey: String,
        control: SyncControl? = nil,
        onProgress: SyncOrchestrator.ProgressHandler? = nil
    ) async throws -> SyncOutcome {
        try beginOperation()
        defer { endOperation() }
        return try await withOperationContext {
            try await runSync(
                connector: connector,
                cursorKey: cursorKey,
                control: control,
                onProgress: onProgress
            )
        }
    }

    /// Re-ingests explicitly supplied payloads while preserving the same cancellation and
    /// progress semantics as connector-based work. This is used for targeted retries after
    /// a recorded local-file failure.
    public func ingest(
        payloads: [SourcePayload],
        control: SyncControl? = nil,
        onProgress: SyncOrchestrator.ProgressHandler? = nil
    ) async throws -> IngestOutcome {
        try beginOperation()
        defer { endOperation() }
        return try await withOperationContext {
            try await withRegisteredSyncControl(control) { control in
                let orchestrator = SyncOrchestrator(engine: try requiredEngine(), cursorStore: cursorStore)
                return await orchestrator.ingest(payloads: payloads, control: control, onProgress: onProgress)
            }
        }
    }

    /// Re-embeds stored content the active model cannot read. See `IndexEngineClient.rebuildEmbeddings`.
    public func rebuildEmbeddings() async throws -> EmbeddingRebuildSummary {
        try beginOperation()
        defer { endOperation() }
        return try await withOperationContext {
            try await requiredEngine().rebuildEmbeddings()
        }
    }

    public func deleteDocuments(_ documentIDs: [DocumentID]) async throws -> DeletionSummary {
        try beginOperation()
        defer { endOperation() }
        return try await withOperationContext {
            try await requiredEngine().delete(.init(documentIDs: documentIDs))
        }
    }

    /// Best-effort compatibility projection. Use `diagnosticHistory(limit:)` when durable
    /// completeness matters.
    public func failures(limit: Int = 1_000) async throws -> [FailureSnapshot] {
        try await diagnosticHistory(limit: limit).failures
    }

    /// Best-effort compatibility projection. Use `diagnosticHistory(limit:)` for completeness.
    public func jobs(limit: Int = 1_000) async throws -> [JobSnapshot] {
        try await diagnosticHistory(limit: limit).jobs
    }

    public func diagnosticHistory(limit: Int = 1_000) async throws -> DiagnosticHistory {
        try beginOperation()
        defer { endOperation() }
        return try await withOperationContext {
            try await requiredEngine().diagnosticHistory(limit: limit)
        }
    }

    public func clearFailures(ids: Set<EngineID>? = nil) async throws {
        try beginOperation()
        defer { endOperation() }
        try await withOperationContext {
            try await requiredEngine().clearFailures(ids: ids)
        }
    }

    public func benchmark(queries: [String], iterations: Int = 5, limit: Int = 10) async throws -> SearchBenchmark.Report {
        try beginOperation()
        defer { endOperation() }
        return try await withOperationContext {
            try await SearchBenchmark.run(
                engine: requiredEngine(),
                queries: queries,
                iterations: iterations,
                limit: limit
            )
        }
    }

    public func diagnosticsBundle(limit: Int = 1_000) async throws -> DiagnosticsBundle {
        try beginOperation()
        defer { endOperation() }
        return try await withOperationContext {
            await DiagnosticsBundle.capture(
                from: try requiredEngine(),
                lastSearch: lastSearch,
                limit: limit
            )
        }
    }

    public func retrievalPipeline() throws -> RetrievalPipelineDescriptor {
        try requiredEngine().retrievalPipeline
    }

    private func runSync(
        connector: any SourceConnector,
        cursorKey: String,
        control: SyncControl?,
        onProgress: SyncOrchestrator.ProgressHandler?
    ) async throws -> SyncOutcome {
        try await withRegisteredSyncControl(control) { control in
            let acquired = await syncCoordinator.acquire(cursorKey)
            guard acquired else {
                // `acquire` returns false only when this task is already cancelled, and
                // cancellation is sticky, so the orchestrator's first check returns a
                // stopped outcome before any connector or cursor work — the fall-through
                // never runs a sync outside the per-key serialization.
                let orchestrator = SyncOrchestrator(engine: try requiredEngine(), cursorStore: cursorStore)
                return try await orchestrator.sync(
                    connector: connector,
                    cursorKey: cursorKey,
                    control: control,
                    onProgress: onProgress
                )
            }
            defer { syncCoordinator.release(cursorKey) }

            let orchestrator = SyncOrchestrator(engine: try requiredEngine(), cursorStore: cursorStore)
            return try await orchestrator.sync(
                connector: connector,
                cursorKey: cursorKey,
                control: control,
                onProgress: onProgress
            )
        }
    }

    /// Every sync runs under a control the session can reach: the caller's when supplied,
    /// a session-created one otherwise. `close()` stops whatever is registered, which is
    /// how in-flight work observes the lifecycle without the session owning its task.
    private func withRegisteredSyncControl<T>(
        _ control: SyncControl?,
        operation: (SyncControl) async throws -> T
    ) async rethrows -> T {
        let effective = control ?? SyncControl()
        let token = UUID()
        activeSyncControls[token] = effective
        defer { activeSyncControls.removeValue(forKey: token) }
        return try await operation(effective)
    }

    private func currentStatus(limit: Int = 1_000) async -> ChartroomStatus {
        guard let engine else {
            return ChartroomStatus(state: state, storeURL: storeURL, lastError: lastError)
        }

        let diagnosticHistory = await engine.diagnosticHistory(limit: limit)
        var status = await ChartroomStatus(
            state: state,
            storeURL: storeURL,
            snapshot: engine.snapshot(),
            health: engine.health(),
            modelStatus: engine.modelStatus(),
            jobs: diagnosticHistory.jobs,
            failures: diagnosticHistory.failures,
            lastError: lastError
        )
        status.diagnosticHistory = diagnosticHistory
        return status
    }

    private func currentStatus(
        validatingGeneration generation: UInt64,
        limit: Int = 1_000
    ) async throws -> ChartroomStatus {
        let status = await currentStatus(limit: limit)
        guard !isInvalidated else {
            throw invalidatedError()
        }
        guard generation == lifecycleGeneration else {
            throw CancellationError()
        }
        return status
    }

    private func beginOperation() throws {
        guard !isInvalidated else {
            throw invalidatedError()
        }
        activeOperationCount += 1
    }

    private func endOperation(isOpen: Bool = false) {
        precondition(activeOperationCount > 0)
        activeOperationCount -= 1
        if isOpen {
            precondition(activeOpenCount > 0)
            activeOpenCount -= 1
        }

        var remaining: [QuiescenceWaiter] = []
        for waiter in quiescenceWaiters {
            let satisfied = waiter.includesOpens
                ? activeOperationCount == 0
                : activeOperationCount - activeOpenCount == 0
            if satisfied {
                waiter.continuation.resume()
            } else {
                remaining.append(waiter)
            }
        }
        quiescenceWaiters = remaining
    }

    private func awaitQuiescence(includingOpens: Bool) async {
        let pending = includingOpens
            ? activeOperationCount
            : activeOperationCount - activeOpenCount
        guard pending > 0 else { return }
        await withCheckedContinuation { continuation in
            quiescenceWaiters.append(
                QuiescenceWaiter(includesOpens: includingOpens, continuation: continuation)
            )
        }
    }

    nonisolated func isOperationActiveOnCurrentTask() -> Bool {
        ChartroomSessionOperationContext.sessionID == operationContextID
    }

    private func withOperationContext<T>(
        _ operation: () async throws -> T
    ) async rethrows -> T {
        try await ChartroomSessionOperationContext.$sessionID.withValue(
            operationContextID,
            operation: operation
        )
    }

    private func requiredEngine() throws -> any IndexEngineClient {
        guard !isInvalidated else {
            throw invalidatedError()
        }
        guard let engine else {
            throw IndexEngineError(
                .configurationInvalid,
                code: "chartroom.session.not-open",
                recoverability: .needsConfiguration,
                summary: "Open the Chartroom session before running this command."
            )
        }
        return engine
    }

    private func invalidatedError() -> IndexEngineError {
        IndexEngineError(
            .configurationInvalid,
            code: "chartroom.session.invalidated",
            recoverability: .unrecoverable,
            summary: "This index was deleted and its session is no longer available."
        )
    }
}
