import ConnectorEngine
import Foundation
import IndexEngine
import SyncEngine

public enum ChartroomSessionState: String, Codable, Hashable, Sendable {
    case unopened
    case opening
    case ready
    case failed
}

public struct ChartroomStatus: Codable, Sendable {
    public var state: ChartroomSessionState
    public var storeURL: URL?
    public var snapshot: IndexEngineSnapshot?
    public var health: IndexHealthSnapshot?
    public var modelStatus: ModelStatusSnapshot?
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
        self.jobs = jobs
        self.failures = failures
        self.lastError = lastError
    }
}

public actor ChartroomSession {
    public typealias EngineFactory = @Sendable () async throws -> (any IndexEngineClient, URL?)

    private let engineFactory: EngineFactory
    private let cursorStore: any CursorStore
    private var state: ChartroomSessionState = .unopened
    private var engine: (any IndexEngineClient)?
    private var storeURL: URL?
    private var lastSearch: SearchResponse?
    private var lastError: String?

    public init(engineFactory: @escaping EngineFactory, cursorStore: any CursorStore) {
        self.engineFactory = engineFactory
        self.cursorStore = cursorStore
    }

    public func open() async throws -> ChartroomStatus {
        if state == .ready {
            return await status()
        }

        state = .opening
        lastError = nil

        do {
            let opened = try await engineFactory()
            engine = opened.0
            storeURL = opened.1
            state = .ready
            return await status()
        } catch {
            state = .failed
            lastError = String(describing: error)
            throw error
        }
    }

    /// Releases the current engine instance. The next `open()` rebuilds it through this
    /// session's factory, which lets a host apply a changed model configuration without
    /// constructing an engine outside the product command surface.
    public func close() {
        engine = nil
        storeURL = nil
        lastSearch = nil
        lastError = nil
        state = .unopened
    }

    public func status(limit: Int = 1_000) async -> ChartroomStatus {
        guard let engine else {
            return ChartroomStatus(state: state, storeURL: storeURL, lastError: lastError)
        }

        return await ChartroomStatus(
            state: state,
            storeURL: storeURL,
            snapshot: engine.snapshot(),
            health: engine.health(),
            modelStatus: engine.modelStatus(),
            jobs: engine.jobs(limit: limit),
            failures: engine.failures(limit: limit),
            lastError: lastError
        )
    }

    public func search(_ request: SearchRequest) async throws -> SearchResponse {
        let response = try await requiredEngine().search(request)
        lastSearch = response
        return response
    }

    public func browseDocuments(_ request: DocumentBrowseRequest) async throws -> DocumentBrowseResponse {
        try await requiredEngine().browseDocuments(request)
    }

    public func chunks(forDocument documentID: DocumentID) async throws -> [ChunkSummary] {
        try await requiredEngine().chunks(forDocument: documentID)
    }

    public func ingestLocalSource(
        _ rootURL: URL,
        connectorID: ConnectorID = "local-files",
        cursorKey: String? = nil,
        options: LocalFileConnectorOptions = .init(),
        control: SyncControl? = nil,
        onProgress: SyncOrchestrator.ProgressHandler? = nil
    ) async throws -> SyncOutcome {
        let connector = LocalFileConnector(rootURL: rootURL, id: connectorID, options: options)
        let effectiveCursorKey = cursorKey ?? "\(connector.id.rawValue)|\(rootURL.standardizedFileURL.path)"
        return try await sync(
            connector: connector,
            cursorKey: effectiveCursorKey,
            control: control,
            onProgress: onProgress
        )
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
        let orchestrator = SyncOrchestrator(engine: try requiredEngine(), cursorStore: cursorStore)
        return try await orchestrator.sync(
            connector: connector,
            cursorKey: cursorKey,
            control: control,
            onProgress: onProgress
        )
    }

    /// Re-ingests explicitly supplied payloads while preserving the same cancellation and
    /// progress semantics as connector-based work. This is used for targeted retries after
    /// a recorded local-file failure.
    public func ingest(
        payloads: [SourcePayload],
        control: SyncControl? = nil,
        onProgress: SyncOrchestrator.ProgressHandler? = nil
    ) async throws -> IngestOutcome {
        let orchestrator = SyncOrchestrator(engine: try requiredEngine(), cursorStore: cursorStore)
        return await orchestrator.ingest(payloads: payloads, control: control, onProgress: onProgress)
    }

    /// Re-embeds stored content the active model cannot read. See `IndexEngineClient.rebuildEmbeddings`.
    public func rebuildEmbeddings() async throws -> EmbeddingRebuildSummary {
        try await requiredEngine().rebuildEmbeddings()
    }

    public func deleteDocuments(_ documentIDs: [DocumentID]) async throws -> DeletionSummary {
        try await requiredEngine().delete(.init(documentIDs: documentIDs))
    }

    public func failures(limit: Int = 1_000) async throws -> [FailureSnapshot] {
        try await requiredEngine().failures(limit: limit)
    }

    public func jobs(limit: Int = 1_000) async throws -> [JobSnapshot] {
        try await requiredEngine().jobs(limit: limit)
    }

    public func clearFailures(ids: Set<EngineID>? = nil) async throws {
        try await requiredEngine().clearFailures(ids: ids)
    }

    public func benchmark(queries: [String], iterations: Int = 5, limit: Int = 10) async throws -> SearchBenchmark.Report {
        try await SearchBenchmark.run(engine: requiredEngine(), queries: queries, iterations: iterations, limit: limit)
    }

    public func diagnosticsBundle(limit: Int = 1_000) async throws -> DiagnosticsBundle {
        await DiagnosticsBundle.capture(from: try requiredEngine(), lastSearch: lastSearch, limit: limit)
    }

    public func retrievalPipeline() throws -> RetrievalPipelineDescriptor {
        try requiredEngine().retrievalPipeline
    }

    private func requiredEngine() throws -> any IndexEngineClient {
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
}
