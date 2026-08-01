import ConnectorEngine
import Foundation
import IndexEngine

/// A progress snapshot emitted around each per-payload ingest call.
public struct SyncProgressUpdate: Sendable {
    public var totalPayloads: Int
    public var completedPayloads: Int
    public var accepted: Int
    public var failed: Int
    public var currentItem: String?

    public init(totalPayloads: Int, completedPayloads: Int, accepted: Int, failed: Int, currentItem: String?) {
        self.totalPayloads = totalPayloads
        self.completedPayloads = completedPayloads
        self.accepted = accepted
        self.failed = failed
        self.currentItem = currentItem
    }
}

/// The result of driving a set of payloads through the engine one at a time.
public struct IngestOutcome: Sendable {
    public var accepted: Int
    public var failed: Int
    public var stopped: Bool
    public var startedAt: Date
    public var finishedAt: Date
}

/// The result of one full connector sync: what changed, what the engine accepted, and
/// whether the cursor advanced.
public struct SyncOutcome: Sendable {
    /// Upsert payloads the connector emitted this sync.
    public var payloadCount: Int
    public var accepted: Int
    public var ingestFailed: Int
    public var deletedCount: Int
    public var deleteFailed: Int
    /// True when control stopped the run or task cancellation was observed at a sync checkpoint.
    public var stopped: Bool
    /// The checkpoint the store advanced to, or nil when the cursor was held back.
    public var newCursor: SourceCursor?
    /// False when the connector only emitted a checkpoint for the current cursor.
    public var hadChanges: Bool
    /// Every event the connector emitted, in emission order.
    public var events: [SourceEvent]
    /// Unique paths the connector reported as unreadable, sorted.
    public var unreadablePaths: [String]
    /// Distinct still-indexed documents that lost read permission and are kept in the index.
    public var unreadableKeptCount: Int
    public var startedAt: Date
    public var finishedAt: Date

    public var totalFailed: Int { ingestFailed + deleteFailed }
}

/// The connector→engine bridge: consumes a connector's change stream, applies deletions and
/// ingests through the engine facade, and owns the cursor advancement policy. Neither
/// ConnectorEngine (events) nor IndexEngine (payloads) can host this alone — the correctness
/// of incremental sync lives in the ordering and cursor rules here.
public struct SyncOrchestrator: Sendable {
    public typealias ProgressHandler = @Sendable (SyncProgressUpdate) async -> Void

    private enum CollectionResult: Sendable {
        case events([SourceEvent])
        case stopped
    }

    private let engine: any IndexEngineClient
    private let cursorStore: any CursorStore

    public init(engine: any IndexEngineClient, cursorStore: any CursorStore) {
        self.engine = engine
        self.cursorStore = cursorStore
    }

    /// Run one incremental sync. Deletions apply before ingests so a move never leaves both
    /// the old and new document visible. The cursor advances only after every emitted
    /// mutation is durably accepted; failed payloads stay behind the old cursor so retrying
    /// the source does not skip them until they change.
    public func sync(
        connector: any SourceConnector,
        cursorKey: String,
        control: SyncControl? = nil,
        onProgress: ProgressHandler? = nil
    ) async throws -> SyncOutcome {
        let startedAt = Date.now

        if Task.isCancelled || control?.isStopping == true {
            return stoppedOutcome(startedAt: startedAt)
        }
        guard try await racingStop(control, { try await connector.validate() }) != nil else {
            return stoppedOutcome(startedAt: startedAt)
        }
        if Task.isCancelled || control?.isStopping == true {
            return stoppedOutcome(startedAt: startedAt)
        }
        let cursor = cursorStore.cursor(forKey: cursorKey)
        guard let stream = try await racingStop(control, { try await connector.changes(since: cursor) }) else {
            return stoppedOutcome(startedAt: startedAt)
        }
        if Task.isCancelled || control?.isStopping == true {
            return stoppedOutcome(startedAt: startedAt)
        }

        let collectedEvents: [SourceEvent]
        do {
            switch try await collectEvents(from: stream, control: control) {
            case let .events(events):
                collectedEvents = events
            case .stopped:
                return stoppedOutcome(startedAt: startedAt)
            }
        } catch let error as CancellationError {
            guard Task.isCancelled || control?.isStopping == true else {
                throw error
            }
            return stoppedOutcome(startedAt: startedAt)
        }

        var payloads: [SourcePayload] = []
        var deletedDocumentIDs: [DocumentID] = []
        var unreadableDocumentIDs: [DocumentID] = []
        var unreadablePaths: [String] = []
        var events: [SourceEvent] = []
        var pendingCursor: SourceCursor?

        for event in collectedEvents {
            events.append(event)

            switch event {
            case let .upsert(payload):
                payloads.append(payload)
            case let .delete(documentID):
                deletedDocumentIDs.append(documentID)
            case let .move(documentID, _):
                deletedDocumentIDs.append(documentID)
            case let .permissionChanged(documentID):
                unreadableDocumentIDs.append(documentID)
            case let .pathUnavailable(path, _):
                unreadablePaths.append(path)
            case let .checkpoint(cursor):
                pendingCursor = cursor
            default:
                break
            }
        }

        let uniqueDeletedDocumentIDs = Array(Set(deletedDocumentIDs)).sorted { $0.rawValue < $1.rawValue }
        let uniqueUnreadablePaths = Array(Set(unreadablePaths)).sorted()
        let unreadableKeptCount = Set(unreadableDocumentIDs).count

        if let control {
            await control.waitWhilePaused()
        }
        if Task.isCancelled || control?.isStopping == true {
            return stoppedOutcome(
                startedAt: startedAt,
                events: events,
                payloadCount: payloads.count,
                hadChanges: !payloads.isEmpty || !uniqueDeletedDocumentIDs.isEmpty,
                unreadablePaths: uniqueUnreadablePaths,
                unreadableKeptCount: unreadableKeptCount
            )
        }

        guard !payloads.isEmpty || !uniqueDeletedDocumentIDs.isEmpty else {
            var newCursor: SourceCursor?
            if let pendingCursor {
                cursorStore.setCursor(pendingCursor, forKey: cursorKey)
                newCursor = pendingCursor
            }
            return SyncOutcome(
                payloadCount: 0,
                accepted: 0,
                ingestFailed: 0,
                deletedCount: 0,
                deleteFailed: 0,
                stopped: false,
                newCursor: newCursor,
                hadChanges: false,
                events: events,
                unreadablePaths: uniqueUnreadablePaths,
                unreadableKeptCount: unreadableKeptCount,
                startedAt: startedAt,
                finishedAt: .now
            )
        }

        let deletionSummary: DeletionSummary?
        if uniqueDeletedDocumentIDs.isEmpty {
            deletionSummary = nil
        } else {
            deletionSummary = try await engine.delete(.init(documentIDs: uniqueDeletedDocumentIDs))
        }

        let ingestOutcome = await ingest(payloads: payloads, control: control, onProgress: onProgress)

        let deleteFailed = deletionSummary?.failedCount ?? 0
        let totalFailed = ingestOutcome.failed + deleteFailed

        var newCursor: SourceCursor?
        if !ingestOutcome.stopped, totalFailed == 0, let pendingCursor {
            cursorStore.setCursor(pendingCursor, forKey: cursorKey)
            newCursor = pendingCursor
        }

        return SyncOutcome(
            payloadCount: payloads.count,
            accepted: ingestOutcome.accepted,
            ingestFailed: ingestOutcome.failed,
            deletedCount: deletionSummary?.deletedCount ?? 0,
            deleteFailed: deleteFailed,
            stopped: ingestOutcome.stopped,
            newCursor: newCursor,
            hadChanges: true,
            events: events,
            unreadablePaths: uniqueUnreadablePaths,
            unreadableKeptCount: unreadableKeptCount,
            startedAt: startedAt,
            finishedAt: .now
        )
    }

    /// Drive payloads through the engine one at a time so progress, pause, and stop each have
    /// a natural checkpoint. Every payload is its own ingest call, isolating per-item failures;
    /// a thrown ingest counts as one failure rather than aborting the run.
    public func ingest(
        payloads: [SourcePayload],
        control: SyncControl? = nil,
        onProgress: ProgressHandler? = nil
    ) async -> IngestOutcome {
        let startedAt = Date.now
        var accepted = 0
        var failed = 0
        var completed = 0
        var stopped = false

        for payload in payloads {
            if Task.isCancelled {
                stopped = true
                break
            }
            if let control {
                await control.waitWhilePaused()
                if Task.isCancelled || control.isStopping {
                    stopped = true
                    break
                }
            }

            await onProgress?(SyncProgressUpdate(
                totalPayloads: payloads.count,
                completedPayloads: completed,
                accepted: accepted,
                failed: failed,
                currentItem: payload.displayName
            ))

            do {
                let summary = try await engine.ingest(.init(payloads: [payload]))
                accepted += summary.acceptedCount
                failed += summary.failedCount
            } catch {
                failed += 1
            }
            completed += 1

            await onProgress?(SyncProgressUpdate(
                totalPayloads: payloads.count,
                completedPayloads: completed,
                accepted: accepted,
                failed: failed,
                currentItem: payload.displayName
            ))
        }

        if !stopped, Task.isCancelled || control?.isStopping == true {
            stopped = true
        }

        return IngestOutcome(
            accepted: accepted,
            failed: failed,
            stopped: stopped,
            startedAt: startedAt,
            finishedAt: .now
        )
    }

    /// Runs one connector call racing the stop signal, returning nil when stop wins.
    ///
    /// Not a task group: a group awaits its children after cancellation, so a connector
    /// that ignores cancellation while hung in `validate()` or `changes(since:)` would
    /// hold the group — and the stop it was meant to honor — hostage. The losing task is
    /// cancelled and abandoned instead; it finishes whenever the connector returns and
    /// its result is discarded. A leaked task bounded by connector behavior is the price
    /// of `stop()` always meaning stop.
    private func racingStop<T: Sendable>(
        _ control: SyncControl?,
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T? {
        guard let control else { return try await operation() }

        let (stream, continuation) = AsyncStream<Result<T, Error>?>.makeStream()
        let operationTask = Task {
            do {
                continuation.yield(.success(try await operation()))
            } catch {
                continuation.yield(.failure(error))
            }
        }
        let stopTask = Task {
            await control.waitUntilStopping()
            continuation.yield(nil)
        }
        defer {
            operationTask.cancel()
            stopTask.cancel()
            continuation.finish()
        }

        for await first in stream {
            switch first {
            case let .success(value):
                return value
            case let .failure(error):
                throw error
            case nil:
                return nil
            }
        }
        return nil
    }

    private func collectEvents(
        from stream: AsyncThrowingStream<SourceEvent, Error>,
        control: SyncControl?
    ) async throws -> CollectionResult {
        guard let control else {
            return try await Self.consume(stream)
        }

        return try await withThrowingTaskGroup(of: CollectionResult.self) { group in
            group.addTask {
                try await Self.consume(stream)
            }
            group.addTask {
                await control.waitUntilStopping()
                return .stopped
            }

            guard let first = try await group.next() else {
                return .stopped
            }
            group.cancelAll()
            return first
        }
    }

    private static func consume(
        _ stream: AsyncThrowingStream<SourceEvent, Error>
    ) async throws -> CollectionResult {
        var events: [SourceEvent] = []
        for try await event in stream {
            events.append(event)
        }
        return Task.isCancelled ? .stopped : .events(events)
    }

    private func stoppedOutcome(
        startedAt: Date,
        events: [SourceEvent] = [],
        payloadCount: Int = 0,
        hadChanges: Bool = false,
        unreadablePaths: [String] = [],
        unreadableKeptCount: Int = 0
    ) -> SyncOutcome {
        SyncOutcome(
            payloadCount: payloadCount,
            accepted: 0,
            ingestFailed: 0,
            deletedCount: 0,
            deleteFailed: 0,
            stopped: true,
            newCursor: nil,
            hadChanges: hadChanges,
            events: events,
            unreadablePaths: unreadablePaths,
            unreadableKeptCount: unreadableKeptCount,
            startedAt: startedAt,
            finishedAt: .now
        )
    }
}
