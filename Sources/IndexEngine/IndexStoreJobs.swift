import Foundation

extension IndexStore {
    public func recordJob(_ job: JobSnapshot) throws {
        let statement = try db.prepare("""
        INSERT INTO jobs(id,state,kind,completed_unit_count,total_unit_count,message,updated_at)
        VALUES(?1,?2,?3,?4,?5,?6,?7)
        ON CONFLICT(id) DO UPDATE SET state=excluded.state,
          kind=excluded.kind,
          completed_unit_count=excluded.completed_unit_count,
          total_unit_count=excluded.total_unit_count,
          message=excluded.message,
          updated_at=excluded.updated_at
        """)
        statement.bind(1, job.id.rawValue)
        statement.bind(2, job.state.rawValue)
        statement.bind(3, job.kind.rawValue)
        statement.bind(4, job.completedUnitCount)
        if let totalUnitCount = job.totalUnitCount {
            statement.bind(5, totalUnitCount)
        } else {
            statement.bindNull(5)
        }
        statement.bind(6, job.message)
        statement.bind(7, Date.now.timeIntervalSince1970)
        try statement.step()
    }

    public func recordFailure(_ failure: FailureSnapshot) throws {
        let statement = try db.prepare("""
        INSERT OR REPLACE INTO failures(
          id,category,message,detail,source_id,document_id,source_uri,is_recoverable,recoverability,occurred_at
        ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)
        """)
        statement.bind(1, failure.id.rawValue)
        statement.bind(2, failure.category.rawValue)
        statement.bind(3, failure.message)
        statement.bind(4, failure.detail)
        if let sourceID = failure.sourceID {
            statement.bind(5, sourceID.rawValue)
        } else {
            statement.bindNull(5)
        }
        if let documentID = failure.documentID {
            statement.bind(6, documentID.rawValue)
        } else {
            statement.bindNull(6)
        }
        if let sourceURI = failure.sourceURI {
            statement.bind(7, sourceURI.absoluteString)
        } else {
            statement.bindNull(7)
        }
        statement.bind(8, failure.isRecoverable ? 1 : 0)
        statement.bind(9, failure.recoverability.rawValue)
        statement.bind(10, failure.occurredAt.timeIntervalSince1970)
        try statement.step()
    }

    public func jobSnapshots(limit: Int) throws -> [JobSnapshot] {
        guard limit > 0 else { return [] }
        let statement = try db.prepare("""
        SELECT id,state,kind,completed_unit_count,total_unit_count,message
        FROM jobs ORDER BY updated_at DESC LIMIT ?1
        """)
        statement.bind(1, limit)
        var jobs: [JobSnapshot] = []
        while try statement.step() {
            let state = JobSnapshot.State(rawValue: statement.text(1) ?? "") ?? .failed
            let kind = JobSnapshot.Kind(rawValue: statement.text(2) ?? "") ?? .ingest
            jobs.append(
                JobSnapshot(
                    id: EngineID(rawValue: statement.text(0) ?? ""),
                    state: state,
                    kind: kind,
                    completedUnitCount: statement.int(3),
                    totalUnitCount: statement.null(4) ? nil : statement.int(4),
                    message: statement.text(5) ?? ""
                )
            )
        }
        return jobs
    }

    public func recoverInterruptedJobs() throws {
        let statement = try db.prepare("""
        UPDATE jobs
        SET state = ?1,
            message = CASE
              WHEN message = '' THEN ?2
              ELSE message || ' - ' || ?2
            END,
            updated_at = ?3
        WHERE state IN (?4, ?5, ?6, ?7)
        """)
        statement.bind(1, JobSnapshot.State.failed.rawValue)
        statement.bind(2, "Interrupted before the index reopened")
        statement.bind(3, Date.now.timeIntervalSince1970)
        statement.bind(4, JobSnapshot.State.queued.rawValue)
        statement.bind(5, JobSnapshot.State.running.rawValue)
        statement.bind(6, JobSnapshot.State.committing.rawValue)
        statement.bind(7, JobSnapshot.State.recovering.rawValue)
        try statement.step()
    }

    /// Delete durable failure diagnostics. `ids == nil` clears every recorded failure;
    /// otherwise only the listed ones. Failures are diagnostics, not indexed content, so
    /// removing them never touches documents, chunks, or embeddings.
    public func deleteFailures(ids: Set<String>?) throws {
        guard let ids else {
            try db.prepare("DELETE FROM failures").step()
            return
        }
        guard !ids.isEmpty else { return }
        try db.transaction {
            let statement = try db.prepare("DELETE FROM failures WHERE id = ?1")
            for id in ids.sorted() {
                statement.reset()
                statement.bind(1, id)
                try statement.step()
            }
        }
    }

    public func failureSnapshots(limit: Int) throws -> [FailureSnapshot] {
        guard limit > 0 else { return [] }
        let statement = try db.prepare("""
        SELECT id,category,message,detail,source_id,document_id,source_uri,is_recoverable,recoverability,occurred_at
        FROM failures ORDER BY occurred_at DESC LIMIT ?1
        """)
        statement.bind(1, limit)
        var failures: [FailureSnapshot] = []
        while try statement.step() {
            let category = FailureSnapshot.Category(rawValue: statement.text(1) ?? "") ?? .storageFailure
            let recoverability = statement.text(8)
                .flatMap(IndexEngineError.Recoverability.init(rawValue:))
                ?? (statement.int(7) != 0 ? .retryable : .unrecoverable)
            failures.append(
                FailureSnapshot(
                    id: EngineID(rawValue: statement.text(0) ?? ""),
                    category: category,
                    message: statement.text(2) ?? "",
                    detail: statement.text(3) ?? "",
                    sourceID: statement.text(4).map(EngineID.init(rawValue:)),
                    documentID: statement.text(5).map(EngineID.init(rawValue:)),
                    sourceURI: statement.text(6).flatMap(URL.init(string:)),
                    recoverability: recoverability,
                    occurredAt: Date(timeIntervalSince1970: statement.double(9))
                )
            )
        }
        return failures
    }

    public func vectorBackendStatus() -> VectorStorageStatus {
        VectorStorageStatus(
            backendID: EngineID(rawValue: vectorBackendID),
            state: .ready,
            message: "SQLite FTS5 plus exact vector scan over active chunk embeddings"
        )
    }

    /// Probe whether the embedding provider can actually produce a query vector right now.
    ///
    /// `ModelStatusSnapshot` must report *observed* availability, not an assumption. A real
    /// CoreML model can fail to load (missing bundle, incompatible runtime) and a misconfigured
    /// provider can return the wrong dimension — both of which would silently break search. This
    /// is the one place the GUI asks "is the model usable?", so it embeds a minimal query and
    /// reports the true outcome: available only when a correctly sized vector comes back,
    /// otherwise unavailable with the provider's own reason.
    ///
    /// A confirmed-available result is cached for the store's lifetime: the probe is a real
    /// model inference, far too expensive to repeat on every status poll, and a loaded model
    /// does not unload. Failures are never cached, so a transient problem can recover.
    public func embeddingProviderStatus() async -> (isAvailable: Bool, message: String) {
        if let cachedEmbeddingProviderStatus {
            return cachedEmbeddingProviderStatus
        }
        do {
            let probe = try await embedder.embedQuery("probe")
            guard probe.count == dimension else {
                return (false, "Embedding provider returned \(probe.count)-dimension vectors; the index expects \(dimension).")
            }
            let status = (true, "Embedding provider is available.")
            cachedEmbeddingProviderStatus = status
            return status
        } catch {
            return (false, String(describing: error))
        }
    }
}
