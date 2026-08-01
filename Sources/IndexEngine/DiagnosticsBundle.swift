import Foundation

public enum DiagnosticsExportMode: Sendable, Equatable {
    /// Removes filesystem paths, document/query content, source identifiers, and arbitrary error
    /// text. This is the default because exported bundles are commonly attached to support cases.
    case safeForSharing
    /// Writes the complete in-memory payload. Callers must make the sensitivity visible to users.
    case includeSensitiveData
}

/// An engine diagnostic bundle — every snapshot a client reads, as pretty-printed JSON files
/// in a directory. Hosts decide how to package the directory (zip, tar, share sheet).
public struct DiagnosticsBundle: Sendable {
    public var snapshot: IndexEngineSnapshot
    public var health: IndexHealthSnapshot
    public var modelStatus: ModelStatusSnapshot
    public var diagnosticHistory: DiagnosticHistory
    public var jobs: [JobSnapshot] {
        get { diagnosticHistory.jobs }
        set { diagnosticHistory.jobs = newValue }
    }
    public var failures: [FailureSnapshot] {
        get { diagnosticHistory.failures }
        set { diagnosticHistory.failures = newValue }
    }
    public var lastSearch: SearchResponse?

    /// Compatibility initializer for callers that predate history-completeness reporting.
    public init(
        snapshot: IndexEngineSnapshot,
        health: IndexHealthSnapshot,
        modelStatus: ModelStatusSnapshot,
        jobs: [JobSnapshot],
        failures: [FailureSnapshot],
        lastSearch: SearchResponse? = nil
    ) {
        self.snapshot = snapshot
        self.health = health
        self.modelStatus = modelStatus
        self.diagnosticHistory = DiagnosticHistory(
            jobs: jobs,
            failures: failures,
            jobsAvailability: .notSupported,
            failuresAvailability: .notSupported
        )
        self.lastSearch = lastSearch
    }

    public init(
        snapshot: IndexEngineSnapshot,
        health: IndexHealthSnapshot,
        modelStatus: ModelStatusSnapshot,
        diagnosticHistory: DiagnosticHistory,
        lastSearch: SearchResponse? = nil
    ) {
        self.snapshot = snapshot
        self.health = health
        self.modelStatus = modelStatus
        self.diagnosticHistory = diagnosticHistory
        self.lastSearch = lastSearch
    }

    /// Capture a fresh bundle from the engine. `lastSearch` is passed in because the engine
    /// does not retain search responses; the client owns the most recent one.
    public static func capture(
        from engine: any IndexEngineClient,
        lastSearch: SearchResponse? = nil,
        limit: Int
    ) async -> DiagnosticsBundle {
        let snapshot = await engine.snapshot()
        let health = await engine.health()
        let modelStatus = await engine.modelStatus()
        let diagnosticHistory = await engine.diagnosticHistory(limit: limit)
        return DiagnosticsBundle(
            snapshot: snapshot,
            health: health,
            modelStatus: modelStatus,
            diagnosticHistory: diagnosticHistory,
            lastSearch: lastSearch
        )
    }

    /// Write a share-safe projection into `directory` (which must already exist).
    public func write(to directory: URL) throws {
        try write(to: directory, mode: .safeForSharing)
    }

    /// Write the bundle's JSON files using an explicit sensitivity policy.
    public func write(to directory: URL, mode: DiagnosticsExportMode) throws {
        try write(
            to: directory,
            mode: mode,
            fileWriter: { data, url in
                try data.write(to: url, options: .atomic)
            }
        )
    }

    /// Internal I/O seam used to prove that, once cleanup succeeds, a failed safe rewrite cannot
    /// strand payloads from a prior sensitive export. Projection and encoding finish before this
    /// method mutates the destination; existing export artifacts are then cleared before the first
    /// new write.
    func write(
        to directory: URL,
        mode: DiagnosticsExportMode,
        fileWriter: (Data, URL) throws -> Void
    ) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        var files: [(name: String, data: Data)] = []
        func append(_ value: some Encodable, as name: String) throws {
            files.append((name, try encoder.encode(value)))
        }

        let exportedSnapshot: IndexEngineSnapshot
        let exportedHealth: IndexHealthSnapshot
        let exportedModelStatus: ModelStatusSnapshot
        let exportedJobs: [JobSnapshot]
        let exportedFailures: [FailureSnapshot]
        switch mode {
        case .safeForSharing:
            var safeSnapshot = snapshot
            safeSnapshot.storeURL = nil
            safeSnapshot.policyStates = Self.safePolicyStates(safeSnapshot.policyStates)
            exportedSnapshot = safeSnapshot
            exportedHealth = Self.safeHealth(health)
            exportedModelStatus = Self.safeModelStatus(modelStatus)
            exportedJobs = Self.safeJobs(jobs)
            exportedFailures = Self.safeFailures(failures)
        case .includeSensitiveData:
            exportedSnapshot = snapshot
            exportedHealth = health
            exportedModelStatus = modelStatus
            exportedJobs = jobs
            exportedFailures = failures
        }

        try append(exportedSnapshot, as: "snapshot.json")
        try append(exportedHealth, as: "health.json")
        try append(exportedModelStatus, as: "model-status.json")
        try append(exportedJobs, as: "jobs.json")
        try append(exportedFailures, as: "failures.json")
        try append(
            HistoryAvailabilityExport(
                jobs: diagnosticHistory.jobsAvailability,
                failures: diagnosticHistory.failuresAvailability
            ),
            as: "history-availability.json"
        )
        if mode == .includeSensitiveData, let lastSearch {
            try append(lastSearch, as: "last-search.json")
        }

        let fileManager = FileManager.default
        var cleanupFailures: [String] = []
        for name in Self.exportFileNames {
            let url = directory.appendingPathComponent(name)
            do {
                let attributes = try fileManager.attributesOfItem(atPath: url.path)
                let type = attributes[.type] as? FileAttributeType
                guard type == .typeRegular || type == .typeSymbolicLink else {
                    cleanupFailures.append(name)
                    continue
                }
                try fileManager.removeItem(at: url)
            } catch let error as CocoaError
                where error.code == .fileNoSuchFile || error.code == .fileReadNoSuchFile
            {
                continue
            } catch {
                cleanupFailures.append(name)
            }
        }
        guard cleanupFailures.isEmpty else {
            throw DiagnosticsExportCleanupError(artifactNames: cleanupFailures)
        }
        for file in files {
            try fileWriter(file.data, directory.appendingPathComponent(file.name))
        }
    }

    private static let exportFileNames = [
        "snapshot.json",
        "health.json",
        "model-status.json",
        "jobs.json",
        "failures.json",
        "history-availability.json",
        "last-search.json",
    ]

    private static func safeModelStatus(_ status: ModelStatusSnapshot) -> ModelStatusSnapshot {
        let message: String
        if !status.isModelBacked {
            message = "The configured embedding provider is not model-backed."
        } else if status.isAvailable {
            message = "The embedding provider is available."
        } else {
            message = "The embedding provider is unavailable."
        }
        return ModelStatusSnapshot(
            modelID: status.modelID,
            embeddingSpaceID: status.embeddingSpaceID,
            dimension: status.dimension,
            isAvailable: status.isAvailable,
            isModelBacked: status.isModelBacked,
            message: message
        )
    }

    private static func safeHealth(_ health: IndexHealthSnapshot) -> IndexHealthSnapshot {
        var safe = health
        safe.policyStates = safePolicyStates(health.policyStates)
        if var vectorStatus = safe.vectorBackendStatus {
            vectorStatus.message = ""
            safe.vectorBackendStatus = vectorStatus
        }
        return safe
    }

    private static func safePolicyStates(
        _ states: [PolicyResolution]
    ) -> [PolicyResolution] {
        states.map { state in
            PolicyResolution(
                policyID: state.policyID,
                state: state.state,
                missingComponents: state.missingComponents,
                message: ""
            )
        }
    }

    private static func safeJobs(_ jobs: [JobSnapshot]) -> [JobSnapshot] {
        jobs.enumerated().map { index, job in
            JobSnapshot(
                id: EngineID(rawValue: "job-\(index + 1)"),
                state: job.state,
                kind: job.kind,
                completedUnitCount: job.completedUnitCount,
                totalUnitCount: job.totalUnitCount,
                message: "Recorded \(job.kind.rawValue) job."
            )
        }
    }

    private static func safeFailures(_ failures: [FailureSnapshot]) -> [FailureSnapshot] {
        failures.enumerated().map { index, failure in
            FailureSnapshot(
                id: EngineID(rawValue: "failure-\(index + 1)"),
                category: failure.category,
                message: "Recorded \(failure.category.rawValue) failure.",
                detail: "",
                recoverability: failure.recoverability,
                occurredAt: failure.occurredAt
            )
        }
    }
}

private struct HistoryAvailabilityExport: Encodable {
    var jobs: DiagnosticHistoryAvailability
    var failures: DiagnosticHistoryAvailability
}

private struct DiagnosticsExportCleanupError: Error, CustomStringConvertible {
    var artifactNames: [String]

    var description: String {
        "Could not safely replace existing diagnostic artifacts: \(artifactNames.joined(separator: ", "))"
    }
}
