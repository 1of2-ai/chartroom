import ConnectorEngine
import Foundation
import IndexEngine
import SyncEngine

/// A durable, independently searchable local index owned by a Chartroom workspace.
public struct ChartroomIndex: Codable, Hashable, Identifiable, Sendable {
    public var id: UUID
    public var name: String
    public var isSearchEnabled: Bool
    public var storeURL: URL

    public init(
        id: UUID = UUID(),
        name: String,
        isSearchEnabled: Bool = true,
        storeURL: URL
    ) {
        self.id = id
        self.name = name
        self.isSearchEnabled = isSearchEnabled
        self.storeURL = storeURL
    }
}

/// A search result paired with the index that produced it.
public struct ChartroomWorkspaceSearchResult: Codable, Hashable, Identifiable, Sendable {
    public struct ID: Codable, Hashable, Sendable {
        public var indexID: UUID
        public var resultID: EngineID

        public init(indexID: UUID, resultID: EngineID) {
            self.indexID = indexID
            self.resultID = resultID
        }
    }

    public var index: ChartroomIndex
    public var result: SearchResultSnapshot
    public var rank: Int

    public var id: ID {
        ID(indexID: index.id, resultID: result.id)
    }

    public init(index: ChartroomIndex, result: SearchResultSnapshot, rank: Int) {
        self.index = index
        self.result = result
        self.rank = rank
    }
}

/// The outcome for one enabled index in a workspace-wide search.
public struct ChartroomWorkspaceSearchStatus: Codable, Hashable, Identifiable, Sendable {
    public var index: ChartroomIndex
    public var resultCount: Int
    public var diagnostics: SearchDiagnostics?
    public var errorMessage: String?

    public var id: UUID { index.id }

    public init(
        index: ChartroomIndex,
        resultCount: Int,
        diagnostics: SearchDiagnostics? = nil,
        errorMessage: String? = nil
    ) {
        self.index = index
        self.resultCount = resultCount
        self.diagnostics = diagnostics
        self.errorMessage = errorMessage
    }
}

/// The combined response from every index currently included in a workspace search.
public struct ChartroomWorkspaceSearchResponse: Codable, Sendable, Equatable {
    /// How the merged results were ordered, and therefore how far the ordering can be trusted.
    ///
    /// Vector search reports an absolute similarity; fusion reduces it to a per-store rank. Two
    /// stores' rank scores are not the same quantity — a document that hit three channels in one
    /// store outranks a better document that hit one channel in another — so the workspace states
    /// which quantity it actually sorted on rather than leaving callers to assume.
    public enum Ordering: String, Codable, Sendable {
        /// Ordered by absolute cosine similarity. Valid across stores: every merged result carried
        /// a similarity and every one came from the same embedding space.
        case similarity
        /// Ordered by per-store fusion rank, the only signal available. Comparable *within* a
        /// store; across stores it is an approximation.
        case fusionRank
    }

    public var query: String
    public var mode: RetrievalMode
    /// Results with real evidence behind them, best first.
    public var results: [ChartroomWorkspaceSearchResult]
    /// Results whose only evidence is a below-threshold vector similarity. Vector search returns
    /// its top-N unconditionally, so this is the tail it always produces — kept so callers can
    /// offer it deliberately, and excluded from `results` so "no matches" can be told the truth.
    public var weakResults: [ChartroomWorkspaceSearchResult]
    public var statuses: [ChartroomWorkspaceSearchStatus]
    public var ordering: Ordering

    public init(
        query: String,
        mode: RetrievalMode,
        results: [ChartroomWorkspaceSearchResult],
        weakResults: [ChartroomWorkspaceSearchResult] = [],
        statuses: [ChartroomWorkspaceSearchStatus],
        ordering: Ordering = .fusionRank
    ) {
        self.query = query
        self.mode = mode
        self.results = results
        self.weakResults = weakResults
        self.statuses = statuses
        self.ordering = ordering
    }

    /// Every result the search produced, strong first. Selection, inspection, and any other
    /// lookup by ID must search this rather than `results` — a weak row a caller chose to display
    /// is still selectable, and looking it up in `results` alone would silently fail to find it.
    public var allResults: [ChartroomWorkspaceSearchResult] {
        results + weakResults
    }
}

/// Product-level coordinator for multiple durable indices.
///
/// `IndexEngine` remains a single-store retrieval primitive. This actor owns the catalog,
/// creates one session per index, namespaces connector cursors, and aggregates only the
/// indices users have included in search.
public actor ChartroomWorkspace {
    public typealias EngineFactory = @Sendable (ChartroomIndex) async throws -> (any IndexEngineClient, URL?)

    private let catalogURL: URL
    private let storesDirectory: URL
    private let bootstrapIndexes: [ChartroomIndex]
    private let engineFactory: EngineFactory
    private let cursorStore: any CursorStore

    private var loaded = false
    private var storedIndexes: [ChartroomIndex] = []
    private var sessions: [UUID: ChartroomSession] = [:]

    /// - Parameters:
    ///   - catalogURL: Durable JSON catalog location chosen by the host application.
    ///   - storesDirectory: Parent directory for stores created after initialization.
    ///   - bootstrapIndexes: Used only when no catalog exists, for example to adopt a legacy store.
    ///   - engineFactory: Host composition for the concrete embedder and extractors.
    ///   - cursorStore: Cursor storage shared by all sessions; the workspace namespaces keys per index.
    public init(
        catalogURL: URL,
        storesDirectory: URL,
        bootstrapIndexes: [ChartroomIndex] = [],
        engineFactory: @escaping EngineFactory,
        cursorStore: any CursorStore
    ) {
        self.catalogURL = catalogURL
        self.storesDirectory = storesDirectory
        self.bootstrapIndexes = bootstrapIndexes
        self.engineFactory = engineFactory
        self.cursorStore = cursorStore
    }

    /// The stored order is preserved for source-list presentation.
    public func indexes() throws -> [ChartroomIndex] {
        try loadCatalogIfNeeded()
        return storedIndexes
    }

    /// Creates an empty, enabled index with its own future SQLite store location.
    public func createIndex(named name: String) throws -> ChartroomIndex {
        try loadCatalogIfNeeded()
        let name = try validatedName(name, excluding: nil)
        let index = ChartroomIndex(
            name: name,
            storeURL: storesDirectory
                .appending(path: UUID().uuidString, directoryHint: .isDirectory)
                .appending(path: "IndexEngine.sqlite")
        )

        let directory = index.storeURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            storedIndexes.append(index)
            try persistCatalog()
        } catch {
            storedIndexes.removeLast()
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
        return index
    }

    /// Updates inclusion immediately; clients use this state to decide which indices global search queries.
    public func setSearchEnabled(_ isSearchEnabled: Bool, for indexID: UUID) throws {
        try loadCatalogIfNeeded()
        guard let position = storedIndexes.firstIndex(where: { $0.id == indexID }) else {
            throw unavailableIndexError(indexID)
        }
        storedIndexes[position].isSearchEnabled = isSearchEnabled
        try persistCatalog()
    }

    /// Renames an index without moving its durable store or invalidating its session.
    public func renameIndex(_ indexID: UUID, to name: String) throws {
        try loadCatalogIfNeeded()
        guard let position = storedIndexes.firstIndex(where: { $0.id == indexID }) else {
            throw unavailableIndexError(indexID)
        }
        storedIndexes[position].name = try validatedName(name, excluding: indexID)
        try persistCatalog()
    }

    /// Removes an index from the catalog and deletes its durable store.
    ///
    /// Destructive and not recoverable: callers are expected to have confirmed with the user.
    /// The catalog is updated first, so a failure to delete the files leaves an orphaned store on
    /// disk rather than a catalog entry pointing at a store that is partially gone.
    @discardableResult
    public func deleteIndex(_ indexID: UUID) throws -> ChartroomIndex {
        try loadCatalogIfNeeded()
        guard let position = storedIndexes.firstIndex(where: { $0.id == indexID }) else {
            throw unavailableIndexError(indexID)
        }

        let index = storedIndexes[position]
        storedIndexes.remove(at: position)
        do {
            try persistCatalog()
        } catch {
            storedIndexes.insert(index, at: position)
            throw error
        }

        sessions[indexID] = nil
        removeStoreFiles(for: index)
        return index
    }

    /// Deletes the store and the SQLite sidecars that travel with it.
    ///
    /// The containing directory is removed only when the workspace created it — one directory per
    /// index under `storesDirectory`. An adopted store such as the legacy bootstrap index lives
    /// beside the catalog itself, and deleting *that* parent would take the whole workspace with it.
    private func removeStoreFiles(for index: ChartroomIndex) {
        let store = index.storeURL
        for suffix in ["", "-wal", "-shm"] {
            let url = suffix.isEmpty
                ? store
                : store.deletingLastPathComponent()
                    .appending(path: store.lastPathComponent + suffix)
            try? FileManager.default.removeItem(at: url)
        }

        let directory = store.deletingLastPathComponent()
        guard directory.deletingLastPathComponent().standardizedFileURL == storesDirectory.standardizedFileURL else {
            return
        }
        try? FileManager.default.removeItem(at: directory)
    }

    /// Re-embeds content the active model cannot read, restoring an index orphaned by a model
    /// change. Rebuilds from stored text, so it neither re-reads sources nor needs them to exist.
    @discardableResult
    public func rebuildIndex(_ indexID: UUID) async throws -> EmbeddingRebuildSummary {
        let session = try session(for: indexID)
        _ = try await session.open()
        return try await session.rebuildEmbeddings()
    }

    /// Returns the index's lazily created session. The session is retained so opening, browsing,
    /// ingestion, and search all address the same engine instance.
    public func session(for indexID: UUID) throws -> ChartroomSession {
        try loadCatalogIfNeeded()
        guard let index = storedIndexes.first(where: { $0.id == indexID }) else {
            throw unavailableIndexError(indexID)
        }
        if let session = sessions[indexID] {
            return session
        }

        let factory = engineFactory
        let session = ChartroomSession(
            engineFactory: { try await factory(index) },
            cursorStore: NamespacedCursorStore(base: cursorStore, namespace: indexID.uuidString)
        )
        sessions[indexID] = session
        return session
    }

    /// Searches all enabled indexes. One index failing to open or search never suppresses
    /// successful results from the others; the per-index failure remains visible in `statuses`.
    public func search(_ request: SearchRequest) async throws -> ChartroomWorkspaceSearchResponse {
        try loadCatalogIfNeeded()
        let enabledIndexes = storedIndexes.filter(\.isSearchEnabled)
        let indexedSessions = try enabledIndexes.map { index in
            (index, try session(for: index.id))
        }

        let outcomes = await withTaskGroup(of: SearchOutcome.self) { group in
            for (index, session) in indexedSessions {
                group.addTask {
                    do {
                        _ = try await session.open()
                        return .success(index, try await session.search(request))
                    } catch {
                        return .failure(index, Self.searchErrorMessage(error))
                    }
                }
            }

            var outcomes: [SearchOutcome] = []
            for await outcome in group {
                outcomes.append(outcome)
            }
            return outcomes
        }

        var responses: [UUID: SearchResponse] = [:]
        var errors: [UUID: String] = [:]
        for outcome in outcomes {
            switch outcome {
            case let .success(index, response):
                responses[index.id] = response
            case let .failure(index, message):
                errors[index.id] = message
            }
        }

        let statuses = enabledIndexes.map { index in
            if let response = responses[index.id] {
                ChartroomWorkspaceSearchStatus(
                    index: index,
                    // Real answers only, matching what the merged `results` will contain.
                    resultCount: response.strongResults.count,
                    diagnostics: response.diagnostics
                )
            } else {
                ChartroomWorkspaceSearchStatus(
                    index: index,
                    resultCount: 0,
                    errorMessage: errors[index.id] ?? "The index could not be searched."
                )
            }
        }

        // Partition per index using the engine's own definition, then merge. The weak tail is not a
        // worse answer, it is the absence of one, and letting it occupy result slots is what made a
        // nonsense query return the corpus.
        func merge(
            _ selector: (SearchResponse) -> [SearchResultSnapshot]
        ) -> [ChartroomWorkspaceSearchResult] {
            enabledIndexes.flatMap { index in
                (responses[index.id].map(selector) ?? []).map {
                    ChartroomWorkspaceSearchResult(index: index, result: $0, rank: 0)
                }
            }
        }

        let strong = merge(\.strongResults)
        let weak = merge(\.weakResults)

        // Decide the ordering once for the whole set. A comparator that used similarity for some
        // pairs and fusion rank for others would not be a strict weak ordering, which `sorted` is
        // entitled to assume.
        let ordering = Self.ordering(for: strong)

        return ChartroomWorkspaceSearchResponse(
            query: request.query,
            mode: request.mode,
            results: Self.ranked(strong, by: ordering, limit: request.limit),
            weakResults: Self.ranked(weak, by: Self.ordering(for: weak), limit: request.limit),
            statuses: statuses,
            ordering: ordering
        )
    }

    /// Similarity is only comparable across stores when every result carries one *and* they all
    /// come from the same embedding space. Otherwise the merge falls back to fusion rank and says so.
    private static func ordering(
        for results: [ChartroomWorkspaceSearchResult]
    ) -> ChartroomWorkspaceSearchResponse.Ordering {
        guard !results.isEmpty else { return .fusionRank }
        guard results.allSatisfy({ $0.result.similarity != nil }) else { return .fusionRank }
        let spaces = Set(results.compactMap { $0.result.provenance.embeddingSpaceID })
        return spaces.count == 1 ? .similarity : .fusionRank
    }

    private static func ranked(
        _ results: [ChartroomWorkspaceSearchResult],
        by ordering: ChartroomWorkspaceSearchResponse.Ordering,
        limit: Int
    ) -> [ChartroomWorkspaceSearchResult] {
        results
            .sorted { isOrderedBefore($0, $1, by: ordering) }
            .prefix(max(0, limit))
            .enumerated()
            .map { offset, result in
                var ranked = result
                ranked.rank = offset + 1
                return ranked
            }
    }

    private func loadCatalogIfNeeded() throws {
        guard !loaded else { return }

        if FileManager.default.fileExists(atPath: catalogURL.path) {
            do {
                storedIndexes = try JSONDecoder().decode([ChartroomIndex].self, from: Data(contentsOf: catalogURL))
            } catch {
                throw IndexEngineError(
                    .configurationInvalid,
                    code: "chartroom.workspace.catalog-invalid",
                    recoverability: .needsUserAction,
                    summary: "The index catalog could not be read.",
                    detail: String(describing: error)
                )
            }
        } else {
            storedIndexes = bootstrapIndexes
            try persistCatalog()
        }

        loaded = true
    }

    private func persistCatalog() throws {
        do {
            try FileManager.default.createDirectory(
                at: catalogURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(storedIndexes).write(to: catalogURL, options: .atomic)
        } catch {
            throw IndexEngineError(
                .storageUnavailable,
                code: "chartroom.workspace.catalog-write-failed",
                recoverability: .retryable,
                summary: "The index catalog could not be saved.",
                detail: String(describing: error)
            )
        }
    }

    private func validatedName(_ rawName: String, excluding indexID: UUID?) throws -> String {
        let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            throw IndexEngineError(
                .configurationInvalid,
                code: "chartroom.workspace.index-name-empty",
                recoverability: .needsUserAction,
                summary: "Enter a name for the index."
            )
        }

        let normalizedName = Self.normalizedName(name)
        let duplicate = storedIndexes.contains {
            $0.id != indexID && Self.normalizedName($0.name) == normalizedName
        }
        guard !duplicate else {
            throw IndexEngineError(
                .configurationInvalid,
                code: "chartroom.workspace.index-name-duplicate",
                recoverability: .needsUserAction,
                summary: "An index with that name already exists."
            )
        }
        return name
    }

    private func unavailableIndexError(_ indexID: UUID) -> IndexEngineError {
        IndexEngineError(
            .configurationInvalid,
            code: "chartroom.workspace.index-not-found",
            recoverability: .needsUserAction,
            summary: "The selected index is unavailable.",
            detail: indexID.uuidString
        )
    }

    private static func normalizedName(_ name: String) -> String {
        name.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
    }

    private static func searchErrorMessage(_ error: Error) -> String {
        if let engineError = error as? IndexEngineError {
            return engineError.summary
        }
        return String(describing: error)
    }

    private static func isOrderedBefore(
        _ lhs: ChartroomWorkspaceSearchResult,
        _ rhs: ChartroomWorkspaceSearchResult,
        by ordering: ChartroomWorkspaceSearchResponse.Ordering
    ) -> Bool {
        if ordering == .similarity {
            // `ordering(for:)` has already established that both sides carry a similarity from one
            // shared embedding space, so this is a like-for-like comparison.
            let lhsSimilarity = lhs.result.similarity ?? -.infinity
            let rhsSimilarity = rhs.result.similarity ?? -.infinity
            if lhsSimilarity != rhsSimilarity {
                return lhsSimilarity > rhsSimilarity
            }
        } else if lhs.result.score != rhs.result.score {
            return lhs.result.score > rhs.result.score
        }
        if lhs.index.name != rhs.index.name {
            return lhs.index.name.localizedStandardCompare(rhs.index.name) == .orderedAscending
        }
        return lhs.result.rank < rhs.result.rank
    }
}

private struct NamespacedCursorStore: CursorStore {
    let base: any CursorStore
    let namespace: String

    func cursor(forKey key: String) -> SourceCursor? {
        base.cursor(forKey: qualified(key))
    }

    func setCursor(_ cursor: SourceCursor, forKey key: String) {
        base.setCursor(cursor, forKey: qualified(key))
    }

    private func qualified(_ key: String) -> String {
        "chartroom.workspace.\(namespace).\(key)"
    }
}

private enum SearchOutcome: Sendable {
    case success(ChartroomIndex, SearchResponse)
    case failure(ChartroomIndex, String)
}
