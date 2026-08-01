import Darwin
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
    private var storeDeletionBindings: [StoreDeletionKey: StoreDeletionBinding] = [:]
    private var storeDirectoryAnchors: [StoreEntryIdentity: StoreDirectoryAnchor] = [:]

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
        try FileManager.default.createDirectory(
            at: storesDirectory,
            withIntermediateDirectories: true
        )
        let storesAnchor = try storeDirectoryAnchor(for: storesDirectory)
        let directoryName = UUID().uuidString
        let directoryIdentity = try storesAnchor.createDirectory(named: directoryName)
        let index = ChartroomIndex(
            name: name,
            storeURL: storesDirectory
                .appending(path: directoryName, directoryHint: .isDirectory)
                .appending(path: "IndexEngine.sqlite")
        )
        let deletionBinding = StoreDeletionBinding.managed(
            parent: storesAnchor,
            directoryName: directoryName,
            identity: directoryIdentity
        )

        do {
            storedIndexes.append(index)
            try persistCatalog()
        } catch {
            storedIndexes.removeLast()
            try? storesAnchor.removeDirectoryTree(
                named: directoryName,
                expectedIdentity: directoryIdentity
            )
            throw error
        }
        storeDeletionBindings[StoreDeletionKey(index)] = deletionBinding
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
    public func deleteIndex(_ indexID: UUID) async throws -> ChartroomIndex {
        try loadCatalogIfNeeded()
        guard let position = storedIndexes.firstIndex(where: { $0.id == indexID }) else {
            throw unavailableIndexError(indexID)
        }
        if sessions[indexID]?.isOperationActiveOnCurrentTask() == true {
            throw deleteFromActiveOperationError(indexID)
        }

        let index = storedIndexes[position]
        storedIndexes.remove(at: position)
        do {
            try persistCatalog()
        } catch {
            storedIndexes.insert(index, at: position)
            throw error
        }

        let session = sessions.removeValue(forKey: indexID)
        await session?.invalidateAndWait()
        let binding = storeDeletionBindings.removeValue(forKey: StoreDeletionKey(index))
        try removeStoreFiles(for: index, binding: binding)
        return index
    }

    /// Deletes the store and the SQLite sidecars that travel with it.
    ///
    /// The containing directory is removed only when the workspace created it — one directory per
    /// index under `storesDirectory`. An adopted store such as the legacy bootstrap index lives
    /// beside the catalog itself, and deleting *that* parent would take the whole workspace with it.
    private func removeStoreFiles(
        for index: ChartroomIndex,
        binding: StoreDeletionBinding?
    ) throws {
        var failures: [String] = []
        switch binding {
        case let .managed(parent, directoryName, identity):
            do {
                try parent.removeDirectoryTree(
                    named: directoryName,
                    expectedIdentity: identity
                )
            } catch {
                failures.append("\(directoryName): \(error)")
            }
        case let .adopted(
            parent,
            storeName,
            storeIdentity,
            walIdentity,
            sharedMemoryIdentity
        ):
            do {
                guard let storeIdentity else {
                    try parent.verifyEntryAbsent(named: storeName)
                    break
                }
                let sidecars = [
                    (name: storeName + "-wal", identity: walIdentity),
                    (name: storeName + "-shm", identity: sharedMemoryIdentity),
                ]
                for sidecar in sidecars {
                    try parent.verifyFileRemovalTarget(
                        named: sidecar.name,
                        expectedIdentity: sidecar.identity
                    )
                }
                try parent.removeFileIfPresent(
                    named: storeName,
                    expectedIdentity: storeIdentity
                )
                for sidecar in sidecars {
                    guard let expectedIdentity = sidecar.identity else {
                        continue
                    }
                    try parent.removeFileIfPresent(
                        named: sidecar.name,
                        expectedIdentity: expectedIdentity
                    )
                }
            } catch {
                failures.append("\(storeName): \(error)")
            }
        case let .managedLeafAbsent(parent, directoryName):
            do {
                try parent.verifyEntryAbsent(named: directoryName)
            } catch {
                failures.append("\(directoryName): \(error)")
            }
        case let .unavailable(detail):
            failures.append(detail)
        case nil:
            failures.append("No deletion target was bound for \(index.storeURL.lastPathComponent).")
        }

        guard !failures.isEmpty else { return }
        throw IndexEngineError(
            .deletionFailed,
            code: "chartroom.workspace.store-delete-failed",
            recoverability: .needsUserAction,
            summary: "The index was removed from the catalog, but its store could not be deleted.",
            detail: failures.joined(separator: "\n")
        )
    }

    private func bindStoreDeletion(for index: ChartroomIndex) -> StoreDeletionBinding {
        let store = index.storeURL
        let directory = store.deletingLastPathComponent()
        let managed = isManagedStoreDirectory(directory)

        do {
            if managed {
                guard StoreDirectoryAnchor.isSafeLeafName(directory.lastPathComponent) else {
                    return .unavailable("The managed store directory name is invalid.")
                }
                let parent = try storeDirectoryAnchor(for: storesDirectory)
                guard let identity = try parent.entryIdentity(
                    named: directory.lastPathComponent
                ) else {
                    return .managedLeafAbsent(
                        parent: parent,
                        directoryName: directory.lastPathComponent
                    )
                }
                return .managed(
                    parent: parent,
                    directoryName: directory.lastPathComponent,
                    identity: identity
                )
            }

            guard StoreDirectoryAnchor.isSafeLeafName(store.lastPathComponent) else {
                return .unavailable("The adopted store filename is invalid.")
            }
            let parent = try storeDirectoryAnchor(for: directory)
            let storeName = store.lastPathComponent
            return .adopted(
                parent: parent,
                storeName: storeName,
                storeIdentity: try parent.entryIdentity(named: storeName),
                walIdentity: try parent.entryIdentity(named: storeName + "-wal"),
                sharedMemoryIdentity: try parent.entryIdentity(named: storeName + "-shm")
            )
        } catch {
            return .unavailable("Could not bind the store cleanup location: \(error)")
        }
    }

    private func isManagedStoreDirectory(_ directory: URL) -> Bool {
        guard UUID(uuidString: directory.lastPathComponent) != nil else {
            return false
        }

        let candidateRoot = directory.deletingLastPathComponent()
        if candidateRoot.standardizedFileURL == storesDirectory.standardizedFileURL {
            return true
        }

        guard let candidateCanonical = try? StoreDirectoryAnchor.canonicalURL(for: candidateRoot),
              let storesCanonical = try? StoreDirectoryAnchor.canonicalURL(for: storesDirectory) else {
            return false
        }
        return candidateCanonical == storesCanonical
    }

    private func storeDirectoryAnchor(for url: URL) throws -> StoreDirectoryAnchor {
        let opened = try StoreDirectoryAnchor(url: url)
        if let existing = storeDirectoryAnchors[opened.identity] {
            return existing
        }

        storeDirectoryAnchors[opened.identity] = opened
        return opened
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
            engineFactory: { [weak self] in
                guard let self else {
                    return try await factory(index)
                }
                return try await self.openEngine(
                    for: index,
                    using: factory
                )
            },
            cursorStore: NamespacedCursorStore(base: cursorStore, namespace: indexID.uuidString)
        )
        sessions[indexID] = session
        return session
    }

    private func openEngine(
        for index: ChartroomIndex,
        using factory: EngineFactory
    ) async throws -> (any IndexEngineClient, URL?) {
        let before = bindStoreDeletion(for: index)
        guard before.isSafeEngineTarget else {
            storeDeletionBindings[StoreDeletionKey(index)] = .unavailable(
                "The store path is not a regular file or managed directory."
            )
            throw storeBindingChangedError(index)
        }
        let opened = try await factory(index)

        var openedIndex = index
        openedIndex.storeURL = opened.1 ?? index.storeURL
        let after = bindStoreDeletion(for: openedIndex)
        guard before.hasSameTargetIdentity(as: after) else {
            storeDeletionBindings[StoreDeletionKey(index)] = .unavailable(
                "The store identity changed while its engine was opening."
            )
            throw storeBindingChangedError(index)
        }

        storeDeletionBindings[StoreDeletionKey(index)] = after
        return opened
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

        storeDeletionBindings.removeAll(keepingCapacity: true)
        for index in storedIndexes {
            storeDeletionBindings[StoreDeletionKey(index)] = bindStoreDeletion(for: index)
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

    private func deleteFromActiveOperationError(_ indexID: UUID) -> IndexEngineError {
        IndexEngineError(
            .deletionFailed,
            code: "chartroom.workspace.delete-from-active-operation",
            recoverability: .retryable,
            summary: "Finish the current index operation before deleting this index.",
            detail: indexID.uuidString
        )
    }

    private func storeBindingChangedError(_ index: ChartroomIndex) -> IndexEngineError {
        IndexEngineError(
            .storageUnavailable,
            code: "chartroom.workspace.store-binding-changed",
            recoverability: .retryable,
            summary: "The index store changed while it was opening.",
            detail: index.id.uuidString
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

private struct StoreDeletionKey: Hashable {
    var indexID: UUID
    var storeURL: URL

    init(_ index: ChartroomIndex) {
        self.indexID = index.id
        self.storeURL = index.storeURL
    }
}

private struct StoreEntryIdentity: Hashable {
    var device: dev_t
    var inode: ino_t
    var fileType: mode_t

    init(_ status: stat) {
        self.device = status.st_dev
        self.inode = status.st_ino
        self.fileType = status.st_mode & S_IFMT
    }
}

private enum StoreDeletionBinding {
    case managed(
        parent: StoreDirectoryAnchor,
        directoryName: String,
        identity: StoreEntryIdentity
    )
    case adopted(
        parent: StoreDirectoryAnchor,
        storeName: String,
        storeIdentity: StoreEntryIdentity?,
        walIdentity: StoreEntryIdentity?,
        sharedMemoryIdentity: StoreEntryIdentity?
    )
    case managedLeafAbsent(
        parent: StoreDirectoryAnchor,
        directoryName: String
    )
    case unavailable(String)

    var isSafeEngineTarget: Bool {
        switch self {
        case let .managed(_, _, identity):
            return identity.fileType == S_IFDIR
        case let .adopted(_, _, identity, _, _):
            return identity == nil || identity?.fileType == S_IFREG
        case .managedLeafAbsent, .unavailable:
            return true
        }
    }

    func hasSameTargetIdentity(as other: StoreDeletionBinding) -> Bool {
        switch (self, other) {
        case let (
            .managed(lhsParent, lhsName, lhsIdentity),
            .managed(rhsParent, rhsName, rhsIdentity)
        ):
            return lhsParent.identity == rhsParent.identity
                && lhsName == rhsName
                && lhsIdentity == rhsIdentity
        case let (
            .adopted(lhsParent, lhsName, lhsIdentity, _, _),
            .adopted(rhsParent, rhsName, rhsIdentity, _, _)
        ):
            return lhsParent.identity == rhsParent.identity
                && lhsName == rhsName
                && lhsIdentity == rhsIdentity
        case let (
            .managedLeafAbsent(lhsParent, lhsName),
            .managedLeafAbsent(rhsParent, rhsName)
        ):
            return lhsParent.identity == rhsParent.identity
                && lhsName == rhsName
        default:
            return false
        }
    }
}

/// A directory identity captured before deletion is requested. Cleanup stays relative to this
/// descriptor, so replacing any pathname component later cannot redirect destructive work.
private final class StoreDirectoryAnchor {
    struct Error: Swift.Error, CustomStringConvertible {
        var operation: String
        var item: String
        var code: Int32

        var description: String {
            "\(operation) \(item): \(String(cString: strerror(code)))"
        }
    }

    let canonicalURL: URL
    let identity: StoreEntryIdentity
    private let descriptor: Int32

    init(url: URL) throws {
        let canonicalURL = try Self.canonicalURL(for: url)
        let descriptor = try Self.openCanonicalDirectory(canonicalURL)
        var status = stat()
        guard Darwin.fstat(descriptor, &status) == 0 else {
            let code = Darwin.errno
            Darwin.close(descriptor)
            throw Error(operation: "inspect", item: canonicalURL.path, code: code)
        }
        self.canonicalURL = canonicalURL
        self.identity = StoreEntryIdentity(status)
        self.descriptor = descriptor
    }

    deinit {
        Darwin.close(descriptor)
    }

    static func isSafeLeafName(_ name: String) -> Bool {
        !name.isEmpty && name != "." && name != ".." && !name.contains("/")
    }

    static func canonicalURL(for url: URL) throws -> URL {
        let path = url.path
        let resolvedPath = path.withCString { Darwin.realpath($0, nil) }
        guard let resolvedPath else {
            let code = Darwin.errno
            throw Error(operation: "resolve", item: path, code: code)
        }
        defer { Darwin.free(resolvedPath) }
        return URL(filePath: String(cString: resolvedPath), directoryHint: .isDirectory)
    }

    func removeFileIfPresent(
        named name: String,
        expectedIdentity: StoreEntryIdentity
    ) throws {
        guard Self.isSafeLeafName(name) else {
            throw Error(operation: "unlink", item: name, code: EINVAL)
        }
        guard expectedIdentity.fileType != S_IFDIR else {
            throw Error(operation: "unlink", item: name, code: EISDIR)
        }
        guard let tombstone = try detachVerifiedEntry(
            named: name,
            expectedIdentity: expectedIdentity
        ) else {
            return
        }
        let result = tombstone.withCString {
            Darwin.unlinkat(descriptor, $0, 0)
        }
        guard result == 0 else {
            throw Error(operation: "unlink", item: name, code: Darwin.errno)
        }
    }

    func entryIdentity(named name: String) throws -> StoreEntryIdentity? {
        guard Self.isSafeLeafName(name) else {
            throw Error(operation: "inspect", item: name, code: EINVAL)
        }

        var status = stat()
        let result = name.withCString {
            Darwin.fstatat(descriptor, $0, &status, AT_SYMLINK_NOFOLLOW)
        }
        guard result == 0 else {
            let code = Darwin.errno
            if code == ENOENT {
                return nil
            }
            throw Error(operation: "inspect", item: name, code: code)
        }
        return StoreEntryIdentity(status)
    }

    func verifyEntryAbsent(named name: String) throws {
        guard try entryIdentity(named: name) == nil else {
            throw Error(operation: "verify", item: name, code: EBUSY)
        }
    }

    func verifyFileRemovalTarget(
        named name: String,
        expectedIdentity: StoreEntryIdentity?
    ) throws {
        if expectedIdentity?.fileType == S_IFDIR {
            throw Error(operation: "verify", item: name, code: EISDIR)
        }
        guard let currentIdentity = try entryIdentity(named: name) else {
            return
        }
        guard currentIdentity == expectedIdentity else {
            throw Error(operation: "verify", item: name, code: EBUSY)
        }
    }

    func createDirectory(named name: String) throws -> StoreEntryIdentity {
        guard Self.isSafeLeafName(name) else {
            throw Error(operation: "create", item: name, code: EINVAL)
        }
        let result = name.withCString {
            Darwin.mkdirat(descriptor, $0, mode_t(0o700))
        }
        guard result == 0 else {
            throw Error(operation: "create", item: name, code: Darwin.errno)
        }
        guard let identity = try entryIdentity(named: name),
              identity.fileType == S_IFDIR else {
            throw Error(operation: "verify", item: name, code: EIO)
        }
        return identity
    }

    func removeDirectoryTree(
        named name: String,
        expectedIdentity: StoreEntryIdentity
    ) throws {
        guard let tombstone = try detachVerifiedEntry(
            named: name,
            expectedIdentity: expectedIdentity
        ) else {
            return
        }

        try removeEntry(named: tombstone, from: descriptor)
    }

    /// Detaches one exact directory entry so later traversal cannot be redirected through its
    /// public name. The identity is checked on both sides of the rename; a raced replacement is
    /// restored without overwriting any entry that appeared at the original name.
    private func detachVerifiedEntry(
        named name: String,
        expectedIdentity: StoreEntryIdentity
    ) throws -> String? {
        guard Self.isSafeLeafName(name) else {
            throw Error(operation: "detach", item: name, code: EINVAL)
        }
        guard let currentIdentity = try entryIdentity(named: name) else {
            return nil
        }
        guard currentIdentity == expectedIdentity else {
            throw Error(operation: "verify", item: name, code: EBUSY)
        }

        let tombstone = "\(name).chartroom-delete-\(UUID().uuidString)"
        let renameResult = name.withCString { sourceName in
            tombstone.withCString { tombstoneName in
                Darwin.renameat(descriptor, sourceName, descriptor, tombstoneName)
            }
        }
        guard renameResult == 0 else {
            let code = Darwin.errno
            if code == ENOENT {
                return nil
            }
            throw Error(operation: "detach", item: name, code: code)
        }

        guard try entryIdentity(named: tombstone) == expectedIdentity else {
            let restoreResult = tombstone.withCString { tombstoneName in
                name.withCString { originalName in
                    Darwin.renameatx_np(
                        descriptor,
                        tombstoneName,
                        descriptor,
                        originalName,
                        UInt32(RENAME_EXCL)
                    )
                }
            }
            let code = restoreResult == 0 ? EBUSY : Darwin.errno
            throw Error(operation: "verify", item: name, code: code)
        }
        return tombstone
    }

    private func removeEntry(named name: String, from parentDescriptor: Int32) throws {
        try name.withCString { namePointer in
            try removeEntry(
                namePointer: namePointer,
                displayName: name,
                from: parentDescriptor
            )
        }
    }

    private func removeEntry(
        namePointer: UnsafePointer<CChar>,
        displayName: String,
        from parentDescriptor: Int32
    ) throws {
        var status = stat()
        guard Darwin.fstatat(
            parentDescriptor,
            namePointer,
            &status,
            AT_SYMLINK_NOFOLLOW
        ) == 0 else {
            let code = Darwin.errno
            if code == ENOENT {
                return
            }
            throw Error(operation: "inspect", item: displayName, code: code)
        }

        guard (status.st_mode & S_IFMT) == S_IFDIR else {
            guard Darwin.unlinkat(parentDescriptor, namePointer, 0) == 0 else {
                throw Error(
                    operation: "unlink",
                    item: displayName,
                    code: Darwin.errno
                )
            }
            return
        }

        let childDescriptor = Darwin.openat(
            parentDescriptor,
            namePointer,
            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW
        )
        guard childDescriptor >= 0 else {
            throw Error(operation: "open", item: displayName, code: Darwin.errno)
        }
        defer { Darwin.close(childDescriptor) }

        var openedStatus = stat()
        guard Darwin.fstat(childDescriptor, &openedStatus) == 0 else {
            throw Error(operation: "inspect", item: displayName, code: Darwin.errno)
        }
        guard openedStatus.st_dev == status.st_dev,
              openedStatus.st_ino == status.st_ino else {
            throw Error(operation: "open", item: displayName, code: EBUSY)
        }

        try removeDirectoryContents(childDescriptor)

        var finalStatus = stat()
        guard Darwin.fstatat(
            parentDescriptor,
            namePointer,
            &finalStatus,
            AT_SYMLINK_NOFOLLOW
        ) == 0 else {
            throw Error(operation: "inspect", item: displayName, code: Darwin.errno)
        }
        guard finalStatus.st_dev == openedStatus.st_dev,
              finalStatus.st_ino == openedStatus.st_ino else {
            throw Error(operation: "remove", item: displayName, code: EBUSY)
        }
        guard Darwin.unlinkat(parentDescriptor, namePointer, AT_REMOVEDIR) == 0 else {
            throw Error(operation: "remove", item: displayName, code: Darwin.errno)
        }
    }

    private func removeDirectoryContents(_ directoryDescriptor: Int32) throws {
        let iteratorDescriptor = Darwin.dup(directoryDescriptor)
        guard iteratorDescriptor >= 0 else {
            throw Error(operation: "duplicate", item: canonicalURL.path, code: Darwin.errno)
        }
        guard let stream = Darwin.fdopendir(iteratorDescriptor) else {
            let code = Darwin.errno
            Darwin.close(iteratorDescriptor)
            throw Error(operation: "enumerate", item: canonicalURL.path, code: code)
        }
        defer { Darwin.closedir(stream) }

        while true {
            Darwin.errno = 0
            guard let entry = Darwin.readdir(stream) else {
                let code = Darwin.errno
                guard code == 0 else {
                    throw Error(operation: "enumerate", item: canonicalURL.path, code: code)
                }
                return
            }

            let capacity = MemoryLayout.size(ofValue: entry.pointee.d_name)
            try withUnsafePointer(to: &entry.pointee.d_name) { tuplePointer in
                try tuplePointer.withMemoryRebound(
                    to: CChar.self,
                    capacity: capacity
                ) { namePointer in
                    let name = String(cString: namePointer)
                    guard name != ".", name != ".." else { return }
                    try removeEntry(
                        namePointer: namePointer,
                        displayName: name,
                        from: directoryDescriptor
                    )
                }
            }
        }
    }

    private static func openCanonicalDirectory(_ url: URL) throws -> Int32 {
        var descriptor = Darwin.open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw Error(operation: "open", item: "/", code: Darwin.errno)
        }

        for component in url.pathComponents where component != "/" {
            let nextDescriptor = component.withCString {
                Darwin.openat(
                    descriptor,
                    $0,
                    O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW
                )
            }
            guard nextDescriptor >= 0 else {
                let code = Darwin.errno
                Darwin.close(descriptor)
                throw Error(operation: "open", item: url.path, code: code)
            }
            Darwin.close(descriptor)
            descriptor = nextDescriptor
        }
        return descriptor
    }
}
