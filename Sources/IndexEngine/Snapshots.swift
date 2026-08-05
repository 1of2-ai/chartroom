import Foundation

public struct DocumentBrowseResponse: Codable, Hashable, Sendable {
    public var request: DocumentBrowseRequest
    public var documents: [DocumentSummary]
    public var totalMatching: Int
    public var facets: DocumentBrowseFacets

    public init(
        request: DocumentBrowseRequest,
        documents: [DocumentSummary],
        totalMatching: Int,
        facets: DocumentBrowseFacets = .empty
    ) {
        self.request = request
        self.documents = documents
        self.totalMatching = totalMatching
        self.facets = facets
    }

    public var returnedCount: Int {
        documents.count
    }

    public var offset: Int {
        request.offset
    }

    public var startIndex: Int? {
        documents.isEmpty ? nil : offset + 1
    }

    public var endIndex: Int? {
        documents.isEmpty ? nil : offset + returnedCount
    }

    public var hasPreviousPage: Bool {
        offset > 0
    }

    public var hasNextPage: Bool {
        offset + returnedCount < totalMatching
    }

    public var isTruncated: Bool {
        hasNextPage
    }
}

public struct DocumentBrowseFacets: Codable, Hashable, Sendable {
    public static let empty = DocumentBrowseFacets(sourceIDs: [], contentTypes: [])

    public var sourceIDs: [SourceID]
    public var contentTypes: [String]

    public init(sourceIDs: [SourceID], contentTypes: [String]) {
        self.sourceIDs = sourceIDs
        self.contentTypes = contentTypes
    }
}

/// A typed, GUI-facing projection of one stored document. It is enough to list, filter, and inspect
/// the corpus without exposing storage internals. Counts and identity come from durable records; the
/// body stays behind `search` and future chunk inspection contracts.
public struct DocumentSummary: Codable, Hashable, Sendable, Identifiable {
    public var id: DocumentID
    public var title: String
    public var sourceID: SourceID?
    public var sourceURI: URL?
    public var contentType: String
    public var byteSize: Int
    public var chunkCount: Int
    public var ingestedAt: Date
    public var modifiedAt: Date?
    public var policyID: PolicyID?
    public var clusterID: EngineID?

    public init(
        id: DocumentID,
        title: String,
        sourceID: SourceID? = nil,
        sourceURI: URL? = nil,
        contentType: String,
        byteSize: Int,
        chunkCount: Int,
        ingestedAt: Date,
        modifiedAt: Date? = nil,
        policyID: PolicyID? = nil,
        clusterID: EngineID? = nil
    ) {
        self.id = id
        self.title = title
        self.sourceID = sourceID
        self.sourceURI = sourceURI
        self.contentType = contentType
        self.byteSize = byteSize
        self.chunkCount = chunkCount
        self.ingestedAt = ingestedAt
        self.modifiedAt = modifiedAt
        self.policyID = policyID
        self.clusterID = clusterID
    }
}

/// A typed, GUI-facing projection of one active chunk — enough for a retrieval harness
/// to browse a document's chunks (text, ordinal, heading path, offset ranges) and see
/// whether each one carries an embedding.
public struct ChunkSummary: Codable, Hashable, Sendable, Identifiable {
    public var id: ChunkID
    public var documentID: DocumentID
    public var ordinal: Int
    public var text: String
    public var headingPath: String
    public var byteStart: Int
    public var byteEnd: Int
    public var characterStart: Int
    public var characterEnd: Int
    public var tokenStart: Int
    public var tokenEnd: Int
    /// 1-based inclusive line range in the document. `nil` for chunks written before line
    /// tracking existed — re-ingesting the document fills them.
    public var lineStart: Int?
    public var lineEnd: Int?
    /// Text immediately before and after the chunk, so a caller can see where the window sits.
    public var contextPrefix: String
    public var contextSuffix: String
    public var contentHash: String
    public var hasEmbedding: Bool

    public init(
        id: ChunkID,
        documentID: DocumentID,
        ordinal: Int,
        text: String,
        headingPath: String = "",
        byteStart: Int,
        byteEnd: Int,
        characterStart: Int,
        characterEnd: Int,
        tokenStart: Int,
        tokenEnd: Int,
        lineStart: Int? = nil,
        lineEnd: Int? = nil,
        contextPrefix: String = "",
        contextSuffix: String = "",
        contentHash: String,
        hasEmbedding: Bool
    ) {
        self.id = id
        self.documentID = documentID
        self.ordinal = ordinal
        self.text = text
        self.headingPath = headingPath
        self.byteStart = byteStart
        self.byteEnd = byteEnd
        self.characterStart = characterStart
        self.characterEnd = characterEnd
        self.tokenStart = tokenStart
        self.tokenEnd = tokenEnd
        self.lineStart = lineStart
        self.lineEnd = lineEnd
        self.contextPrefix = contextPrefix
        self.contextSuffix = contextSuffix
        self.contentHash = contentHash
        self.hasEmbedding = hasEmbedding
    }
}

public struct IndexEngineSnapshot: Codable, Sendable, Equatable {
    public var storeURL: URL?
    /// On-disk footprint of the store in bytes (database + WAL/SHM sidecars), or nil for
    /// an in-memory store or when the size cannot be read. Optional so older diagnostics
    /// JSON without the field still decodes.
    public var storeByteSize: Int64?
    public var objectCount: Int
    public var documentCount: Int
    public var chunkCount: Int
    public var embeddingCount: Int
    public var modelID: String
    public var embeddingDimension: Int
    public var embeddingSpaceID: EmbeddingSpaceID
    /// Which spaces the store's vectors actually live in. `embeddingCount` above counts only the
    /// active space, so an index that is fully populated but unreadable by the current embedder
    /// reports zero there and looks identical to an empty one — this is what tells them apart.
    /// Optional so older diagnostics JSON without the field still decodes.
    public var embeddingSpaceCoverage: EmbeddingSpaceCoverage?
    public var lastIngestedAt: Date?
    public var policyStates: [PolicyResolution]

    public init(
        storeURL: URL?,
        storeByteSize: Int64? = nil,
        objectCount: Int,
        documentCount: Int = 0,
        chunkCount: Int = 0,
        embeddingCount: Int = 0,
        modelID: String,
        embeddingDimension: Int,
        embeddingSpaceID: EmbeddingSpaceID,
        embeddingSpaceCoverage: EmbeddingSpaceCoverage? = nil,
        lastIngestedAt: Date?,
        policyStates: [PolicyResolution]
    ) {
        self.storeURL = storeURL
        self.storeByteSize = storeByteSize
        self.objectCount = objectCount
        self.documentCount = documentCount
        self.chunkCount = chunkCount
        self.embeddingCount = embeddingCount
        self.modelID = modelID
        self.embeddingDimension = embeddingDimension
        self.embeddingSpaceID = embeddingSpaceID
        self.embeddingSpaceCoverage = embeddingSpaceCoverage
        self.lastIngestedAt = lastIngestedAt
        self.policyStates = policyStates
    }
}

public struct IngestionSummary: Sendable, Equatable {
    public var jobID: JobID
    public var acceptedCount: Int
    public var failedCount: Int
    public var failures: [FailureSnapshot]
    public var startedAt: Date
    public var finishedAt: Date

    public init(
        jobID: JobID,
        acceptedCount: Int,
        failedCount: Int,
        failures: [FailureSnapshot],
        startedAt: Date,
        finishedAt: Date
    ) {
        self.jobID = jobID
        self.acceptedCount = acceptedCount
        self.failedCount = failedCount
        self.failures = failures
        self.startedAt = startedAt
        self.finishedAt = finishedAt
    }
}

public struct DeletionSummary: Sendable, Equatable {
    public var jobID: JobID
    public var requestedCount: Int
    public var deletedCount: Int
    public var failedCount: Int
    public var failures: [FailureSnapshot]
    public var startedAt: Date
    public var finishedAt: Date

    public init(
        jobID: JobID,
        requestedCount: Int,
        deletedCount: Int,
        failedCount: Int,
        failures: [FailureSnapshot],
        startedAt: Date,
        finishedAt: Date
    ) {
        self.jobID = jobID
        self.requestedCount = requestedCount
        self.deletedCount = deletedCount
        self.failedCount = failedCount
        self.failures = failures
        self.startedAt = startedAt
        self.finishedAt = finishedAt
    }
}

public struct SearchResponse: Codable, Sendable, Equatable {
    public var query: String
    public var mode: RetrievalMode
    /// Every retrieved result, ranked, including the weak tail. Most callers want
    /// `strongResults` — see below for why the two are separated here rather than per client.
    public var results: [SearchResultSnapshot]
    public var diagnostics: SearchDiagnostics

    public init(
        query: String,
        mode: RetrievalMode,
        results: [SearchResultSnapshot],
        diagnostics: SearchDiagnostics
    ) {
        self.query = query
        self.mode = mode
        self.results = results
        self.diagnostics = diagnostics
    }

    /// Results with real evidence behind them.
    ///
    /// Vector search returns its top-N by cosine unconditionally, so `results` always contains a
    /// tail whether or not anything matched. The split is defined here, once, because every
    /// consumer needs the same rule: the GUI, the workspace merge, and the agent tool surface each
    /// have to distinguish "here are your answers" from "nothing matched", and three
    /// implementations of that would be three chances to disagree.
    public var strongResults: [SearchResultSnapshot] {
        results.filter { !$0.isWeak }
    }

    /// The below-threshold tail. Offer it deliberately; never as the answer.
    public var weakResults: [SearchResultSnapshot] {
        results.filter(\.isWeak)
    }
}

public struct SearchResultSnapshot: Codable, Hashable, Sendable {
    public var id: EngineID
    public var documentID: DocumentID
    public var chunkID: ChunkID
    public var sourceID: SourceID?
    public var title: String
    public var snippet: String?
    public var sourceURI: URL?
    public var contentType: String
    /// Fusion score — ordering only. See `similarity` for how good the match actually is.
    public var score: Double
    public var rank: Int
    /// Cosine similarity to the query, when the vector channel produced this result.
    /// `nil` means the result arrived by title or keyword match, where `diagnostics.keywordScore`
    /// carries the signal instead.
    public var similarity: Double?
    /// Vector-only evidence below the embedder's threshold: tail, not an answer. Clients should
    /// exclude these from the primary result set rather than each inventing a floor.
    public var isWeak: Bool
    /// 1-based inclusive line range of the matching chunk within the document. This is the unit a
    /// consumer acts on — an editor jump, a file read, an edit — so a path plus a snippet without
    /// it forces the whole file to be re-read. `nil` when the chunk predates line tracking.
    public var lineStart: Int?
    public var lineEnd: Int?
    public var diagnostics: SearchResultDiagnostics
    public var provenance: ResultProvenance

    public init(
        id: EngineID,
        documentID: DocumentID,
        chunkID: ChunkID,
        sourceID: SourceID?,
        title: String,
        snippet: String?,
        sourceURI: URL?,
        contentType: String,
        score: Double,
        rank: Int,
        similarity: Double? = nil,
        isWeak: Bool = false,
        lineStart: Int? = nil,
        lineEnd: Int? = nil,
        diagnostics: SearchResultDiagnostics,
        provenance: ResultProvenance
    ) {
        self.id = id
        self.documentID = documentID
        self.chunkID = chunkID
        self.sourceID = sourceID
        self.title = title
        self.snippet = snippet
        self.sourceURI = sourceURI
        self.contentType = contentType
        self.score = score
        self.rank = rank
        self.similarity = similarity
        self.isWeak = isWeak
        self.lineStart = lineStart
        self.lineEnd = lineEnd
        self.diagnostics = diagnostics
        self.provenance = provenance
    }
}

/// The embedding spaces a store holds vectors in, against the space the active embedder reads.
///
/// A store whose vectors all sit in some *other* space is not empty and not broken — it is
/// unreadable by the current model, which is a different problem with a different remedy
/// (re-index, or restore the previous embedder).
public struct EmbeddingSpaceCoverage: Codable, Hashable, Sendable {
    public struct Space: Codable, Hashable, Sendable {
        public var id: EmbeddingSpaceID
        public var embeddingCount: Int

        public init(id: EmbeddingSpaceID, embeddingCount: Int) {
            self.id = id
            self.embeddingCount = embeddingCount
        }
    }

    public var activeSpaceID: EmbeddingSpaceID
    public var storedSpaces: [Space]

    public init(activeSpaceID: EmbeddingSpaceID, storedSpaces: [Space] = []) {
        self.activeSpaceID = activeSpaceID
        self.storedSpaces = storedSpaces
    }

    public var activeSpaceEmbeddingCount: Int {
        storedSpaces.first { $0.id == activeSpaceID }?.embeddingCount ?? 0
    }

    public var totalEmbeddingCount: Int {
        storedSpaces.reduce(0) { $0 + $1.embeddingCount }
    }

    /// The store holds vectors, but none the active embedder can read. Vector retrieval is
    /// silently blind until the index is rebuilt or the previous embedder is restored.
    public var isOrphaned: Bool {
        activeSpaceEmbeddingCount == 0 && totalEmbeddingCount > 0
    }

    /// Vectors exist in a space the active embedder cannot read, alongside ones it can.
    public var hasUnreadableVectors: Bool {
        totalEmbeddingCount > activeSpaceEmbeddingCount
    }
}

/// Result of inspecting whether stored vectors belong to the active embedder's space.
///
/// `SearchDiagnostics.embeddingSpaceCoverageState == nil` means the read was not needed. An
/// unavailable read is distinct from a healthy store: retrieval may still return lexical results,
/// but it cannot claim whether the vector space is complete.
public enum EmbeddingSpaceCoverageState: String, Codable, Hashable, Sendable {
    /// The coverage read succeeded and the active space is not wholly absent. Other historical
    /// spaces may still be present, so this is deliberately narrower than "healthy."
    case notOrphaned
    case orphaned
    case unavailable
}

public struct SearchDiagnostics: Codable, Hashable, Sendable {
    public var degraded: Bool
    public var missingChannels: [RetrievalChannel]
    /// The index holds vectors, but none in the embedder's space. Vector retrieval contributed
    /// nothing and no error was raised — without this flag the result is indistinguishable from
    /// an honestly empty index.
    public var embeddingSpaceMismatch: Bool
    public var embeddingSpaceCoverageState: EmbeddingSpaceCoverageState?
    public var sqlFilterLatency: TimeInterval?
    public var ftsLatency: TimeInterval?
    public var vectorLatency: TimeInterval?
    public var fusionLatency: TimeInterval?
    public var snippetLatency: TimeInterval?
    public var totalLatency: TimeInterval?

    public init(
        degraded: Bool = false,
        missingChannels: [RetrievalChannel] = [],
        embeddingSpaceMismatch: Bool = false,
        sqlFilterLatency: TimeInterval? = nil,
        ftsLatency: TimeInterval? = nil,
        vectorLatency: TimeInterval? = nil,
        fusionLatency: TimeInterval? = nil,
        snippetLatency: TimeInterval? = nil,
        totalLatency: TimeInterval? = nil
    ) {
        self.degraded = degraded
        self.missingChannels = missingChannels
        self.embeddingSpaceMismatch = embeddingSpaceMismatch
        self.embeddingSpaceCoverageState = nil
        self.sqlFilterLatency = sqlFilterLatency
        self.ftsLatency = ftsLatency
        self.vectorLatency = vectorLatency
        self.fusionLatency = fusionLatency
        self.snippetLatency = snippetLatency
        self.totalLatency = totalLatency
    }

    public init(
        degraded: Bool = false,
        missingChannels: [RetrievalChannel] = [],
        embeddingSpaceMismatch: Bool = false,
        embeddingSpaceCoverageState: EmbeddingSpaceCoverageState?,
        sqlFilterLatency: TimeInterval? = nil,
        ftsLatency: TimeInterval? = nil,
        vectorLatency: TimeInterval? = nil,
        fusionLatency: TimeInterval? = nil,
        snippetLatency: TimeInterval? = nil,
        totalLatency: TimeInterval? = nil
    ) {
        self.degraded = degraded
        self.missingChannels = missingChannels
        self.embeddingSpaceMismatch = embeddingSpaceMismatch
        self.embeddingSpaceCoverageState = embeddingSpaceCoverageState
        self.sqlFilterLatency = sqlFilterLatency
        self.ftsLatency = ftsLatency
        self.vectorLatency = vectorLatency
        self.fusionLatency = fusionLatency
        self.snippetLatency = snippetLatency
        self.totalLatency = totalLatency
    }
}

public enum RetrievalChannel: String, Codable, Hashable, Sendable {
    case sql
    case fts
    case vector
    case exact
    case graph
    case reranker
}

public struct SearchResultDiagnostics: Codable, Hashable, Sendable {
    public var ftsRank: Int?
    public var vectorRank: Int?
    public var exactRank: Int?
    /// BM25 relevance from the keyword channel, negated so larger is better.
    public var keywordScore: Double?
    public var graphReason: String?
    public var appliedBoosts: [AppliedBoost]

    public init(
        ftsRank: Int? = nil,
        vectorRank: Int? = nil,
        exactRank: Int? = nil,
        keywordScore: Double? = nil,
        graphReason: String? = nil,
        appliedBoosts: [AppliedBoost] = []
    ) {
        self.ftsRank = ftsRank
        self.vectorRank = vectorRank
        self.exactRank = exactRank
        self.keywordScore = keywordScore
        self.graphReason = graphReason
        self.appliedBoosts = appliedBoosts
    }
}

public struct AppliedBoost: Codable, Hashable, Sendable {
    public var id: EngineID
    public var label: String
    public var value: Double

    public init(id: EngineID, label: String, value: Double) {
        self.id = id
        self.label = label
        self.value = value
    }
}

public struct ResultProvenance: Codable, Hashable, Sendable {
    public var connectorID: ConnectorID?
    public var policyID: PolicyID?
    public var representationID: RepresentationID?
    public var embeddingSpaceID: EmbeddingSpaceID?

    public init(
        connectorID: ConnectorID? = nil,
        policyID: PolicyID? = nil,
        representationID: RepresentationID? = nil,
        embeddingSpaceID: EmbeddingSpaceID? = nil
    ) {
        self.connectorID = connectorID
        self.policyID = policyID
        self.representationID = representationID
        self.embeddingSpaceID = embeddingSpaceID
    }
}

public struct FailureSnapshot: Codable, Hashable, Sendable {
    public enum Category: String, Codable, Sendable {
        case sourceUnavailable
        case permissionDenied
        case unsupportedContentType
        case decodeFailure
        case extractionFailure
        case chunkingFailure
        case embeddingFailure
        case storageFailure
        case migrationFailure
        case connectorProtocolFailure
        case mcpFailure
    }

    public var id: EngineID
    public var category: Category
    public var message: String
    public var detail: String
    public var sourceID: SourceID?
    public var documentID: DocumentID?
    /// Where the failed payload came from, so retries can re-fetch it directly
    /// instead of reconstructing a location from the document ID.
    public var sourceURI: URL?
    public var recoverability: IndexEngineError.Recoverability
    public var occurredAt: Date
    public var isRecoverable: Bool { recoverability != .unrecoverable }

    public init(
        id: EngineID,
        category: Category,
        message: String,
        detail: String,
        sourceID: SourceID? = nil,
        documentID: DocumentID? = nil,
        sourceURI: URL? = nil,
        recoverability: IndexEngineError.Recoverability,
        occurredAt: Date
    ) {
        self.id = id
        self.category = category
        self.message = message
        self.detail = detail
        self.sourceID = sourceID
        self.documentID = documentID
        self.sourceURI = sourceURI
        self.recoverability = recoverability
        self.occurredAt = occurredAt
    }

    public init(
        id: EngineID,
        category: Category,
        message: String,
        detail: String,
        sourceID: SourceID? = nil,
        documentID: DocumentID? = nil,
        isRecoverable: Bool,
        occurredAt: Date
    ) {
        self.init(
            id: id,
            category: category,
            message: message,
            detail: detail,
            sourceID: sourceID,
            documentID: documentID,
            recoverability: isRecoverable ? .retryable : .unrecoverable,
            occurredAt: occurredAt
        )
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case category
        case message
        case detail
        case sourceID
        case documentID
        case sourceURI
        case recoverability
        case isRecoverable
        case occurredAt
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(EngineID.self, forKey: .id)
        category = try container.decode(Category.self, forKey: .category)
        message = try container.decode(String.self, forKey: .message)
        detail = try container.decode(String.self, forKey: .detail)
        sourceID = try container.decodeIfPresent(SourceID.self, forKey: .sourceID)
        documentID = try container.decodeIfPresent(DocumentID.self, forKey: .documentID)
        sourceURI = try container.decodeIfPresent(URL.self, forKey: .sourceURI)
        occurredAt = try container.decode(Date.self, forKey: .occurredAt)

        if let storedRecoverability = try container.decodeIfPresent(
            IndexEngineError.Recoverability.self,
            forKey: .recoverability
        ) {
            recoverability = storedRecoverability
        } else {
            let legacyIsRecoverable = try container.decodeIfPresent(Bool.self, forKey: .isRecoverable) ?? true
            recoverability = legacyIsRecoverable ? .retryable : .unrecoverable
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(category, forKey: .category)
        try container.encode(message, forKey: .message)
        try container.encode(detail, forKey: .detail)
        try container.encodeIfPresent(sourceID, forKey: .sourceID)
        try container.encodeIfPresent(documentID, forKey: .documentID)
        try container.encodeIfPresent(sourceURI, forKey: .sourceURI)
        try container.encode(recoverability, forKey: .recoverability)
        try container.encode(isRecoverable, forKey: .isRecoverable)
        try container.encode(occurredAt, forKey: .occurredAt)
    }
}

public struct IndexHealthSnapshot: Codable, Hashable, Sendable {
    public var objectCount: Int
    public var documentCount: Int
    public var chunkCount: Int
    public var embeddingCount: Int
    public var policyStates: [PolicyResolution]
    public var vectorBackendStatus: VectorStorageStatus?

    public init(
        objectCount: Int,
        documentCount: Int = 0,
        chunkCount: Int = 0,
        embeddingCount: Int = 0,
        policyStates: [PolicyResolution],
        vectorBackendStatus: VectorStorageStatus? = nil
    ) {
        self.objectCount = objectCount
        self.documentCount = documentCount
        self.chunkCount = chunkCount
        self.embeddingCount = embeddingCount
        self.policyStates = policyStates
        self.vectorBackendStatus = vectorBackendStatus
    }
}

public struct JobSnapshot: Codable, Hashable, Sendable {
    public enum State: String, Codable, Sendable {
        case queued
        case running
        case committing
        case succeeded
        case failed
        case cancelled
        case recovering
    }

    public enum Kind: String, Codable, Sendable {
        case ingest
        case delete
    }

    public var id: JobID
    public var state: State
    public var kind: Kind
    public var completedUnitCount: Int
    public var totalUnitCount: Int?
    public var message: String

    public init(
        id: JobID,
        state: State,
        kind: Kind = .ingest,
        completedUnitCount: Int = 0,
        totalUnitCount: Int? = nil,
        message: String = ""
    ) {
        self.id = id
        self.state = state
        self.kind = kind
        self.completedUnitCount = completedUnitCount
        self.totalUnitCount = totalUnitCount
        self.message = message
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case state
        case kind
        case completedUnitCount
        case totalUnitCount
        case message
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(JobID.self, forKey: .id)
        state = try container.decode(State.self, forKey: .state)
        kind = try container.decodeIfPresent(Kind.self, forKey: .kind) ?? .ingest
        completedUnitCount = try container.decode(Int.self, forKey: .completedUnitCount)
        totalUnitCount = try container.decodeIfPresent(Int.self, forKey: .totalUnitCount)
        message = try container.decode(String.self, forKey: .message)
    }
}

public struct ModelStatusSnapshot: Codable, Hashable, Sendable {
    public var modelID: String
    public var embeddingSpaceID: EmbeddingSpaceID?
    public var dimension: Int

    /// The provider answered a probe with a correctly shaped vector.
    ///
    /// Availability is not the same claim as semantic retrieval: a hashing stand-in is perfectly
    /// available and has no semantic channel at all. Read this together with `isModelBacked`.
    public var isAvailable: Bool

    /// The vectors come from a real embedding model rather than a stand-in.
    ///
    /// Carried separately and without a default so a fallback cannot present itself as the real
    /// thing. Hosts that show only `isAvailable` will report a hashing fallback as a healthy
    /// model, which is precisely the state this field exists to make visible.
    public var isModelBacked: Bool

    /// Image files embed by content. When false, ingest embeds an image's filename text
    /// instead — the image stays findable by name, not by what it shows. Surfaced so hosts
    /// can say which of the two an index actually delivers.
    public var supportsImageEmbedding: Bool

    public var message: String

    public init(
        modelID: String,
        embeddingSpaceID: EmbeddingSpaceID?,
        dimension: Int,
        isAvailable: Bool,
        isModelBacked: Bool,
        supportsImageEmbedding: Bool = false,
        message: String = ""
    ) {
        self.modelID = modelID
        self.embeddingSpaceID = embeddingSpaceID
        self.dimension = dimension
        self.isAvailable = isAvailable
        self.isModelBacked = isModelBacked
        self.supportsImageEmbedding = supportsImageEmbedding
        self.message = message
    }
}

public struct VectorStorageStatus: Codable, Hashable, Sendable {
    public enum State: String, Codable, Sendable {
        case unavailable
        case preparing
        case ready
        case degraded
        case failed
    }

    public var backendID: VectorBackendID
    public var state: State
    public var message: String

    public init(backendID: VectorBackendID, state: State, message: String = "") {
        self.backendID = backendID
        self.state = state
        self.message = message
    }
}
