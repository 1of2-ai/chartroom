import Foundation
import UniformTypeIdentifiers

/// Stable identifier used by the public engine boundary.
///
/// The current type intentionally stays small and string-backed so connectors,
/// GUI fixtures, and tests can create deterministic IDs without touching storage
/// internals. More specific wrappers can be added later if the call sites prove
/// they need stronger separation.
public struct EngineID: RawRepresentable, Hashable, Codable, Sendable, ExpressibleByStringLiteral, CustomStringConvertible {
    public var rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public init(stringLiteral value: String) {
        self.rawValue = value
    }

    public var description: String { rawValue }
}

public typealias SourceID = EngineID
public typealias DocumentID = EngineID
public typealias DocumentVersionID = EngineID
public typealias RepresentationID = EngineID
public typealias RepresentationLineageID = EngineID
public typealias ChunkID = EngineID
public typealias EmbeddingID = EngineID
public typealias EmbeddingSpaceID = EngineID
public typealias PolicyID = EngineID
public typealias JobID = EngineID
public typealias ComponentID = EngineID
public typealias VectorBackendID = EngineID
public typealias ConnectorID = EngineID

public extension EngineID {
    static let builtInTextExtractor: EngineID = "indexengine.extractor.text"
    static let builtInTextChunker: EngineID = "indexengine.chunker.text-window"
    static let builtInEmbeddingProvider: EngineID = "indexengine.embedding.default"
    static let builtInSQLiteVectorBackend: EngineID = "indexengine.vector.sqlite-exact"
}

/// Whether one diagnostic-history channel was read from durable storage.
public enum DiagnosticHistoryAvailability: String, Codable, Hashable, Sendable {
    /// The durable query completed, including when it returned no rows.
    case available
    /// The client supports durable history, but its read failed.
    case unavailable
    /// The client does not expose a durable-history completeness signal.
    case notSupported
}

/// Diagnostic entries together with a completeness signal for each durable table.
///
/// Jobs and failures remain separate because one table can be readable while the other is not.
public struct DiagnosticHistory: Codable, Hashable, Sendable {
    public var jobs: [JobSnapshot]
    public var failures: [FailureSnapshot]
    public var jobsAvailability: DiagnosticHistoryAvailability
    public var failuresAvailability: DiagnosticHistoryAvailability

    public init(
        jobs: [JobSnapshot] = [],
        failures: [FailureSnapshot] = [],
        jobsAvailability: DiagnosticHistoryAvailability,
        failuresAvailability: DiagnosticHistoryAvailability
    ) {
        self.jobs = jobs
        self.failures = failures
        self.jobsAvailability = jobsAvailability
        self.failuresAvailability = failuresAvailability
    }
}

/// Public facade used by apps, GUI view models, and test harnesses.
public protocol IndexEngineClient: Sendable {
    func ingest(_ request: IngestRequest) async throws -> IngestionSummary
    func delete(_ request: DeleteRequest) async throws -> DeletionSummary
    func search(_ request: SearchRequest) async throws -> SearchResponse
    func browseDocuments(_ request: DocumentBrowseRequest) async throws -> DocumentBrowseResponse
    func chunks(forDocument documentID: DocumentID) async throws -> [ChunkSummary]
    func health() async -> IndexHealthSnapshot
    func failures(limit: Int) async -> [FailureSnapshot]
    func jobs(limit: Int) async -> [JobSnapshot]
    func diagnosticHistory(limit: Int) async -> DiagnosticHistory
    func modelStatus() async -> ModelStatusSnapshot
    func snapshot() async -> IndexEngineSnapshot

    /// Re-embed content the active embedder cannot read, restoring an index orphaned by a model
    /// change. Rebuilds from stored chunk text, so it needs no access to the original sources.
    func rebuildEmbeddings() async throws -> EmbeddingRebuildSummary

    /// Clear recorded failure diagnostics. `ids == nil` clears them all; otherwise only the
    /// listed ones. Failures are diagnostics, so this never affects indexed content.
    func clearFailures(ids: Set<EngineID>?) async throws

    /// A static description of the retrieval pipeline — the stages a query passes through — for
    /// diagnostics display. Synchronous and side-effect-free: it reports architecture, not state.
    var retrievalPipeline: RetrievalPipelineDescriptor { get }
}

public extension IndexEngineClient {
    /// Chunk inspection is optional for lightweight clients (fixtures, mocks);
    /// the real engine overrides this with the store-backed projection.
    func chunks(forDocument documentID: DocumentID) async throws -> [ChunkSummary] { [] }

    /// Lightweight clients (fixtures, mocks) hold no durable failures; the real engine
    /// overrides this to prune its store and in-memory diagnostics.
    func clearFailures(ids: Set<EngineID>?) async throws {}

    /// Existing lightweight clients keep compiling and explicitly report that they cannot make a
    /// durability claim. The concrete engine overrides this with independent store reads.
    func diagnosticHistory(limit: Int) async -> DiagnosticHistory {
        DiagnosticHistory(
            jobs: await jobs(limit: limit),
            failures: await failures(limit: limit),
            jobsAvailability: .notSupported,
            failuresAvailability: .notSupported
        )
    }

    /// Lightweight clients report the standard hybrid pipeline; the engine overrides this to
    /// fill in its live fusion parameter.
    var retrievalPipeline: RetrievalPipelineDescriptor { .hybridDefault }
}

/// A description of the engine's retrieval pipeline: the stages a query flows through before
/// results return. Exposed so diagnostics UIs render the real pipeline instead of hardcoding
/// stage names that silently drift when the pipeline changes.
public struct RetrievalPipelineDescriptor: Codable, Hashable, Sendable {
    /// How the candidate set is narrowed before ranking.
    public var filterStage: String
    /// Candidate-generation channels fused for ranking, in display order.
    public var candidateChannels: [String]
    /// The rank-fusion method applied across channels.
    public var fusion: String

    public init(filterStage: String, candidateChannels: [String], fusion: String) {
        self.filterStage = filterStage
        self.candidateChannels = candidateChannels
        self.fusion = fusion
    }

    /// The standard hybrid pipeline: an SQL pre-filter feeding exact, FTS5 BM25, and
    /// vector-cosine channels, fused by reciprocal rank fusion. The engine overrides `fusion`
    /// with its live `k` constant.
    public static let hybridDefault = RetrievalPipelineDescriptor(
        filterStage: "SQL pre-filter",
        candidateChannels: ["Exact match", "FTS5 BM25", "Vector cosine"],
        fusion: "Reciprocal Rank Fusion"
    )
}

public struct IndexEngineConfiguration: Sendable {
    public var embedder: any Embedder
    public var registry: PipelineRegistry
    public var defaultPolicy: IngestionPolicy
    public var retrievalProfile: RetrievalProfile
    /// Live content extractors consulted during ingestion. The registry tracks
    /// extractor *descriptors* for policy resolution; these are the runnable
    /// instances. Heavy extractors (PDF, OCR) live in optional target-separated
    /// packages and are registered here by the host app.
    public var extractors: [any ContentExtractor]

    public init(
        embedder: any Embedder = HashingEmbedder(),
        registry: PipelineRegistry = .empty,
        defaultPolicy: IngestionPolicy = .default,
        retrievalProfile: RetrievalProfile = .fast,
        extractors: [any ContentExtractor] = []
    ) {
        self.embedder = embedder
        self.registry = registry
        self.defaultPolicy = defaultPolicy
        self.retrievalProfile = retrievalProfile
        self.extractors = extractors
    }

    func resolvedForOpen() -> IndexEngineConfiguration {
        var configuration = self
        configuration.registry = registry.withBuiltIns()
        configuration.registry.policyStates = [configuration.registry.resolve(policy: defaultPolicy)]
        return configuration
    }
}

public extension IndexEngineClient {
    func documents(limit: Int) async -> [DocumentSummary] {
        let request = DocumentBrowseRequest(limit: limit)
        return (try? await browseDocuments(request).documents) ?? []
    }
}

public struct DocumentBrowseRequest: Codable, Hashable, Sendable {
    public var query: String
    public var filters: SearchFilters
    public var sort: DocumentSort
    public var limit: Int
    public var offset: Int

    public init(
        query: String = "",
        filters: SearchFilters = .init(),
        sort: DocumentSort = .ingestedAtDescending,
        limit: Int = 50,
        offset: Int = 0
    ) {
        self.query = query
        self.filters = filters
        self.sort = sort
        self.limit = max(0, limit)
        self.offset = max(0, offset)
    }
}

public enum DocumentSort: String, Codable, Hashable, Sendable, CaseIterable {
    case ingestedAtDescending
    case ingestedAtAscending
    case modifiedAtDescending
    case modifiedAtAscending
    case titleAscending
    case titleDescending
    case sizeDescending
    case sizeAscending
    case chunkCountDescending
    case chunkCountAscending
}

public struct PipelineRegistry: Sendable, Equatable {
    public static let empty = PipelineRegistry()

    public var extractors: [ComponentDescriptor]
    public var chunkers: [ComponentDescriptor]
    public var embeddingProviders: [ComponentDescriptor]
    public var vectorBackends: [ComponentDescriptor]
    public var relationProducers: [ComponentDescriptor]
    public var policyStates: [PolicyResolution]

    public init(
        extractors: [ComponentDescriptor] = [],
        chunkers: [ComponentDescriptor] = [],
        embeddingProviders: [ComponentDescriptor] = [],
        vectorBackends: [ComponentDescriptor] = [],
        relationProducers: [ComponentDescriptor] = [],
        policyStates: [PolicyResolution] = []
    ) {
        self.extractors = extractors
        self.chunkers = chunkers
        self.embeddingProviders = embeddingProviders
        self.vectorBackends = vectorBackends
        self.relationProducers = relationProducers
        self.policyStates = policyStates
    }

    public func withBuiltIns() -> PipelineRegistry {
        var registry = self
        registry.extractors = Self.insertBuiltIn(
            ComponentDescriptor(id: .builtInTextExtractor, version: "1", capabilities: ["text", "markdown", "json", "sourceCode"]),
            into: registry.extractors
        )
        registry.chunkers = Self.insertBuiltIn(
            ComponentDescriptor(id: .builtInTextChunker, version: "1", capabilities: ["plainText", "markdown", "code", "structuredJSON"]),
            into: registry.chunkers
        )
        registry.embeddingProviders = Self.insertBuiltIn(
            ComponentDescriptor(id: .builtInEmbeddingProvider, version: "1", capabilities: ["text", "query", "document"]),
            into: registry.embeddingProviders
        )
        registry.vectorBackends = Self.insertBuiltIn(
            ComponentDescriptor(id: .builtInSQLiteVectorBackend, version: "1", capabilities: ["exactScan", "sqlite", "fallback"]),
            into: registry.vectorBackends
        )
        return registry
    }

    public func resolve(policy: IngestionPolicy) -> PolicyResolution {
        var missing: [ComponentID] = []
        if !extractors.contains(where: { $0.id == policy.extractorID }) {
            missing.append(policy.extractorID)
        }
        if !chunkers.contains(where: { $0.id == policy.chunkerID }) {
            missing.append(policy.chunkerID)
        }
        if !embeddingProviders.contains(where: { $0.id == policy.embeddingProviderID }) {
            missing.append(policy.embeddingProviderID)
        }
        if !vectorBackends.contains(where: { $0.id == policy.vectorBackendID }) {
            missing.append(policy.vectorBackendID)
        }

        if missing.isEmpty {
            return PolicyResolution(
                policyID: policy.id,
                state: .satisfied,
                message: "Policy components are present and compatible."
            )
        }

        return PolicyResolution(
            policyID: policy.id,
            state: .quarantined,
            missingComponents: missing,
            message: "Missing components: \(missing.map(\.rawValue).joined(separator: ", "))"
        )
    }

    private static func insertBuiltIn(
        _ descriptor: ComponentDescriptor,
        into descriptors: [ComponentDescriptor]
    ) -> [ComponentDescriptor] {
        if descriptors.contains(where: { $0.id == descriptor.id }) {
            return descriptors
        }
        return descriptors + [descriptor]
    }
}

public struct ComponentDescriptor: Codable, Hashable, Sendable {
    public var id: ComponentID
    public var version: String
    public var capabilities: Set<String>

    public init(id: ComponentID, version: String, capabilities: Set<String> = []) {
        self.id = id
        self.version = version
        self.capabilities = capabilities
    }
}

public struct PolicyResolution: Codable, Hashable, Sendable {
    public enum State: String, Codable, Sendable {
        case satisfied
        case quarantined
        case degraded
        case invalid
    }

    public var policyID: PolicyID
    public var state: State
    public var missingComponents: [ComponentID]
    public var message: String

    public init(
        policyID: PolicyID,
        state: State,
        missingComponents: [ComponentID] = [],
        message: String = ""
    ) {
        self.policyID = policyID
        self.state = state
        self.missingComponents = missingComponents
        self.message = message
    }
}

public struct IngestionPolicy: Codable, Hashable, Sendable {
    public static let `default` = IngestionPolicy(id: "default", version: 1)

    public var id: PolicyID
    public var version: Int
    public var rawRetention: RawRetentionPolicy
    public var extractorID: ComponentID
    public var chunkerID: ComponentID
    public var embeddingProviderID: ComponentID
    public var vectorBackendID: VectorBackendID

    public init(
        id: PolicyID,
        version: Int,
        rawRetention: RawRetentionPolicy = .representationOnly,
        extractorID: ComponentID = .builtInTextExtractor,
        chunkerID: ComponentID = .builtInTextChunker,
        embeddingProviderID: ComponentID = .builtInEmbeddingProvider,
        vectorBackendID: VectorBackendID = .builtInSQLiteVectorBackend
    ) {
        self.id = id
        self.version = version
        self.rawRetention = rawRetention
        self.extractorID = extractorID
        self.chunkerID = chunkerID
        self.embeddingProviderID = embeddingProviderID
        self.vectorBackendID = vectorBackendID
    }
}

public enum RawRetentionPolicy: Codable, Hashable, Sendable {
    case inlineBlob(maxBytes: Int)
    case externalBlob(maxBytesPerSource: Int)
    case sourceReferenceOnly
    case representationOnly
    case redactedRepresentation
}

public struct IngestRequest: Sendable {
    public var jobID: JobID
    public var payloads: [SourcePayload]
    public var policy: IngestionPolicy

    public init(
        payloads: [SourcePayload],
        policy: IngestionPolicy = .default,
        jobID: JobID = EngineID(rawValue: UUID().uuidString)
    ) {
        self.jobID = jobID
        self.payloads = payloads
        self.policy = policy
    }
}

public struct DeleteRequest: Sendable, Equatable {
    public var jobID: JobID
    public var documentIDs: [DocumentID]

    public init(
        documentIDs: [DocumentID],
        jobID: JobID = EngineID(rawValue: UUID().uuidString)
    ) {
        self.jobID = jobID
        self.documentIDs = documentIDs
    }
}

public struct SourcePayload: Codable, Hashable, Sendable {
    public enum Body: Codable, Hashable, Sendable {
        case text(String)
        case binaryReference(URL)
        case preExtracted(kind: RepresentationKind, text: String)
    }

    public var documentID: DocumentID
    public var sourceID: SourceID?
    public var sourceURI: URL?
    public var displayName: String
    public var contentType: String
    public var body: Body
    public var metadata: [String: MetadataValue]
    public var clusterID: EngineID?

    public init(
        documentID: DocumentID,
        sourceID: SourceID? = nil,
        sourceURI: URL? = nil,
        displayName: String,
        contentType: String = "public.plain-text",
        body: Body,
        metadata: [String: MetadataValue] = [:],
        clusterID: EngineID? = nil
    ) {
        self.documentID = documentID
        self.sourceID = sourceID
        self.sourceURI = sourceURI
        self.displayName = displayName
        self.contentType = contentType
        self.body = body
        self.metadata = metadata
        self.clusterID = clusterID
    }

    func indexedObject(policy: IngestionPolicy) throws -> IndexedObject {
        let bodyText: String
        let representationKind: RepresentationKind
        switch body {
        case let .text(text):
            bodyText = text
            representationKind = .plainText
        case let .preExtracted(kind, text):
            bodyText = text
            representationKind = kind
        case let .binaryReference(url):
            if Self.isTextLike(contentType) {
                bodyText = try Self.readTextReference(url)
                representationKind = resolveRepresentationKind(forContentType: contentType)
            } else {
                bodyText = [displayName, url.path].joined(separator: "\n")
                representationKind = .plainText
            }
        }

        return IndexedObject(
            id: documentID.rawValue,
            type: contentType,
            title: displayName,
            body: bodyText,
            clusterID: clusterID?.rawValue,
            sourceID: sourceID?.rawValue,
            sourceURI: sourceURI,
            policyID: policy.id.rawValue,
            representationID: "\(documentID.rawValue):representation:\(representationKind.rawValue)"
        )
    }

    private static func readTextReference(_ url: URL) throws -> String {
        guard url.isFileURL else {
            throw PayloadExtractionError.unsupportedReference(url)
        }

        do {
            return try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw PayloadExtractionError.unreadableTextReference(url, String(describing: error))
        }
    }

    private static func isTextLike(_ contentType: String) -> Bool {
        if contentType == "net.daringfireball.markdown" {
            return true
        }
        guard let type = UTType(contentType) else {
            return contentType.hasPrefix("text/") || contentType.contains("text")
        }

        return type.conforms(to: .text) || type.conforms(to: .sourceCode) || type == .json
    }

}

public enum MetadataValue: Codable, Hashable, Sendable {
    case string(String)
    case integer(Int64)
    case double(Double)
    case bool(Bool)
    case strings([String])
}

public enum RepresentationKind: String, Codable, Hashable, Sendable {
    case plainText
    case markdown
    case code
    case ocrText
    case transcript
    case caption
    case structuredJSON
}

public struct SearchRequest: Sendable, Equatable {
    public var query: String
    public var mode: RetrievalMode
    public var limit: Int
    public var filters: SearchFilters
    public var allowDegradedResults: Bool

    public init(
        query: String,
        mode: RetrievalMode = .fast,
        limit: Int = 10,
        filters: SearchFilters = .init(),
        allowDegradedResults: Bool = true
    ) {
        self.query = query
        self.mode = mode
        self.limit = limit
        self.filters = filters
        self.allowDegradedResults = allowDegradedResults
    }
}

/// `CaseIterable` so surfaces that must publish the valid values — the MCP tool schema, for one —
/// derive them here instead of maintaining a hand-written copy that can fall out of step.
public enum RetrievalMode: String, Codable, Hashable, Sendable, CaseIterable {
    case fast
    case quality
    case diagnostic
}

/// Per-mode candidate budgets for the retrieval channels.
///
/// Every field here has to correspond to a stage that actually runs. An earlier
/// `maxRerankCandidates` did not: no reranker exists in the pipeline, and the value was clamped
/// to be unreachable at the one place it was read, so it described a stage `RetrievalPipelineDescriptor`
/// correctly never listed. It was removed rather than given a plausible-looking implementation.
public struct RetrievalProfile: Codable, Hashable, Sendable {
    public static let fast = RetrievalProfile(
        id: "fast",
        version: 1,
        maxFTSCandidates: 160,
        maxVectorCandidates: 160,
        maxSnippets: 80
    )
    public static let quality = RetrievalProfile(
        id: "quality",
        version: 1,
        maxFTSCandidates: 800,
        maxVectorCandidates: 800,
        maxSnippets: 160
    )
    public static let diagnostic = RetrievalProfile(
        id: "diagnostic",
        version: 1,
        maxFTSCandidates: 1_000,
        maxVectorCandidates: 1_000,
        maxSnippets: 200
    )

    public var id: EngineID
    public var version: Int
    public var maxFTSCandidates: Int
    public var maxVectorCandidates: Int
    public var maxSnippets: Int

    public init(
        id: EngineID,
        version: Int,
        maxFTSCandidates: Int = 500,
        maxVectorCandidates: Int = 500,
        maxSnippets: Int = 100
    ) {
        self.id = id
        self.version = version
        self.maxFTSCandidates = maxFTSCandidates
        self.maxVectorCandidates = maxVectorCandidates
        self.maxSnippets = maxSnippets
    }

    func resolved(for mode: RetrievalMode) -> RetrievalProfile {
        switch id.rawValue {
        case Self.fast.id.rawValue, Self.quality.id.rawValue, Self.diagnostic.id.rawValue:
            switch mode {
            case .fast:
                return .fast
            case .quality:
                return .quality
            case .diagnostic:
                return .diagnostic
            }
        default:
            return self
        }
    }

    /// Clamps the budgets to non-negative. A budget of zero disables that channel and stays
    /// zero: callers use it to switch a channel off, so nothing here may raise it to meet a
    /// return limit.
    func normalized() -> RetrievalProfile {
        var copy = self
        copy.maxFTSCandidates = max(0, copy.maxFTSCandidates)
        copy.maxVectorCandidates = max(0, copy.maxVectorCandidates)
        copy.maxSnippets = max(0, copy.maxSnippets)
        return copy
    }
}

public struct SearchFilters: Codable, Hashable, Sendable {
    public var sourceIDs: Set<SourceID>
    public var contentTypes: Set<String>
    public var clusterID: EngineID?
    public var embeddingSpaceID: EmbeddingSpaceID?
    public var policyID: PolicyID?

    public init(
        sourceIDs: Set<SourceID> = [],
        contentTypes: Set<String> = [],
        clusterID: EngineID? = nil,
        embeddingSpaceID: EmbeddingSpaceID? = nil,
        policyID: PolicyID? = nil
    ) {
        self.sourceIDs = sourceIDs
        self.contentTypes = contentTypes
        self.clusterID = clusterID
        self.embeddingSpaceID = embeddingSpaceID
        self.policyID = policyID
    }
}

public protocol ContentExtractor: Sendable {
    var id: ComponentID { get }
    var version: String { get }
    var supportedContentTypes: Set<String> { get }

    func extract(_ payload: SourcePayload, options: ExtractionOptions) async throws -> [RepresentationInput]
}

/// Marker for errors thrown by out-of-package `ContentExtractor`s so the engine can
/// categorize them as extraction failures in `FailureSnapshot` without knowing their
/// concrete type.
public protocol ContentExtractionError: Error {}

public struct ExtractionOptions: Codable, Hashable, Sendable {
    public init() {}
}

public struct RepresentationInput: Codable, Hashable, Sendable {
    public var kind: RepresentationKind
    public var text: String
    public var metadata: [String: MetadataValue]

    public init(kind: RepresentationKind, text: String, metadata: [String: MetadataValue] = [:]) {
        self.kind = kind
        self.text = text
        self.metadata = metadata
    }
}

public enum EmbeddingModality: String, Codable, Hashable, Sendable {
    case text
    case image
    case audio
    case video
    case multimodal
}
