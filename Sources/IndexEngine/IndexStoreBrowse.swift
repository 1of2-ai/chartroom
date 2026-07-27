import Foundation

extension IndexStore {
    public func count() throws -> Int {
        try counts().documentCount
    }

    public func counts() throws -> IndexStoreCounts {
        let embeddingStatement = try db.prepare("""
        SELECT COUNT(*) FROM embeddings
        JOIN chunks ON chunks.id = embeddings.chunk_id
        WHERE chunks.active = 1 AND embeddings.embedding_space_id = ?1
        """)
        embeddingStatement.bind(1, embeddingSpaceID)
        let embeddingCount = try embeddingStatement.step() ? embeddingStatement.int(0) : 0

        return IndexStoreCounts(
            documentCount: try intValue("SELECT COUNT(*) FROM documents WHERE is_deleted = 0"),
            chunkCount: try intValue("SELECT COUNT(*) FROM chunks WHERE active = 1"),
            embeddingCount: embeddingCount
        )
    }

    /// Which embedding spaces this store actually holds vectors for, against the one the active
    /// embedder searches.
    ///
    /// Every retrieval channel filters on the active `embedding_space_id`, so an embedder whose
    /// identity has changed — a new model, a new dimension, or for Jina bundles a regenerated
    /// manifest hash — matches nothing and search goes quiet without erroring. This is derived
    /// from the stored vectors rather than a recorded marker so it cannot disagree with the data
    /// it describes.
    public func embeddingSpaceCoverage() throws -> EmbeddingSpaceCoverage {
        let statement = try db.prepare("""
        SELECT embeddings.embedding_space_id, COUNT(*) FROM embeddings
        JOIN chunks ON chunks.id = embeddings.chunk_id
        WHERE chunks.active = 1
        GROUP BY embeddings.embedding_space_id
        ORDER BY embeddings.embedding_space_id ASC
        """)
        var counts: [String: Int] = [:]
        while try statement.step() {
            guard let spaceID = statement.text(0) else { continue }
            counts[spaceID] = statement.int(1)
        }
        return EmbeddingSpaceCoverage(
            activeSpaceID: EngineID(rawValue: embeddingSpaceID),
            storedSpaces: counts
                .map { EmbeddingSpaceCoverage.Space(id: EngineID(rawValue: $0.key), embeddingCount: $0.value) }
                .sorted { $0.id.rawValue < $1.id.rawValue }
        )
    }

    public func documentSummaries(request: DocumentBrowseRequest) throws -> DocumentBrowseResponse {
        let filter = documentBrowseFilterSQL(request)
        let total = try documentCount(filter: filter)
        let facets = try documentBrowseFacets(filter: filter)
        guard request.limit > 0 else {
            return DocumentBrowseResponse(request: request, documents: [], totalMatching: total, facets: facets)
        }

        let statement = try db.prepare("""
        SELECT documents.id, documents.title, documents.source_id, documents.source_uri,
               documents.content_type, documents.size, documents.ingested_at, documents.modified_at,
               documents.cluster_id,
               (SELECT COUNT(*) FROM chunks WHERE chunks.document_id = documents.id AND chunks.active = 1) AS active_chunk_count,
               (SELECT policy_id FROM chunks WHERE chunks.document_id = documents.id AND chunks.active = 1 ORDER BY created_at DESC LIMIT 1) AS active_policy_id
        FROM documents
        WHERE \(filter.whereSQL)
        ORDER BY \(documentOrderSQL(request.sort))
        LIMIT ? OFFSET ?
        """)
        var bindIndex = bind(filter, to: statement)
        statement.bind(bindIndex, request.limit)
        bindIndex += 1
        statement.bind(bindIndex, request.offset)

        var summaries: [DocumentSummary] = []
        while try statement.step() {
            summaries.append(
                DocumentSummary(
                    id: EngineID(rawValue: statement.text(0) ?? ""),
                    title: statement.text(1) ?? "",
                    sourceID: emptyStringAsNil(statement.text(2)).map(EngineID.init(rawValue:)),
                    sourceURI: statement.text(3).flatMap(URL.init(string:)),
                    contentType: statement.text(4) ?? "",
                    byteSize: statement.int(5),
                    chunkCount: statement.int(9),
                    ingestedAt: Date(timeIntervalSince1970: statement.double(6)),
                    modifiedAt: Date(timeIntervalSince1970: statement.double(7)),
                    policyID: emptyStringAsNil(statement.text(10)).map(EngineID.init(rawValue:)),
                    clusterID: emptyStringAsNil(statement.text(8)).map(EngineID.init(rawValue:))
                )
            )
        }

        return DocumentBrowseResponse(request: request, documents: summaries, totalMatching: total, facets: facets)
    }

    private func documentBrowseFilterSQL(_ request: DocumentBrowseRequest) -> CandidateFilter {
        var clauses = ["documents.is_deleted = 0"]
        var bindings: [String] = []

        let query = request.query.trimmingCharacters(in: .whitespacesAndNewlines)
        if !query.isEmpty {
            let pattern = Self.likePattern(for: query)
            clauses.append("""
            (documents.title LIKE ? ESCAPE '\\'
             OR documents.id LIKE ? ESCAPE '\\'
             OR documents.source_uri LIKE ? ESCAPE '\\')
            """)
            bindings.append(contentsOf: [pattern, pattern, pattern])
        }

        if !request.filters.sourceIDs.isEmpty {
            let sourceIDs = request.filters.sourceIDs.map(\.rawValue).sorted()
            clauses.append("documents.source_id IN (\(Self.placeholders(count: sourceIDs.count)))")
            bindings.append(contentsOf: sourceIDs)
        }

        if !request.filters.contentTypes.isEmpty {
            let contentTypes = request.filters.contentTypes.sorted()
            clauses.append("documents.content_type IN (\(Self.placeholders(count: contentTypes.count)))")
            bindings.append(contentsOf: contentTypes)
        }

        if let clusterID = request.filters.clusterID {
            clauses.append("documents.cluster_id = ?")
            bindings.append(clusterID.rawValue)
        }

        if let policyID = request.filters.policyID {
            clauses.append("""
            EXISTS (
              SELECT 1 FROM chunks
              WHERE chunks.document_id = documents.id
                AND chunks.active = 1
                AND chunks.policy_id = ?
            )
            """)
            bindings.append(policyID.rawValue)
        }

        if let embeddingSpaceID = request.filters.embeddingSpaceID {
            clauses.append("""
            EXISTS (
              SELECT 1 FROM chunks
              JOIN embeddings ON embeddings.chunk_id = chunks.id
              WHERE chunks.document_id = documents.id
                AND chunks.active = 1
                AND embeddings.embedding_space_id = ?
            )
            """)
            bindings.append(embeddingSpaceID.rawValue)
        }

        return CandidateFilter(whereSQL: clauses.joined(separator: " AND "), bindings: bindings)
    }

    private func documentCount(filter: CandidateFilter) throws -> Int {
        let statement = try db.prepare("""
        SELECT COUNT(*) FROM documents
        WHERE \(filter.whereSQL)
        """)
        _ = bind(filter, to: statement)
        return try statement.step() ? statement.int(0) : 0
    }

    private func documentBrowseFacets(filter: CandidateFilter) throws -> DocumentBrowseFacets {
        let sourceIDs = try distinctDocumentValues(
            column: "source_id",
            filter: filter
        ).map(EngineID.init(rawValue:))
        let contentTypes = try distinctDocumentValues(
            column: "content_type",
            filter: filter
        )
        return DocumentBrowseFacets(sourceIDs: sourceIDs, contentTypes: contentTypes)
    }

    private func distinctDocumentValues(column: String, filter: CandidateFilter) throws -> [String] {
        let statement = try db.prepare("""
        SELECT DISTINCT documents.\(column)
        FROM documents
        WHERE \(filter.whereSQL)
          AND documents.\(column) IS NOT NULL
          AND documents.\(column) != ''
        ORDER BY lower(documents.\(column)) ASC, documents.\(column) ASC
        """)
        _ = bind(filter, to: statement)

        var values: [String] = []
        while try statement.step() {
            if let value = emptyStringAsNil(statement.text(0)) {
                values.append(value)
            }
        }
        return values
    }

    @discardableResult
    private func bind(_ filter: CandidateFilter, to statement: Statement, startingAt startIndex: Int32 = 1) -> Int32 {
        var bindIndex = startIndex
        for value in filter.bindings {
            statement.bind(bindIndex, value)
            bindIndex += 1
        }
        return bindIndex
    }

    private func documentOrderSQL(_ sort: DocumentSort) -> String {
        switch sort {
        case .ingestedAtDescending:
            "documents.ingested_at DESC, documents.id ASC"
        case .ingestedAtAscending:
            "documents.ingested_at ASC, documents.id ASC"
        case .modifiedAtDescending:
            "documents.modified_at DESC, documents.id ASC"
        case .modifiedAtAscending:
            "documents.modified_at ASC, documents.id ASC"
        case .titleAscending:
            "lower(documents.title) ASC, documents.id ASC"
        case .titleDescending:
            "lower(documents.title) DESC, documents.id ASC"
        case .sizeDescending:
            "documents.size DESC, documents.id ASC"
        case .sizeAscending:
            "documents.size ASC, documents.id ASC"
        case .chunkCountDescending:
            "active_chunk_count DESC, documents.id ASC"
        case .chunkCountAscending:
            "active_chunk_count ASC, documents.id ASC"
        }
    }

    /// Active chunks of one document in ordinal order, with offset metadata and
    /// whether each chunk carries an embedding — the chunk-inspector projection.
    public func chunkSummaries(documentID: String) throws -> [ChunkSummary] {
        let statement = try db.prepare("""
        SELECT id, ordinal, text, heading_path, byte_start, byte_end,
               character_start, character_end, token_start, token_end, content_hash,
               EXISTS(SELECT 1 FROM embeddings WHERE embeddings.chunk_id = chunks.id) AS has_embedding,
               line_start, line_end, context_prefix, context_suffix
        FROM chunks
        WHERE document_id = ?1 AND active = 1
        ORDER BY ordinal ASC
        """)
        statement.bind(1, documentID)
        var summaries: [ChunkSummary] = []
        while try statement.step() {
            summaries.append(
                ChunkSummary(
                    id: EngineID(rawValue: statement.text(0) ?? ""),
                    documentID: EngineID(rawValue: documentID),
                    ordinal: statement.int(1),
                    text: statement.text(2) ?? "",
                    headingPath: statement.text(3) ?? "",
                    byteStart: statement.int(4),
                    byteEnd: statement.int(5),
                    characterStart: statement.int(6),
                    characterEnd: statement.int(7),
                    tokenStart: statement.int(8),
                    tokenEnd: statement.int(9),
                    lineStart: statement.optionalInt(12),
                    lineEnd: statement.optionalInt(13),
                    contextPrefix: statement.text(14) ?? "",
                    contextSuffix: statement.text(15) ?? "",
                    contentHash: statement.text(10) ?? "",
                    hasEmbedding: statement.int(11) != 0
                )
            )
        }
        return summaries
    }

    private func intValue(_ sql: String) throws -> Int {
        let statement = try db.prepare(sql)
        return try statement.step() ? statement.int(0) : 0
    }
}
