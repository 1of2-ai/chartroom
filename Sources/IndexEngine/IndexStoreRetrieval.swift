import Foundation

private let maxSearchLimit = 1_000

/// Per-chunk row data the result projection needs. A struct rather than a tuple because it is
/// assembled positionally from a SELECT and consumed by name — the two drifted apart every time
/// a column was added.
private struct ChunkRetrievalMetadata {
    var documentID: String
    var contentType: String
    var title: String
    var text: String
    var sourceID: String?
    var sourceURI: URL?
    var policyID: String?
    var representationID: String?
    var embeddingSpaceID: String?
    var lineStart: Int?
    var lineEnd: Int?
}

extension IndexStore {
    public func search(
        _ query: String,
        scope: Scope = .global,
        filters: SearchFilters = .init(),
        limit: Int = 10
    ) async throws -> [SearchHit] {
        try await searchDetailed(
            query,
            scope: scope,
            filters: filters,
            limit: limit,
            allowDegradedResults: false
        ).hits
    }

    public func searchDetailed(
        _ query: String,
        scope: Scope = .global,
        filters: SearchFilters = .init(),
        limit: Int = 10,
        allowDegradedResults: Bool = true,
        profile: RetrievalProfile = .fast
    ) async throws -> (hits: [SearchHit], diagnostics: SearchDiagnostics) {
        let started = Date.now
        let normalizedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedQuery.isEmpty, limit > 0 else {
            return (
                [],
                SearchDiagnostics(totalLatency: Date.now.timeIntervalSince(started))
            )
        }

        let clampedLimit = min(limit, maxSearchLimit)
        let profile = profile.normalized()
        let pool = max(clampedLimit * 4, 40)
        let exactLimit = min(pool, 200, profile.maxSnippets)
        let keywordLimit = min(pool, profile.maxFTSCandidates)
        let vectorLimit = min(pool, profile.maxVectorCandidates)
        let hardCluster = scope.hardScope ? scope.clusterID : nil

        let sqlStart = Date.now
        let exact = exactLimit > 0
            ? try exactIDs(normalizedQuery, hardCluster: hardCluster, filters: filters, limit: exactLimit)
            : []
        let sqlLatency = Date.now.timeIntervalSince(sqlStart)

        let ftsStart = Date.now
        let scoredKeyword = keywordLimit > 0
            ? try keywordIDs(normalizedQuery, hardCluster: hardCluster, filters: filters, limit: keywordLimit)
            : []
        let keyword = scoredKeyword.map(\.id)
        let keywordScores = Dictionary(
            scoredKeyword.map { ($0.id, $0.keywordScore) },
            uniquingKeysWith: { first, _ in first }
        )
        let ftsLatency = Date.now.timeIntervalSince(ftsStart)

        var missingChannels: [RetrievalChannel] = []
        var vector: [String] = []
        var similarities: [String: Double] = [:]
        var vectorLatency: TimeInterval?

        if vectorLimit > 0 {
            do {
                let qvec = try await embedder.embed(normalizedQuery, kind: .query)
                try validateEmbedding(qvec, kind: .query)
                let vectorStart = Date.now
                let scoredVector = try vectorIDs(qvec, hardCluster: hardCluster, filters: filters, limit: vectorLimit)
                vector = scoredVector.map(\.id)
                similarities = Dictionary(
                    scoredVector.map { ($0.id, Double($0.similarity)) },
                    uniquingKeysWith: { first, _ in first }
                )
                vectorLatency = Date.now.timeIntervalSince(vectorStart)
            } catch {
                if allowDegradedResults {
                    missingChannels.append(.vector)
                } else {
                    throw error
                }
            }
        }

        let fusionStart = Date.now
        var fused: [String: Double] = [:]
        var exactRank: [String: Int] = [:]
        var keywordRank: [String: Int] = [:]
        var vectorRank: [String: Int] = [:]

        for (index, id) in exact.enumerated() {
            fused[id, default: 0] += 1 / (Self.reciprocalRankK + Double(index + 1))
            exactRank[id] = index + 1
        }
        for (index, id) in keyword.enumerated() {
            fused[id, default: 0] += 1 / (Self.reciprocalRankK + Double(index + 1))
            keywordRank[id] = index + 1
        }
        for (index, id) in vector.enumerated() {
            fused[id, default: 0] += 1 / (Self.reciprocalRankK + Double(index + 1))
            vectorRank[id] = index + 1
        }

        if let clusterID = scope.clusterID, !scope.hardScope, scope.boostInScope != 1 {
            let clusterValues = try clusterMap(Set(fused.keys))
            for id in fused.keys where clusterValues[id] == clusterID {
                if let score = fused[id] {
                    fused[id] = score * scope.boostInScope
                }
            }
        }

        let orderedIDs = fused.sorted { lhs, rhs in
            if lhs.value == rhs.value {
                return lhs.key < rhs.key
            }
            return lhs.value > rhs.value
        }
        .prefix(clampedLimit)

        let weakThreshold = Double(embedder.weakSimilarityThreshold)
        let metadataByChunkID = try meta(for: orderedIDs.map(\.key))
        var hits: [SearchHit] = []
        hits.reserveCapacity(orderedIDs.count)
        for (chunkID, score) in orderedIDs {
            if let metadata = metadataByChunkID[chunkID] {
                let similarity = similarities[chunkID]
                // A literal title match or an FTS hit is evidence in its own right, so only a
                // vector-only hit can be weak. Without this, an exact title match on a short
                // query would be discarded for having a mediocre cosine.
                let hasLexicalEvidence = exactRank[chunkID] != nil || keywordRank[chunkID] != nil
                let isWeak = !hasLexicalEvidence && (similarity.map { $0 < weakThreshold } ?? false)
                hits.append(
                    SearchHit(
                        id: chunkID,
                        documentID: metadata.documentID,
                        chunkID: chunkID,
                        type: metadata.contentType,
                        title: metadata.title,
                        snippet: Self.snippet(text: metadata.text, query: normalizedQuery),
                        sourceID: metadata.sourceID,
                        sourceURI: metadata.sourceURI,
                        policyID: metadata.policyID,
                        representationID: metadata.representationID,
                        embeddingSpaceID: metadata.embeddingSpaceID,
                        score: score,
                        exactRank: exactRank[chunkID],
                        keywordRank: keywordRank[chunkID],
                        vectorRank: vectorRank[chunkID],
                        similarity: similarity,
                        keywordScore: keywordScores[chunkID],
                        isWeak: isWeak,
                        lineStart: metadata.lineStart,
                        lineEnd: metadata.lineEnd
                    )
                )
            }
        }

        let fusionLatency = Date.now.timeIntervalSince(fusionStart)
        // Only worth a query when the vector channel was asked for and came back empty: that is
        // the one outcome an embedding-space mismatch is indistinguishable from.
        let embeddingSpaceMismatch = vectorLimit > 0 && vector.isEmpty
            && ((try? embeddingSpaceCoverage().isOrphaned) ?? false)
        return (
            hits,
            SearchDiagnostics(
                degraded: !missingChannels.isEmpty || embeddingSpaceMismatch,
                missingChannels: missingChannels,
                embeddingSpaceMismatch: embeddingSpaceMismatch,
                sqlFilterLatency: sqlLatency,
                ftsLatency: ftsLatency,
                vectorLatency: vectorLatency,
                fusionLatency: fusionLatency,
                snippetLatency: nil,
                totalLatency: Date.now.timeIntervalSince(started)
            )
        )
    }

    private func exactIDs(
        _ query: String,
        hardCluster: String?,
        filters: SearchFilters,
        limit: Int
    ) throws -> [String] {
        let filter = candidateFilterSQL(hardCluster: hardCluster, filters: filters)
        let statement = try db.prepare("""
        SELECT chunks.id FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        JOIN embeddings ON embeddings.chunk_id = chunks.id
        WHERE chunks.active = 1 AND \(filter.whereSQL)
          AND (documents.title LIKE ? ESCAPE '\\'
               OR documents.source_uri LIKE ? ESCAPE '\\')
        ORDER BY documents.title ASC, chunks.ordinal ASC LIMIT ?
        """)
        var bindIndex: Int32 = 1
        for value in filter.bindings {
            statement.bind(bindIndex, value)
            bindIndex += 1
        }
        let pattern = Self.likePattern(for: query)
        statement.bind(bindIndex, pattern)
        bindIndex += 1
        statement.bind(bindIndex, pattern)
        bindIndex += 1
        statement.bind(bindIndex, limit)

        var ids: [String] = []
        while try statement.step() {
            if let id = statement.text(0) {
                ids.append(id)
            }
        }
        return ids
    }

    /// Ordered best-first, each paired with its BM25 relevance negated so larger is better.
    private func keywordIDs(
        _ query: String,
        hardCluster: String?,
        filters: SearchFilters,
        limit: Int
    ) throws -> [(id: String, keywordScore: Double)] {
        let tokens = HashingEmbedder.tokens(query)
        guard !tokens.isEmpty else { return [] }
        let match = tokens.map { "\"\($0)\"" }.joined(separator: " OR ")
        let filter = candidateFilterSQL(hardCluster: hardCluster, filters: filters)
        let statement = try db.prepare("""
        SELECT chunks_fts.chunk_id, bm25(chunks_fts) FROM chunks_fts
        JOIN chunks ON chunks.id = chunks_fts.chunk_id
        JOIN documents ON documents.id = chunks.document_id
        JOIN embeddings ON embeddings.chunk_id = chunks.id
        WHERE chunks_fts MATCH ? AND chunks.active = 1 AND \(filter.whereSQL)
        ORDER BY bm25(chunks_fts) ASC LIMIT ?
        """)
        statement.bind(1, match)
        var bindIndex: Int32 = 2
        for value in filter.bindings {
            statement.bind(bindIndex, value)
            bindIndex += 1
        }
        statement.bind(bindIndex, limit)

        var results: [(id: String, keywordScore: Double)] = []
        while try statement.step() {
            if let id = statement.text(0) {
                results.append((id, -statement.double(1)))
            }
        }
        return results
    }

    /// Ordered best-first, each paired with its cosine similarity to the query. The similarity
    /// is the only absolute relevance signal retrieval produces; fusion below turns it into a
    /// rank, so it has to be carried out rather than recomputed.
    ///
    /// The scan is exhaustive by construction: this is exact cosine search, and the top-k is not
    /// knowable until every candidate has been scored, so no SQL `LIMIT` can be applied without
    /// changing the answer. What the scan avoids instead is per-row cost — the query vector is
    /// dotted straight against SQLite's own buffer, and only rows that reach the running top-k
    /// pay for a Swift `String` id.
    private func vectorIDs(
        _ qvec: [Float],
        hardCluster: String?,
        filters: SearchFilters,
        limit: Int
    ) throws -> [(id: String, similarity: Float)] {
        guard limit > 0 else { return [] }
        let filter = candidateFilterSQL(hardCluster: hardCluster, filters: filters)
        let statement = try db.prepare("""
        SELECT chunks.id, vectors.vec FROM embeddings
        JOIN vectors ON vectors.id = embeddings.id
        JOIN chunks ON chunks.id = embeddings.chunk_id
        JOIN documents ON documents.id = chunks.document_id
        WHERE chunks.active = 1 AND vectors.dim = ? AND \(filter.whereSQL)
        """)
        statement.bind(1, dimension)
        var bindIndex: Int32 = 2
        for value in filter.bindings {
            statement.bind(bindIndex, value)
            bindIndex += 1
        }

        var best = BoundedTopKHits(capacity: limit)
        while try statement.step() {
            let similarity = statement.withBlob(1) { Vector.cosine(query: qvec, storedBytes: $0) }
            guard let similarity else {
                // Only now is the id worth reading: it exists to name the offending row.
                let id = statement.text(0) ?? "(unknown)"
                let actual = statement.withBlob(1) { $0.count / MemoryLayout<Float>.stride }
                throw IndexStoreError.storedVectorDimensionMismatch(id: id, expected: dimension, actual: actual)
            }
            best.insert(similarity: similarity, id: statement.text(0))
        }
        return best.sortedDescending()
    }

    private func candidateFilterSQL(
        hardCluster: String?,
        filters: SearchFilters
    ) -> CandidateFilter {
        var clauses = [
            "documents.is_deleted = 0",
            "embeddings.embedding_space_id = ?"
        ]
        var bindings = [filters.embeddingSpaceID?.rawValue ?? embeddingSpaceID]

        if let hardCluster {
            clauses.append("documents.cluster_id = ?")
            bindings.append(hardCluster)
        }
        if !filters.sourceIDs.isEmpty {
            let sourceIDs = filters.sourceIDs.map(\.rawValue).sorted()
            clauses.append("documents.source_id IN (\(Self.placeholders(count: sourceIDs.count)))")
            bindings.append(contentsOf: sourceIDs)
        }
        if !filters.contentTypes.isEmpty {
            let contentTypes = filters.contentTypes.sorted()
            clauses.append("documents.content_type IN (\(Self.placeholders(count: contentTypes.count)))")
            bindings.append(contentsOf: contentTypes)
        }
        if let policyID = filters.policyID {
            clauses.append("chunks.policy_id = ?")
            bindings.append(policyID.rawValue)
        }

        return CandidateFilter(whereSQL: clauses.joined(separator: " AND "), bindings: bindings)
    }

    private func clusterMap(_ ids: Set<String>) throws -> [String: String] {
        var map: [String: String] = [:]
        let statement = try db.prepare("""
        SELECT documents.cluster_id FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        WHERE chunks.id = ?1
        """)
        for id in ids {
            statement.reset()
            statement.bind(1, id)
            if try statement.step(), let clusterID = statement.text(0), !clusterID.isEmpty {
                map[id] = clusterID
            }
        }
        return map
    }

    /// Hydrates every result in one statement.
    ///
    /// This was one prepare-and-execute per returned hit — an N+1 whose N is the caller's page
    /// size, so a 100-result page parsed and ran 100 identical queries. One `IN (…)` returns the
    /// same rows in a single pass; the caller re-imposes the fused order, which SQL does not
    /// preserve and was never asked to.
    private func meta(for chunkIDs: [String]) throws -> [String: ChunkRetrievalMetadata] {
        guard !chunkIDs.isEmpty else { return [:] }

        // Chunk ids are bound, never interpolated; only the placeholder count varies with the page.
        let placeholders = Self.placeholders(count: chunkIDs.count, startingAt: 2)
        let statement = try db.prepare("""
        SELECT chunks.id,chunks.document_id,documents.content_type,documents.title,chunks.text,
               documents.source_id,documents.source_uri,chunks.policy_id,chunks.representation_id,
               embeddings.embedding_space_id,chunks.line_start,chunks.line_end
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        LEFT JOIN embeddings ON embeddings.chunk_id = chunks.id AND embeddings.embedding_space_id = ?1
        WHERE chunks.id IN (\(placeholders)) AND chunks.active = 1
        """)
        statement.bind(1, embeddingSpaceID)
        for (offset, chunkID) in chunkIDs.enumerated() {
            statement.bind(Int32(offset + 2), chunkID)
        }

        var metadata: [String: ChunkRetrievalMetadata] = [:]
        metadata.reserveCapacity(chunkIDs.count)
        while try statement.step() {
            guard let id = statement.text(0) else { continue }
            metadata[id] = ChunkRetrievalMetadata(
                documentID: statement.text(1) ?? "",
                contentType: statement.text(2) ?? "",
                title: statement.text(3) ?? "",
                text: statement.text(4) ?? "",
                sourceID: emptyStringAsNil(statement.text(5)),
                sourceURI: statement.text(6).flatMap(URL.init(string:)),
                policyID: emptyStringAsNil(statement.text(7)),
                representationID: emptyStringAsNil(statement.text(8)),
                embeddingSpaceID: emptyStringAsNil(statement.text(9)),
                lineStart: statement.optionalInt(10),
                lineEnd: statement.optionalInt(11)
            )
        }
        return metadata
    }

    private static func snippet(text: String, query: String) -> String {
        let limit = 240
        guard text.count > limit else { return text }
        let tokens = HashingEmbedder.tokens(query)
        let matchRange = tokens.compactMap { token in
            text.range(of: token, options: [.caseInsensitive, .diacriticInsensitive])
        }.first
        let center = matchRange.map { text.distance(from: text.startIndex, to: $0.lowerBound) } ?? 0
        let startOffset = max(0, center - 80)
        let start = text.index(text.startIndex, offsetBy: startOffset, limitedBy: text.endIndex) ?? text.startIndex
        let end = text.index(start, offsetBy: limit, limitedBy: text.endIndex) ?? text.endIndex
        let prefix = start == text.startIndex ? "" : "..."
        let suffix = end == text.endIndex ? "" : "..."
        return prefix + text[start..<end].trimmingCharacters(in: .whitespacesAndNewlines) + suffix
    }
}
