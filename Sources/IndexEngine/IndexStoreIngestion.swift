import Foundation
import UniformTypeIdentifiers

private struct PreparedChunk: Sendable {
    var id: String
    var ordinal: Int
    var text: String
    var contentHash: String
    var byteStart: Int
    var byteEnd: Int
    var characterStart: Int
    var characterEnd: Int
    var tokenStart: Int
    var tokenEnd: Int
    /// 1-based inclusive line range within the normalized representation text — the unit every
    /// downstream consumer (an editor jump, a file read, a code edit) actually takes.
    var lineStart: Int
    var lineEnd: Int
    /// Text immediately before and after the chunk. Retrieval returns a window into a document;
    /// without its edges a caller cannot tell whether a match sits mid-sentence or mid-function.
    var contextPrefix: String
    var contextSuffix: String
}

private struct PreparedEmbedding: Sendable {
    var id: String
    var chunkID: String
    var vector: [Float]
    var vectorHash: String
    var modality: EmbeddingModality
}

/// What a rebuild of the active embedding space actually managed to restore.
public struct EmbeddingRebuildSummary: Sendable, Equatable {
    /// Chunks re-embedded from their stored text.
    public var rebuiltChunkCount: Int
    /// Chunks whose vector came from a non-text modality. Their bytes are not in the store, so
    /// they cannot be rebuilt from it — the document has to be ingested again from its source.
    public var skippedNonTextChunkCount: Int

    public init(rebuiltChunkCount: Int = 0, skippedNonTextChunkCount: Int = 0) {
        self.rebuiltChunkCount = rebuiltChunkCount
        self.skippedNonTextChunkCount = skippedNonTextChunkCount
    }

    /// Whether the rebuild left the index fully readable by the active embedder.
    public var isComplete: Bool { skippedNonTextChunkCount == 0 }
}

extension IndexStore {
    /// Re-embeds every active chunk that has no vector in the embedder's current space.
    ///
    /// This is the recovery path for a changed embedder. Retrieval filters every channel on the
    /// active `embedding_space_id`, so a new model, a new dimension, or a regenerated model
    /// manifest orphans the whole index and search goes quiet. Nothing about the *content* is
    /// stale in that case — only the vectors — and the chunk text is already in the store, so the
    /// repair reads no source files and needs none of them to still exist.
    ///
    /// Idempotent: chunks that already have a vector in the active space are skipped, so an
    /// interrupted rebuild resumes rather than restarting.
    public func rebuildActiveEmbeddingSpace(batchSize: Int = 32) async throws -> EmbeddingRebuildSummary {
        let pending = try chunksMissingActiveEmbedding()
        guard !pending.isEmpty else { return EmbeddingRebuildSummary() }

        var summary = EmbeddingRebuildSummary(
            skippedNonTextChunkCount: pending.count { !$0.isTextModality }
        )
        let rebuildable = pending.filter(\.isTextModality)

        for batch in stride(from: 0, to: rebuildable.count, by: max(1, batchSize)).map({
            Array(rebuildable[$0..<min($0 + max(1, batchSize), rebuildable.count)])
        }) {
            let vectors = try await embedder.embed(batch.map(\.text), kind: .document)
            var prepared: [PreparedEmbedding] = []
            prepared.reserveCapacity(batch.count)
            for (index, chunk) in batch.enumerated() {
                let vector = vectors[index]
                try validateEmbedding(vector, kind: .document)
                prepared.append(
                    PreparedEmbedding(
                        id: "\(chunk.id):embedding:\(embeddingSpaceID)",
                        chunkID: chunk.id,
                        vector: vector,
                        vectorHash: Self.stableHash(
                            "vector",
                            Vector.toBytes(vector).map(String.init).joined(separator: ",")
                        ),
                        modality: .text
                    )
                )
            }

            // One transaction per batch: a rebuild can span a large corpus, and a failure part way
            // through should leave the batches already written intact for the resume to skip.
            let now = Date.now.timeIntervalSince1970
            try db.transaction {
                try self.persistEmbeddings(prepared, activeEmbeddingSpaceID: self.embeddingSpaceID, now: now)
            }
            summary.rebuiltChunkCount += prepared.count
        }

        return summary
    }

    private struct RebuildableChunk {
        var id: String
        var text: String
        var isTextModality: Bool
    }

    /// Active chunks with no vector in the current space, tagged with the modality any *other*
    /// space recorded for them — that is how an image-derived chunk is recognised without its file.
    private func chunksMissingActiveEmbedding() throws -> [RebuildableChunk] {
        let statement = try db.prepare("""
        SELECT chunks.id, chunks.text, (
          SELECT modality FROM embeddings WHERE embeddings.chunk_id = chunks.id LIMIT 1
        ) FROM chunks
        WHERE chunks.active = 1 AND NOT EXISTS (
          SELECT 1 FROM embeddings
          WHERE embeddings.chunk_id = chunks.id AND embeddings.embedding_space_id = ?1
        )
        ORDER BY chunks.document_id ASC, chunks.ordinal ASC
        """)
        statement.bind(1, embeddingSpaceID)

        var chunks: [RebuildableChunk] = []
        while try statement.step() {
            guard let id = statement.text(0) else { continue }
            let recordedModality = statement.text(2)
            chunks.append(
                RebuildableChunk(
                    id: id,
                    text: statement.text(1) ?? "",
                    // No recorded modality means nothing was ever embedded for this chunk, which
                    // only happens for text: image documents write their vector at ingest.
                    isTextModality: recordedModality == nil
                        || recordedModality == EmbeddingModality.text.rawValue
                )
            )
        }
        return chunks
    }

    public func upsert(_ obj: IndexedObject) async throws {
        let normalizedBody = Self.normalizedText(obj.body)
        let policyID = obj.policyID ?? IngestionPolicy.default.id.rawValue
        let policyVersion = IngestionPolicy.default.version
        let sourceID = obj.sourceID ?? "manual"
        let sourceURI = obj.sourceURI?.absoluteString ?? ""
        let clusterID = obj.clusterID ?? ""
        let activeEmbeddingSpaceID = obj.embeddingSpaceID ?? embeddingSpaceID
        let documentHash = Self.stableHash("document", obj.type, obj.title, normalizedBody)
        let versionID = "\(obj.id):version:\(documentHash)"
        let lineageID = obj.representationID ?? "\(obj.id):representation:plainText"
        let representationHash = Self.stableHash("representation", normalizedBody)
        let representationID = "\(lineageID):\(representationHash)"
        let chunks = makeChunks(
            documentID: obj.id,
            lineageID: lineageID,
            policyID: policyID,
            policyVersion: policyVersion,
            text: normalizedBody
        )

        let imageURL = imageEmbeddingSource(for: obj)
        let upsertGeneration = nextUpsertGeneration(for: obj.id)

        // Compute one vector per chunk. Text documents embed all chunks in a single batch so the
        // embedder can group them by bucket and run them together; image documents embed the file
        // once and attach that vector to the (single) chunk.
        let vectors: [[Float]]
        let modality: EmbeddingModality
        if let imageURL {
            let vector = try await embedder.embedImage(at: imageURL)
            vectors = Array(repeating: vector, count: chunks.count)
            modality = .image
        } else {
            vectors = try await embedder.embed(chunks.map(\.text), kind: .document)
            modality = .text
        }

        var embeddings: [PreparedEmbedding] = []
        embeddings.reserveCapacity(chunks.count)
        for (index, chunk) in chunks.enumerated() {
            let vector = vectors[index]
            try validateEmbedding(vector, kind: .document)
            embeddings.append(
                PreparedEmbedding(
                    id: "\(chunk.id):embedding:\(activeEmbeddingSpaceID)",
                    chunkID: chunk.id,
                    vector: vector,
                    vectorHash: Self.stableHash("vector", Vector.toBytes(vector).map(String.init).joined(separator: ",")),
                    modality: modality
                )
            )
        }

        guard isCurrentUpsertGeneration(upsertGeneration, for: obj.id) else { return }

        let now = Date.now.timeIntervalSince1970
        try db.transaction {
            try self.persistObjectProjection(obj, now: now, embeddingSpaceID: activeEmbeddingSpaceID)
            try self.persistSource(id: sourceID, sourceURI: sourceURI, now: now)
            try self.persistDocument(
                obj: obj,
                sourceID: sourceID,
                sourceURI: sourceURI,
                clusterID: clusterID,
                documentHash: documentHash,
                versionID: versionID,
                now: now
            )
            try self.persistDocumentVersion(
                id: versionID,
                documentID: obj.id,
                policyID: policyID,
                policyVersion: policyVersion,
                contentHash: documentHash,
                now: now
            )
            try self.persistRepresentation(
                id: representationID,
                lineageID: lineageID,
                documentID: obj.id,
                versionID: versionID,
                policyID: policyID,
                policyVersion: policyVersion,
                contentType: obj.type,
                text: normalizedBody,
                contentHash: representationHash,
                now: now
            )
            try self.removeChunkRecords(
                documentID: obj.id,
                retaining: Set(chunks.map(\.id)),
                activeEmbeddingSpaceID: activeEmbeddingSpaceID
            )
            try self.persistChunks(
                chunks,
                documentID: obj.id,
                versionID: versionID,
                lineageID: lineageID,
                representationID: representationID,
                policyID: policyID,
                policyVersion: policyVersion,
                title: obj.title,
                sourceURI: sourceURI,
                now: now
            )
            try self.persistEmbeddings(
                embeddings,
                activeEmbeddingSpaceID: activeEmbeddingSpaceID,
                now: now
            )
        }
    }

    @discardableResult
    public func delete(id: String) throws -> Bool {
        guard try activeDocumentExists(id: id) else { return false }
        let now = Date.now.timeIntervalSince1970
        try db.transaction {
            try removeChunkRecords(documentID: id)

            let document = try db.prepare("""
            UPDATE documents SET is_deleted = 1, updated_at = ?2 WHERE id = ?1
            """)
            document.bind(1, id)
            document.bind(2, now)
            try document.step()

            for sql in [
                "DELETE FROM representations WHERE document_id=?1",
                "DELETE FROM representation_lineages WHERE document_id=?1",
                "DELETE FROM objects WHERE id=?1"
            ] {
                let statement = try db.prepare(sql)
                statement.bind(1, id)
                try statement.step()
            }
        }
        return true
    }

    public func persistPolicy(_ policy: IngestionPolicy, resolutions: [PolicyResolution]) throws {
        try db.transaction {
            let policyStatement = try db.prepare("""
            INSERT INTO policies(
              id,version,raw_retention,extractor_id,chunker_id,embedding_provider_id,vector_backend_id,updated_at
            ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)
            ON CONFLICT(id) DO UPDATE SET version=excluded.version,
              raw_retention=excluded.raw_retention,
              extractor_id=excluded.extractor_id,
              chunker_id=excluded.chunker_id,
              embedding_provider_id=excluded.embedding_provider_id,
              vector_backend_id=excluded.vector_backend_id,
              updated_at=excluded.updated_at
            """)
            policyStatement.bind(1, policy.id.rawValue)
            policyStatement.bind(2, policy.version)
            policyStatement.bind(3, policy.rawRetention.storageCode)
            policyStatement.bind(4, policy.extractorID.rawValue)
            policyStatement.bind(5, policy.chunkerID.rawValue)
            policyStatement.bind(6, policy.embeddingProviderID.rawValue)
            policyStatement.bind(7, policy.vectorBackendID.rawValue)
            policyStatement.bind(8, Date.now.timeIntervalSince1970)
            try policyStatement.step()

            let delete = try db.prepare("DELETE FROM policy_component_resolutions WHERE policy_id = ?1")
            delete.bind(1, policy.id.rawValue)
            try delete.step()

            for resolution in resolutions {
                let statement = try db.prepare("""
                INSERT INTO policy_component_resolutions(policy_id,component_id,component_kind,state,message)
                VALUES(?1,?2,?3,?4,?5)
                """)
                statement.bind(1, resolution.policyID.rawValue)
                statement.bind(2, resolution.missingComponents.map(\.rawValue).joined(separator: ","))
                statement.bind(3, "policy")
                statement.bind(4, resolution.state.rawValue)
                statement.bind(5, resolution.message)
                try statement.step()
            }
        }
    }

    private func persistObjectProjection(_ obj: IndexedObject, now: TimeInterval, embeddingSpaceID: String) throws {
        let statement = try db.prepare("""
        INSERT INTO objects(
          id,type,title,cluster_id,source_id,source_uri,policy_id,representation_id,embedding_space_id,model_id,updated_at
        )
        VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)
        ON CONFLICT(id) DO UPDATE SET type=excluded.type,
          title=excluded.title,
          cluster_id=excluded.cluster_id,
          source_id=excluded.source_id,
          source_uri=excluded.source_uri,
          policy_id=excluded.policy_id,
          representation_id=excluded.representation_id,
          embedding_space_id=excluded.embedding_space_id,
          model_id=excluded.model_id,
          updated_at=excluded.updated_at
        """)
        statement.bind(1, obj.id)
        statement.bind(2, obj.type)
        statement.bind(3, obj.title)
        bindNullable(statement, 4, obj.clusterID)
        bindNullable(statement, 5, obj.sourceID)
        bindNullable(statement, 6, obj.sourceURI?.absoluteString)
        bindNullable(statement, 7, obj.policyID)
        bindNullable(statement, 8, obj.representationID)
        statement.bind(9, embeddingSpaceID)
        statement.bind(10, modelID)
        statement.bind(11, now)
        try statement.step()
    }

    private func persistSource(id: String, sourceURI: String, now: TimeInterval) throws {
        let statement = try db.prepare("""
        INSERT INTO sources(id,connector_kind,connector_instance_id,source_uri,external_stable_id,updated_at)
        VALUES(?1,?2,?3,?4,?5,?6)
        ON CONFLICT(id) DO UPDATE SET source_uri=excluded.source_uri, updated_at=excluded.updated_at
        """)
        statement.bind(1, id)
        statement.bind(2, id)
        statement.bind(3, id)
        statement.bind(4, sourceURI)
        statement.bind(5, sourceURI)
        statement.bind(6, now)
        try statement.step()
    }

    private func persistDocument(
        obj: IndexedObject,
        sourceID: String,
        sourceURI: String,
        clusterID: String,
        documentHash: String,
        versionID: String,
        now: TimeInterval
    ) throws {
        let statement = try db.prepare("""
        INSERT INTO documents(
          id,source_id,source_uri,external_id,content_hash,version,active_version_id,title,
          content_type,file_extension,size,created_at,modified_at,ingested_at,updated_at,
          is_deleted,permission_scope_id,provenance,cluster_id
        ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,0,?16,?17,?18)
        ON CONFLICT(id) DO UPDATE SET source_id=excluded.source_id,
          source_uri=excluded.source_uri,
          external_id=excluded.external_id,
          content_hash=excluded.content_hash,
          version=documents.version + CASE WHEN documents.content_hash = excluded.content_hash THEN 0 ELSE 1 END,
          active_version_id=excluded.active_version_id,
          title=excluded.title,
          content_type=excluded.content_type,
          file_extension=excluded.file_extension,
          size=excluded.size,
          modified_at=excluded.modified_at,
          ingested_at=excluded.ingested_at,
          updated_at=excluded.updated_at,
          is_deleted=0,
          provenance=excluded.provenance,
          cluster_id=excluded.cluster_id
        """)
        statement.bind(1, obj.id)
        statement.bind(2, sourceID)
        statement.bind(3, sourceURI)
        statement.bind(4, obj.id)
        statement.bind(5, documentHash)
        statement.bind(6, 1)
        statement.bind(7, versionID)
        statement.bind(8, obj.title)
        statement.bind(9, obj.type)
        statement.bind(10, obj.sourceURI?.pathExtension ?? "")
        statement.bind(11, obj.body.utf8.count)
        statement.bind(12, now)
        statement.bind(13, now)
        statement.bind(14, now)
        statement.bind(15, now)
        statement.bind(16, "")
        statement.bind(17, "{}")
        statement.bind(18, clusterID)
        try statement.step()
    }

    private func persistDocumentVersion(
        id: String,
        documentID: String,
        policyID: String,
        policyVersion: Int,
        contentHash: String,
        now: TimeInterval
    ) throws {
        let statement = try db.prepare("""
        INSERT OR IGNORE INTO document_versions(
          id,document_id,content_hash,policy_id,policy_version,created_at,retained_state
        ) VALUES(?1,?2,?3,?4,?5,?6,?7)
        """)
        statement.bind(1, id)
        statement.bind(2, documentID)
        statement.bind(3, contentHash)
        statement.bind(4, policyID)
        statement.bind(5, policyVersion)
        statement.bind(6, now)
        statement.bind(7, AvailabilityState.representationAvailable.rawValue)
        try statement.step()
    }

    private func persistRepresentation(
        id: String,
        lineageID: String,
        documentID: String,
        versionID: String,
        policyID: String,
        policyVersion: Int,
        contentType: String,
        text: String,
        contentHash: String,
        now: TimeInterval
    ) throws {
        let kind = resolveRepresentationKind(forContentType: contentType)
        let lineages = try db.prepare("""
        INSERT INTO representation_lineages(
          id,document_id,policy_id,representation_kind,active_representation_id,updated_at
        ) VALUES(?1,?2,?3,?4,?5,?6)
        ON CONFLICT(id) DO UPDATE SET active_representation_id=excluded.active_representation_id,
          updated_at=excluded.updated_at
        """)
        lineages.bind(1, lineageID)
        lineages.bind(2, documentID)
        lineages.bind(3, policyID)
        lineages.bind(4, kind.rawValue)
        lineages.bind(5, id)
        lineages.bind(6, now)
        try lineages.step()

        // Superseded representations are deleted, not deactivated: they carry the
        // full extracted text, nothing reads inactive rows, and keeping one row per
        // edit grew the store without bound. The lineage keeps exactly the active
        // representation (the upsert below reuses the row when content reverts).
        let oldRepresentations = try db.prepare("DELETE FROM representations WHERE lineage_id = ?1 AND id <> ?2")
        oldRepresentations.bind(1, lineageID)
        oldRepresentations.bind(2, id)
        try oldRepresentations.step()

        let statement = try db.prepare("""
        INSERT INTO representations(
          id,lineage_id,document_id,document_version_id,representation_kind,extractor_id,
          extractor_version,policy_id,policy_version,text,token_count,content_hash,
          source_material_availability,upstream_dependency_state,active,created_at
        ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,1,?15)
        ON CONFLICT(id) DO UPDATE SET active=1, created_at=excluded.created_at
        """)
        statement.bind(1, id)
        statement.bind(2, lineageID)
        statement.bind(3, documentID)
        statement.bind(4, versionID)
        statement.bind(5, kind.rawValue)
        statement.bind(6, ComponentID.builtInTextExtractor.rawValue)
        statement.bind(7, "1")
        statement.bind(8, policyID)
        statement.bind(9, policyVersion)
        statement.bind(10, text)
        statement.bind(11, Self.tokenCount(text))
        statement.bind(12, contentHash)
        statement.bind(13, AvailabilityState.representationAvailable.rawValue)
        statement.bind(14, "satisfied")
        statement.bind(15, now)
        try statement.step()
    }

    /// Clears a document's chunk records, with their FTS, embedding, and vector rows, ahead of a
    /// re-ingest or a delete. Superseded chunk state is deleted outright: nothing reads
    /// `active = 0` rows, and keeping them grew the store on every edit. Runs inside the caller's
    /// transaction, so search never observes the document without its replacement chunks.
    ///
    /// Chunk IDs hash their occurrence and content, so an ID that appears in both the old and
    /// new chunk sets denotes the same text at the same position. Those rows are left in place
    /// for `persistChunks` to update, and only the *active* embedding space is cleared for them —
    /// another model's vectors for that chunk are still valid and must survive. Chunks that are
    /// not retained can never be referenced again, so they take every space's embeddings with them.
    ///
    /// - Parameters:
    ///   - retaining: Chunk IDs the caller is about to re-persist. Empty means the document is
    ///     going away entirely, which clears every chunk in every space.
    ///   - activeEmbeddingSpaceID: The space being rebuilt. `nil` clears all spaces.
    private func removeChunkRecords(
        documentID: String,
        retaining retainedChunkIDs: Set<String> = [],
        activeEmbeddingSpaceID: String? = nil
    ) throws {
        let existing = try allChunkIDs(documentID: documentID)
        let obsolete = existing.filter { !retainedChunkIDs.contains($0) }
        let retained = existing.filter { retainedChunkIDs.contains($0) }

        for chunkID in obsolete {
            try deleteFTSChunk(chunkID: chunkID)
        }
        try deleteEmbeddingsAndVectors(chunkIDs: obsolete, embeddingSpaceID: nil)

        if let activeEmbeddingSpaceID, !retained.isEmpty {
            try deleteEmbeddingsAndVectors(chunkIDs: retained, embeddingSpaceID: activeEmbeddingSpaceID)
        }

        guard !obsolete.isEmpty else { return }
        let chunks = try db.prepare("""
        DELETE FROM chunks WHERE document_id = ?1
          AND id IN (\(Self.placeholders(count: obsolete.count, startingAt: 2)))
        """)
        chunks.bind(1, documentID)
        for (offset, chunkID) in obsolete.enumerated() {
            chunks.bind(Int32(offset + 2), chunkID)
        }
        try chunks.step()
    }

    private func persistChunks(
        _ chunks: [PreparedChunk],
        documentID: String,
        versionID: String,
        lineageID: String,
        representationID: String,
        policyID: String,
        policyVersion: Int,
        title: String,
        sourceURI: String,
        now: TimeInterval
    ) throws {
        // One prepare per statement, reset per chunk — preparing inside the loop
        // was measurable on large documents.
        let statement = try db.prepare("""
        INSERT INTO chunks(
          id,document_id,document_version_id,representation_id,representation_lineage_id,ordinal,
          chunker_id,chunker_version,policy_id,policy_version,text,context_prefix,context_suffix,
          heading_path,byte_start,byte_end,character_start,character_end,token_start,token_end,
          page_start,page_end,section_label,content_hash,active,availability_state,created_at,
          line_start,line_end
        ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,1,?25,?26,?27,?28)
        ON CONFLICT(id) DO UPDATE SET text=excluded.text,
          active=1,
          availability_state=excluded.availability_state,
          created_at=excluded.created_at,
          context_prefix=excluded.context_prefix,
          context_suffix=excluded.context_suffix,
          line_start=excluded.line_start,
          line_end=excluded.line_end
        """)
        let ftsDelete = try db.prepare("DELETE FROM chunks_fts WHERE chunk_id = ?1")
        let fts = try db.prepare("""
        INSERT INTO chunks_fts(text,title,path,heading_path,chunk_id)
        VALUES(?1,?2,?3,?4,?5)
        """)
        for chunk in chunks {
            statement.reset()
            statement.bind(1, chunk.id)
            statement.bind(2, documentID)
            statement.bind(3, versionID)
            statement.bind(4, representationID)
            statement.bind(5, lineageID)
            statement.bind(6, chunk.ordinal)
            statement.bind(7, ComponentID.builtInTextChunker.rawValue)
            statement.bind(8, Self.builtInChunkerVersion)
            statement.bind(9, policyID)
            statement.bind(10, policyVersion)
            statement.bind(11, chunk.text)
            statement.bind(12, chunk.contextPrefix)
            statement.bind(13, chunk.contextSuffix)
            // heading_path stays empty until the chunker understands document structure; the
            // built-in chunker splits on blank lines and knows nothing about headings or symbols.
            statement.bind(14, "")
            statement.bind(15, chunk.byteStart)
            statement.bind(16, chunk.byteEnd)
            statement.bind(17, chunk.characterStart)
            statement.bind(18, chunk.characterEnd)
            statement.bind(19, chunk.tokenStart)
            statement.bind(20, chunk.tokenEnd)
            statement.bind(21, 0)
            statement.bind(22, 0)
            statement.bind(23, "")
            statement.bind(24, chunk.contentHash)
            statement.bind(25, AvailabilityState.chunkTextAvailable.rawValue)
            statement.bind(26, now)
            statement.bind(27, chunk.lineStart)
            statement.bind(28, chunk.lineEnd)
            try statement.step()

            ftsDelete.reset()
            ftsDelete.bind(1, chunk.id)
            try ftsDelete.step()

            fts.reset()
            fts.bind(1, chunk.text)
            fts.bind(2, title)
            fts.bind(3, sourceURI)
            fts.bind(4, "")
            fts.bind(5, chunk.id)
            try fts.step()
        }
    }

    private func persistEmbeddings(
        _ embeddings: [PreparedEmbedding],
        activeEmbeddingSpaceID: String,
        now: TimeInterval
    ) throws {
        // One prepare per statement, reset per embedding — matching the chunk writer above, which
        // hoisted its prepares for exactly this reason. These two were still re-preparing on every
        // iteration, so a document's embedding write paid for a full parse per chunk.
        let record = try db.prepare("""
        INSERT INTO embeddings(
          id,chunk_id,embedding_space_id,model_id,model_version,dimension,modality,prompt_kind,
          vector_backend_id,vector_backend_version,vector_hash,created_at
        ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)
        ON CONFLICT(id) DO UPDATE SET vector_hash=excluded.vector_hash,
          created_at=excluded.created_at
        """)
        let vector = try db.prepare("""
        INSERT INTO vectors(id,dim,vec) VALUES(?1,?2,?3)
        ON CONFLICT(id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec
        """)

        for embedding in embeddings {
            record.reset()
            record.bind(1, embedding.id)
            record.bind(2, embedding.chunkID)
            record.bind(3, activeEmbeddingSpaceID)
            record.bind(4, modelID)
            record.bind(5, "1")
            record.bind(6, dimension)
            record.bind(7, embedding.modality.rawValue)
            record.bind(8, EmbedKind.document.rawValue)
            record.bind(9, vectorBackendID)
            record.bind(10, vectorBackendVersion)
            record.bind(11, embedding.vectorHash)
            record.bind(12, now)
            try record.step()

            vector.reset()
            vector.bind(1, embedding.id)
            vector.bind(2, dimension)
            vector.bindBlob(3, Vector.toBytes(embedding.vector))
            try vector.step()
        }
    }

    private func activeChunkIDs(documentID: String) throws -> [String] {
        let statement = try db.prepare("SELECT id FROM chunks WHERE document_id = ?1 AND active = 1")
        statement.bind(1, documentID)
        var ids: [String] = []
        while try statement.step() {
            if let id = statement.text(0) {
                ids.append(id)
            }
        }
        return ids
    }

    private func allChunkIDs(documentID: String) throws -> [String] {
        let statement = try db.prepare("SELECT id FROM chunks WHERE document_id = ?1")
        statement.bind(1, documentID)
        var ids: [String] = []
        while try statement.step() {
            if let id = statement.text(0) {
                ids.append(id)
            }
        }
        return ids
    }

    private func activeDocumentExists(id: String) throws -> Bool {
        let statement = try db.prepare("SELECT 1 FROM documents WHERE id = ?1 AND is_deleted = 0 LIMIT 1")
        statement.bind(1, id)
        return try statement.step()
    }

    /// - Parameter embeddingSpaceID: Restricts the delete to one space. `nil` removes the chunk's
    ///   embeddings in every space, which is only correct when the chunk itself is going away.
    private func deleteEmbeddingsAndVectors(chunkIDs: [String], embeddingSpaceID: String?) throws {
        guard !chunkIDs.isEmpty else { return }
        let spacePredicate = embeddingSpaceID == nil ? "" : " AND embedding_space_id = ?2"
        for chunkID in chunkIDs {
            let vectors = try db.prepare("""
            DELETE FROM vectors WHERE id IN (
              SELECT id FROM embeddings WHERE chunk_id = ?1\(spacePredicate)
            )
            """)
            vectors.bind(1, chunkID)
            if let embeddingSpaceID { vectors.bind(2, embeddingSpaceID) }
            try vectors.step()

            let embeddings = try db.prepare("DELETE FROM embeddings WHERE chunk_id = ?1\(spacePredicate)")
            embeddings.bind(1, chunkID)
            if let embeddingSpaceID { embeddings.bind(2, embeddingSpaceID) }
            try embeddings.step()
        }
    }

    private func deleteFTSChunk(chunkID: String) throws {
        let statement = try db.prepare("DELETE FROM chunks_fts WHERE chunk_id = ?1")
        statement.bind(1, chunkID)
        try statement.step()
    }

    private func nextUpsertGeneration(for documentID: String) -> UInt64 {
        let next = (upsertGenerations[documentID] ?? 0) &+ 1
        upsertGenerations[documentID] = next
        return next
    }

    private func isCurrentUpsertGeneration(_ generation: UInt64, for documentID: String) -> Bool {
        upsertGenerations[documentID] == generation
    }

    /// The local image file to embed for this object, or nil to use the text path. Images
    /// embed into the same vector space as text (enabling cross-modal retrieval) only when
    /// the active embedder advertises image support and the object references a local image
    /// file. Otherwise an image falls back to embedding its filename text.
    private func imageEmbeddingSource(for obj: IndexedObject) -> URL? {
        let contentType = UTType(obj.type) ?? UTType(mimeType: obj.type)
        guard embedder.supportsImageEmbedding,
              let type = contentType, type.conforms(to: .image),
              let uri = obj.sourceURI, uri.isFileURL
        else { return nil }
        return uri
    }

    private func makeChunks(
        documentID: String,
        lineageID: String,
        policyID: String,
        policyVersion: Int,
        text: String
    ) -> [PreparedChunk] {
        guard !text.isEmpty else {
            let contentHash = Self.stableHash("chunk", "")
            return [
                PreparedChunk(
                    id: Self.chunkID(
                        documentID: documentID,
                        lineageID: lineageID,
                        policyID: policyID,
                        policyVersion: policyVersion,
                        occurrence: 0,
                        contentHash: contentHash
                    ),
                    ordinal: 0,
                    text: "",
                    contentHash: contentHash,
                    byteStart: 0,
                    byteEnd: 0,
                    characterStart: 0,
                    characterEnd: 0,
                    tokenStart: 0,
                    tokenEnd: 0,
                    lineStart: 1,
                    lineEnd: 1,
                    contextPrefix: "",
                    contextSuffix: ""
                )
            ]
        }

        var chunks: [PreparedChunk] = []
        var start = text.startIndex
        var ordinal = 0
        var occurrenceByHash: [String: Int] = [:]

        // Prefix offsets measured incrementally: chunk starts are monotone, so each
        // advance costs the distance moved instead of rescanning the whole prefix
        // per chunk (which made this O(n²) on long documents).
        var measuredIndex = text.startIndex
        var measuredCharacters = 0
        var measuredBytes = 0
        var measuredTokens = 0
        var measuredNewlines = 0
        func advanceMeasurements(to index: String.Index) {
            guard index > measuredIndex else { return }
            let segment = text[measuredIndex..<index]
            measuredCharacters += segment.count
            measuredBytes += segment.utf8.count
            measuredNewlines += segment.lazy.filter(\.isNewline).count
            var segmentTokens = Self.tokenCount(String(segment))
            // The tokenizer splits on non-alphanumerics; a token straddling the
            // previous measurement boundary would be counted by both segments.
            if measuredIndex > text.startIndex, segmentTokens > 0 {
                let before = text[text.index(before: measuredIndex)]
                let after = text[measuredIndex]
                if (before.isLetter || before.isNumber), (after.isLetter || after.isNumber) {
                    segmentTokens -= 1
                }
            }
            measuredTokens += segmentTokens
            measuredIndex = index
        }

        while start < text.endIndex {
            let hardEnd = text.index(start, offsetBy: defaultChunkCharacterLimit, limitedBy: text.endIndex) ?? text.endIndex
            let end = Self.preferredChunkEnd(in: text, start: start, hardEnd: hardEnd)
            let effectiveRange = Self.trimmedRange(in: text, range: start..<end)
            let effectiveText = String(text[effectiveRange])
            let contentHash = Self.stableHash("chunk", effectiveText)
            let occurrence = occurrenceByHash[contentHash, default: 0]
            occurrenceByHash[contentHash] = occurrence + 1
            advanceMeasurements(to: effectiveRange.lowerBound)
            let lineStart = measuredNewlines + 1
            chunks.append(
                PreparedChunk(
                    id: Self.chunkID(
                        documentID: documentID,
                        lineageID: lineageID,
                        policyID: policyID,
                        policyVersion: policyVersion,
                        occurrence: occurrence,
                        contentHash: contentHash
                    ),
                    ordinal: ordinal,
                    text: effectiveText,
                    contentHash: contentHash,
                    byteStart: measuredBytes,
                    byteEnd: measuredBytes + effectiveText.utf8.count,
                    characterStart: measuredCharacters,
                    characterEnd: measuredCharacters + effectiveText.count,
                    tokenStart: measuredTokens,
                    tokenEnd: measuredTokens + Self.tokenCount(effectiveText),
                    lineStart: lineStart,
                    // Inclusive: a chunk with no newline occupies exactly its starting line.
                    lineEnd: lineStart + effectiveText.lazy.filter(\.isNewline).count,
                    contextPrefix: Self.context(in: text, before: effectiveRange.lowerBound),
                    contextSuffix: Self.context(in: text, after: effectiveRange.upperBound)
                )
            )

            if end == text.endIndex {
                break
            }

            let nextStart = text.index(end, offsetBy: -min(defaultChunkOverlap, text.distance(from: start, to: end)), limitedBy: start) ?? end
            if nextStart <= start {
                start = end
            } else {
                start = nextStart
            }
            ordinal += 1
        }

        return chunks
    }

    private static func preferredChunkEnd(in text: String, start: String.Index, hardEnd: String.Index) -> String.Index {
        guard hardEnd < text.endIndex else { return text.endIndex }
        let minimumDistance = 400
        let searchRange = start..<hardEnd
        var best = hardEnd
        var current = searchRange.lowerBound
        while current < searchRange.upperBound {
            if text[current].isNewline, text.distance(from: start, to: current) >= minimumDistance {
                best = current
            }
            current = text.index(after: current)
        }
        return best
    }

    /// Characters of adjacent text stored on each side of a chunk. Enough to show where a chunk
    /// sits without materially growing the row: chunks are ~1600 characters, so this is ~15%.
    private static let contextWindowCharacterLimit = 120

    private static func context(in text: String, before index: String.Index) -> String {
        guard index > text.startIndex else { return "" }
        let lower = text.index(index, offsetBy: -contextWindowCharacterLimit, limitedBy: text.startIndex)
            ?? text.startIndex
        return String(text[lower..<index])
    }

    private static func context(in text: String, after index: String.Index) -> String {
        guard index < text.endIndex else { return "" }
        let upper = text.index(index, offsetBy: contextWindowCharacterLimit, limitedBy: text.endIndex)
            ?? text.endIndex
        return String(text[index..<upper])
    }

    private static func trimmedRange(in text: String, range: Range<String.Index>) -> Range<String.Index> {
        var lower = range.lowerBound
        var upper = range.upperBound

        while lower < upper, text[lower].isWhitespace {
            lower = text.index(after: lower)
        }
        while lower < upper {
            let beforeUpper = text.index(before: upper)
            guard text[beforeUpper].isWhitespace else { break }
            upper = beforeUpper
        }

        return lower < upper ? lower..<upper : range
    }

    /// Version of the built-in chunker, part of every chunk identity and row.
    static let builtInChunkerVersion = "1"

    /// Chunk identity is content-addressed: chunker id+version, policy, content hash, and an
    /// occurrence index for repeated identical text. Position (`ordinal`) is deliberately NOT
    /// part of the identity — inserting a paragraph must not invalidate the chunks (and cached
    /// embeddings) of every unchanged paragraph after it.
    private static func chunkID(
        documentID: String,
        lineageID: String,
        policyID: String,
        policyVersion: Int,
        occurrence: Int,
        contentHash: String
    ) -> String {
        let hash = stableHash(
            lineageID,
            policyID,
            String(policyVersion),
            ComponentID.builtInTextChunker.rawValue,
            builtInChunkerVersion,
            String(occurrence),
            contentHash
        )
        return "\(documentID):chunk:\(hash)"
    }

    private static func normalizedText(_ text: String) -> String {
        text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .precomposedStringWithCanonicalMapping
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func tokenCount(_ text: String) -> Int {
        HashingEmbedder.tokens(text).count
    }

    private static func stableHash(_ parts: String...) -> String {
        var hasher = StableFNV1A()
        for part in parts {
            hasher.update(part)
        }
        return String(hasher.value, radix: 16)
    }
}

private enum AvailabilityState: String {
    case rawAvailable
    case representationAvailable
    case chunkTextAvailable
    case externalReferenceAvailable
    case requiresRefetch
    case unrecoverable
}

private func bindNullable(_ statement: Statement, _ index: Int32, _ value: String?) {
    if let value {
        statement.bind(index, value)
    } else {
        statement.bindNull(index)
    }
}

private extension RawRetentionPolicy {
    var storageCode: String {
        switch self {
        case let .inlineBlob(maxBytes):
            "inlineBlob:\(maxBytes)"
        case let .externalBlob(maxBytesPerSource):
            "externalBlob:\(maxBytesPerSource)"
        case .sourceReferenceOnly:
            "sourceReferenceOnly"
        case .representationOnly:
            "representationOnly"
        case .redactedRepresentation:
            "redactedRepresentation"
        }
    }
}
