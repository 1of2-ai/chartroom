import Foundation
import Testing
@testable import IndexEngine

/// The v2 migration is the one change in this package that rewrites a user's existing store, so it
/// is exercised against a store built the old way rather than only against fresh ones — a fresh
/// store takes the short path and would prove nothing about the rebuild.
@Suite("Schema migration to referential integrity")
struct SchemaMigrationTests {
    private func temporaryStorePath(_ label: String) -> String {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("\(label)-\(UUID().uuidString).sqlite")
            .path
    }

    /// Builds the pre-v2 shape: the same tables with no `REFERENCES` clauses anywhere.
    private func makeLegacyStore(at path: String) throws {
        let db = try SQLite(path: path)
        try db.exec("""
        CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at REAL NOT NULL);
        CREATE TABLE documents (
          id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_uri TEXT NOT NULL, external_id TEXT NOT NULL,
          content_hash TEXT NOT NULL, version INTEGER NOT NULL, active_version_id TEXT NOT NULL,
          title TEXT NOT NULL, content_type TEXT NOT NULL, file_extension TEXT NOT NULL, size INTEGER NOT NULL,
          created_at REAL NOT NULL, modified_at REAL NOT NULL, ingested_at REAL NOT NULL, updated_at REAL NOT NULL,
          is_deleted INTEGER NOT NULL DEFAULT 0, permission_scope_id TEXT NOT NULL DEFAULT '',
          provenance TEXT NOT NULL DEFAULT '{}', cluster_id TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE chunks (
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL, document_version_id TEXT NOT NULL,
          representation_id TEXT NOT NULL, representation_lineage_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
          chunker_id TEXT NOT NULL, chunker_version TEXT NOT NULL, policy_id TEXT NOT NULL,
          policy_version INTEGER NOT NULL, text TEXT NOT NULL, context_prefix TEXT NOT NULL,
          context_suffix TEXT NOT NULL, heading_path TEXT NOT NULL, byte_start INTEGER NOT NULL,
          byte_end INTEGER NOT NULL, character_start INTEGER NOT NULL, character_end INTEGER NOT NULL,
          token_start INTEGER NOT NULL, token_end INTEGER NOT NULL, line_start INTEGER, line_end INTEGER,
          page_start INTEGER NOT NULL, page_end INTEGER NOT NULL, section_label TEXT NOT NULL,
          content_hash TEXT NOT NULL, active INTEGER NOT NULL, availability_state TEXT NOT NULL,
          created_at REAL NOT NULL
        );
        CREATE TABLE embeddings (
          id TEXT PRIMARY KEY, chunk_id TEXT NOT NULL, embedding_space_id TEXT NOT NULL, model_id TEXT NOT NULL,
          model_version TEXT NOT NULL, dimension INTEGER NOT NULL, modality TEXT NOT NULL,
          prompt_kind TEXT NOT NULL, vector_backend_id TEXT NOT NULL, vector_backend_version TEXT NOT NULL,
          vector_hash TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE vectors (id TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL);
        CREATE TABLE document_versions (
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL, content_hash TEXT NOT NULL, policy_id TEXT NOT NULL,
          policy_version INTEGER NOT NULL, created_at REAL NOT NULL, retained_state TEXT NOT NULL
        );
        CREATE TABLE representation_lineages (
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL, policy_id TEXT NOT NULL,
          representation_kind TEXT NOT NULL, active_representation_id TEXT NOT NULL, updated_at REAL NOT NULL
        );
        CREATE TABLE representations (
          id TEXT PRIMARY KEY, lineage_id TEXT NOT NULL, document_id TEXT NOT NULL,
          document_version_id TEXT NOT NULL, representation_kind TEXT NOT NULL, extractor_id TEXT NOT NULL,
          extractor_version TEXT NOT NULL, policy_id TEXT NOT NULL, policy_version INTEGER NOT NULL,
          language TEXT NOT NULL DEFAULT '', text TEXT NOT NULL, token_count INTEGER NOT NULL,
          content_hash TEXT NOT NULL, source_material_availability TEXT NOT NULL,
          upstream_dependency_state TEXT NOT NULL, active INTEGER NOT NULL, created_at REAL NOT NULL
        );
        INSERT INTO schema_migrations VALUES(1,'durable-retrieval-core',0);

        -- One document with a chunk, its embedding, and its vector: the rows that must survive.
        INSERT INTO documents VALUES('doc-keep','src','file:///keep','ext','hash',1,'v1','Keep','text','txt',
          10,0,0,0,0,0,'','{}','');
        INSERT INTO chunks VALUES('chunk-keep','doc-keep','v1','rep','lin',0,'chunker','1','default',1,
          'kept text','','','',0,9,0,9,0,1,1,1,0,0,'','chash',1,'chunkTextAvailable',0);
        INSERT INTO embeddings VALUES('emb-keep','chunk-keep','space','model','1',4,'text','document',
          'backend','1','vhash',0);
        INSERT INTO vectors VALUES('emb-keep',4,X'00000000000000000000000000000000');

        -- Orphans the unenforced schema permitted, each a row the constraints now forbid.
        INSERT INTO chunks VALUES('chunk-orphan','doc-missing','v1','rep','lin',0,'chunker','1','default',1,
          'orphan','','','',0,6,0,6,0,1,1,1,0,0,'','chash',1,'chunkTextAvailable',0);
        INSERT INTO embeddings VALUES('emb-orphan','chunk-missing','space','model','1',4,'text','document',
          'backend','1','vhash',0);
        INSERT INTO vectors VALUES('vec-orphan',4,X'00000000000000000000000000000000');
        """)
    }

    private func count(_ db: SQLite, _ sql: String) throws -> Int {
        let statement = try db.prepare(sql)
        return try statement.step() ? statement.int(0) : 0
    }

    @Test("a pre-v2 store gains enforced foreign keys and keeps its valid rows")
    func migrationAddsConstraintsAndPreservesData() throws {
        let path = temporaryStorePath("legacy-v1")
        defer { try? FileManager.default.removeItem(atPath: path) }
        try makeLegacyStore(at: path)

        _ = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 4))

        let db = try SQLite(path: path)
        #expect(try IndexStore.appliedSchemaVersion(db: db) == 2)

        // The valid chain survived intact.
        #expect(try count(db, "SELECT COUNT(*) FROM chunks WHERE id='chunk-keep'") == 1)
        #expect(try count(db, "SELECT COUNT(*) FROM embeddings WHERE id='emb-keep'") == 1)
        #expect(try count(db, "SELECT COUNT(*) FROM vectors WHERE id='emb-keep'") == 1)

        // The orphans the old schema allowed are gone.
        #expect(try count(db, "SELECT COUNT(*) FROM chunks WHERE id='chunk-orphan'") == 0)
        #expect(try count(db, "SELECT COUNT(*) FROM embeddings WHERE id='emb-orphan'") == 0)
        #expect(try count(db, "SELECT COUNT(*) FROM vectors WHERE id='vec-orphan'") == 0)

        // Every rebuilt table declares its references, and SQLite agrees the store is consistent.
        for table in IndexStore.constrainedTables {
            let statement = try db.prepare("SELECT sql FROM sqlite_master WHERE type='table' AND name=?1")
            statement.bind(1, table)
            #expect(try statement.step())
            #expect(statement.text(0)?.contains("REFERENCES") == true, "\(table) has no foreign key")
        }
        let check = try db.prepare("PRAGMA foreign_key_check")
        #expect(try check.step() == false)
    }

    /// The constraints have to be live, not merely declared — that was the whole defect.
    @Test("an orphan insert is rejected after migration")
    func constraintsAreEnforcedAfterMigration() throws {
        let path = temporaryStorePath("legacy-enforce")
        defer { try? FileManager.default.removeItem(atPath: path) }
        try makeLegacyStore(at: path)
        _ = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 4))

        let db = try SQLite(path: path)
        #expect(throws: SQLiteError.self) {
            try db.exec("""
            INSERT INTO embeddings VALUES('emb-new','chunk-does-not-exist','space','model','1',4,'text',
              'document','backend','1','vhash',0)
            """)
        }
    }

    /// Deleting a document now takes its chunks, embeddings, and vectors with it at the storage
    /// layer. Application code still removes them explicitly; this is the net beneath that.
    @Test("deleting a document cascades to its chunks, embeddings, and vectors")
    func deleteCascades() throws {
        let path = temporaryStorePath("cascade")
        defer { try? FileManager.default.removeItem(atPath: path) }
        try makeLegacyStore(at: path)
        _ = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 4))

        let db = try SQLite(path: path)
        try db.exec("DELETE FROM documents WHERE id='doc-keep'")
        #expect(try count(db, "SELECT COUNT(*) FROM chunks") == 0)
        #expect(try count(db, "SELECT COUNT(*) FROM embeddings") == 0)
        #expect(try count(db, "SELECT COUNT(*) FROM vectors") == 0)
    }

    @Test("a store written by a newer build is refused rather than opened")
    func newerStoreIsRefused() throws {
        let path = temporaryStorePath("future")
        defer { try? FileManager.default.removeItem(atPath: path) }

        _ = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 4))
        let db = try SQLite(path: path)
        try IndexStore.record(migration: 99, name: "from-the-future", db: db)

        #expect(throws: IndexStoreSchemaError.self) {
            _ = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 4))
        }
    }

    @Test("a fresh store is already at the current version and stays openable")
    func freshStoreIsCurrent() throws {
        let path = temporaryStorePath("fresh")
        defer { try? FileManager.default.removeItem(atPath: path) }

        _ = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 4))
        _ = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 4))

        let db = try SQLite(path: path)
        #expect(try IndexStore.appliedSchemaVersion(db: db) == IndexStore.supportedSchemaVersion)
    }
}
