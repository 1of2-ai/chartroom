import Foundation

@discardableResult
private func ensureIndexStoreColumn(db: SQLite, table: String, name: String, definition: String) throws -> Bool {
    let statement = try db.prepare("PRAGMA table_info(\(table))")
    var names = Set<String>()
    while try statement.step() {
        if let columnName = statement.text(1) {
            names.insert(columnName)
        }
    }

    if !names.contains(name) {
        try db.exec("ALTER TABLE \(table) ADD COLUMN \(definition)")
        return true
    }
    return false
}

/// Raised when a store cannot be brought to the schema this build understands.
public enum IndexStoreSchemaError: Error, CustomStringConvertible {
    case storeFromNewerBuild(storeVersion: Int, supportedVersion: Int)
    case referentialIntegrityCheckFailed(detail: String)

    public var description: String {
        switch self {
        case let .storeFromNewerBuild(storeVersion, supportedVersion):
            """
            The index store was written by a newer build (schema v\(storeVersion)); \
            this build supports up to v\(supportedVersion).
            """
        case let .referentialIntegrityCheckFailed(detail):
            "The index store still violates referential integrity after migration: \(detail)"
        }
    }
}

extension IndexStore {
    /// The newest schema version this build knows how to read and write.
    ///
    /// v1 — durable retrieval core.
    /// v2 — foreign keys across the document → chunk → embedding → vector chain.
    static let supportedSchemaVersion = 2

    static func record(migration version: Int, name: String, db: SQLite) throws {
        let statement = try db.prepare("""
        INSERT OR IGNORE INTO schema_migrations(version,name,applied_at) VALUES(?1,?2,?3)
        """)
        statement.bind(1, version)
        statement.bind(2, name)
        statement.bind(3, Date.now.timeIntervalSince1970)
        try statement.step()
    }

    static func appliedSchemaVersion(db: SQLite) throws -> Int {
        let statement = try db.prepare("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
        guard try statement.step() else { return 0 }
        return statement.int(0)
    }

    /// A store from a newer build may contain columns and constraints this one will silently
    /// ignore or violate. Opening it read-write is how a downgrade corrupts data, so refuse
    /// instead — the one thing an unversioned `CREATE TABLE IF NOT EXISTS` schema could never do.
    private static func refuseStoreFromNewerBuild(db: SQLite) throws {
        let version = try appliedSchemaVersion(db: db)
        guard version <= supportedSchemaVersion else {
            throw IndexStoreSchemaError.storeFromNewerBuild(
                storeVersion: version,
                supportedVersion: supportedSchemaVersion
            )
        }
    }

    /// Rebuilds the pre-v2 tables so their foreign keys are declared and enforced.
    ///
    /// `PRAGMA foreign_keys=ON` was set from the beginning while no table declared a single
    /// `REFERENCES` clause, so the store advertised integrity it never had: every orphan bug —
    /// embeddings outliving their chunk, vectors outliving their embedding — had to be caught by
    /// hand in application code, and one of them shipped.
    ///
    /// SQLite cannot add a constraint with `ALTER TABLE`, so the affected tables are rebuilt.
    /// Orphans accumulated while nothing enforced the references are deleted first, otherwise the
    /// rebuild would fail on data the old schema permitted. New stores are already created with
    /// the constraints above and skip straight past this.
    private static func migrateToReferentialIntegrity(db: SQLite) throws {
        guard try appliedSchemaVersion(db: db) < 2 else { return }

        // A store just created by `installSchema` already has the constraints, so there is nothing
        // to rebuild — only the version to record. Asking the schema itself is what keeps this
        // honest: a store is v2 when its tables say so, not when a counter says so.
        guard try !isAlreadyConstrained(db: db) else {
            try record(migration: 2, name: "referential-integrity", db: db)
            return
        }

        // Foreign keys must be off for the drop/rename dance, and the pragma is a no-op inside a
        // transaction — so it is set outside, exactly as SQLite's own migration procedure requires.
        try db.exec("PRAGMA foreign_keys=OFF")
        // Modern `ALTER TABLE ... RENAME` also rewrites references to the table across the rest of
        // the schema, which fails mid-rebuild while a referenced table is still dropped. Legacy
        // rename is the documented way through a multi-table rebuild.
        try db.exec("PRAGMA legacy_alter_table=ON")
        defer {
            try? db.exec("PRAGMA legacy_alter_table=OFF")
            try? db.exec("PRAGMA foreign_keys=ON")
        }

        do {
            try db.transaction {
                // Orphans first: these are rows the old schema allowed and the new one forbids.
                try db.exec("""
                DELETE FROM vectors WHERE id NOT IN (SELECT id FROM embeddings);
                DELETE FROM embeddings WHERE chunk_id NOT IN (SELECT id FROM chunks);
                DELETE FROM chunks WHERE document_id NOT IN (SELECT id FROM documents);
                DELETE FROM representations WHERE document_id NOT IN (SELECT id FROM documents);
                DELETE FROM representation_lineages WHERE document_id NOT IN (SELECT id FROM documents);
                DELETE FROM document_versions WHERE document_id NOT IN (SELECT id FROM documents);
                """)

                // Children before parents, so each rebuilt table's references already resolve.
                for table in constrainedTables.reversed() {
                    try rebuildWithConstraints(table: table, db: db)
                }

                // The rebuild dropped the originals, and their indexes went with them.
                try db.exec("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(active);
                CREATE INDEX IF NOT EXISTS idx_embeddings_space ON embeddings(embedding_space_id);
                CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings(chunk_id);
                """)

                try record(migration: 2, name: "referential-integrity", db: db)
            }
        }

        // Ask SQLite itself rather than trusting the rebuild: a surviving violation means the
        // store is not what the schema now claims, and the caller must hear that at open.
        let check = try db.prepare("PRAGMA foreign_key_check")
        if try check.step() {
            let table = check.text(0) ?? "unknown"
            let parent = check.text(2) ?? "unknown"
            throw IndexStoreSchemaError.referentialIntegrityCheckFailed(
                detail: "\(table) references a missing row in \(parent)"
            )
        }
    }

    /// Copies one table into a freshly created twin that carries the constraints, then swaps it in.
    ///
    /// The twin is created by running the current schema against a temporary name, so there is one
    /// definition of each table rather than a second copy here that could drift from it.
    private static func rebuildWithConstraints(table: String, db: SQLite) throws {
        let columns = try columnNames(of: table, db: db)
        guard !columns.isEmpty else { return }

        let migratedName = "\(table)_migrated_v2"
        try db.exec("DROP TABLE IF EXISTS \(migratedName)")
        try db.exec(Self.constrainedDefinition(for: table, named: migratedName))

        // Only columns present in both survive; the constrained definitions add no columns the old
        // tables lacked, so this is the full set in practice and states the assumption explicitly.
        let shared = try columns.filter(columnNames(of: migratedName, db: db).contains)
        let columnList = shared.joined(separator: ",")
        try db.exec("INSERT INTO \(migratedName)(\(columnList)) SELECT \(columnList) FROM \(table)")
        try db.exec("DROP TABLE \(table)")
        try db.exec("ALTER TABLE \(migratedName) RENAME TO \(table)")
    }

    /// True when every constrained table already declares its foreign keys — the shape a store
    /// created by the current `installSchema` has from the start.
    private static func isAlreadyConstrained(db: SQLite) throws -> Bool {
        let statement = try db.prepare("SELECT sql FROM sqlite_master WHERE type='table' AND name=?1")
        for table in constrainedTables {
            statement.reset()
            statement.bind(1, table)
            guard try statement.step(), let sql = statement.text(0), sql.contains("REFERENCES") else {
                return false
            }
        }
        return true
    }

    private static func columnNames(of table: String, db: SQLite) throws -> [String] {
        let statement = try db.prepare("PRAGMA table_info(\(table))")
        var names: [String] = []
        while try statement.step() {
            if let name = statement.text(1) {
                names.append(name)
            }
        }
        return names
    }


    /// Column bodies for every table that carries a foreign key.
    ///
    /// One definition each, because `installSchema` creates these for new stores and the v2
    /// rebuild recreates them for existing ones. A second copy of this DDL is exactly how a
    /// migrated store and a fresh store end up with different constraints.
    private static let constrainedTableBodies: [String: String] = [
        "document_versions": """
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          content_hash TEXT NOT NULL,
          policy_id TEXT NOT NULL,
          policy_version INTEGER NOT NULL,
          created_at REAL NOT NULL,
          retained_state TEXT NOT NULL
        """,
        "representation_lineages": """
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          policy_id TEXT NOT NULL,
          representation_kind TEXT NOT NULL,
          active_representation_id TEXT NOT NULL,
          updated_at REAL NOT NULL
        """,
        "representations": """
          id TEXT PRIMARY KEY,
          lineage_id TEXT NOT NULL,
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          document_version_id TEXT NOT NULL,
          representation_kind TEXT NOT NULL,
          extractor_id TEXT NOT NULL,
          extractor_version TEXT NOT NULL,
          policy_id TEXT NOT NULL,
          policy_version INTEGER NOT NULL,
          language TEXT NOT NULL DEFAULT '',
          text TEXT NOT NULL,
          token_count INTEGER NOT NULL,
          content_hash TEXT NOT NULL,
          source_material_availability TEXT NOT NULL,
          upstream_dependency_state TEXT NOT NULL,
          active INTEGER NOT NULL,
          created_at REAL NOT NULL
        """,
        "chunks": """
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          document_version_id TEXT NOT NULL,
          representation_id TEXT NOT NULL,
          representation_lineage_id TEXT NOT NULL,
          ordinal INTEGER NOT NULL,
          chunker_id TEXT NOT NULL,
          chunker_version TEXT NOT NULL,
          policy_id TEXT NOT NULL,
          policy_version INTEGER NOT NULL,
          text TEXT NOT NULL,
          context_prefix TEXT NOT NULL,
          context_suffix TEXT NOT NULL,
          heading_path TEXT NOT NULL,
          byte_start INTEGER NOT NULL,
          byte_end INTEGER NOT NULL,
          character_start INTEGER NOT NULL,
          character_end INTEGER NOT NULL,
          token_start INTEGER NOT NULL,
          token_end INTEGER NOT NULL,
          -- 1-based, inclusive, and nullable on purpose: chunks written before line tracking
          -- have *unknown* lines, which is not the same claim as line 0.
          line_start INTEGER,
          line_end INTEGER,
          page_start INTEGER NOT NULL,
          page_end INTEGER NOT NULL,
          section_label TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          active INTEGER NOT NULL,
          availability_state TEXT NOT NULL,
          created_at REAL NOT NULL
        """,
        "embeddings": """
          id TEXT PRIMARY KEY,
          chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
          embedding_space_id TEXT NOT NULL,
          model_id TEXT NOT NULL,
          model_version TEXT NOT NULL,
          dimension INTEGER NOT NULL,
          modality TEXT NOT NULL,
          prompt_kind TEXT NOT NULL,
          vector_backend_id TEXT NOT NULL,
          vector_backend_version TEXT NOT NULL,
          vector_hash TEXT NOT NULL,
          created_at REAL NOT NULL
        """,
        "vectors": """
          id TEXT PRIMARY KEY REFERENCES embeddings(id) ON DELETE CASCADE,
          dim INTEGER NOT NULL,
          vec BLOB NOT NULL
        """,
    ]

    /// Dependency order: a table is rebuilt only after the tables it references.
    static let constrainedTables = [
        "document_versions", "representation_lineages", "representations", "chunks", "embeddings", "vectors",
    ]

    static func constrainedDefinition(for table: String, named name: String) -> String {
        guard let body = constrainedTableBodies[table] else { return "" }
        return "CREATE TABLE \(name) (\n\(body)\n)"
    }

    private static func createConstrainedTables(db: SQLite) throws {
        for table in constrainedTables {
            guard let body = constrainedTableBodies[table] else { continue }
            try db.exec("CREATE TABLE IF NOT EXISTS \(table) (\n\(body)\n)")
        }
    }

    static func installSchema(
        db: SQLite,
        vectorBackendID: String,
        vectorBackendVersion: String
    ) throws {
        try db.exec("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version INTEGER PRIMARY KEY,
          name TEXT NOT NULL,
          applied_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sources (
          id TEXT PRIMARY KEY,
          connector_kind TEXT NOT NULL,
          connector_instance_id TEXT NOT NULL,
          source_uri TEXT NOT NULL,
          external_stable_id TEXT NOT NULL,
          sync_cursor TEXT NOT NULL DEFAULT '',
          capability_snapshot TEXT NOT NULL DEFAULT '{}',
          permission_scope_hash TEXT NOT NULL DEFAULT '',
          auth_reference_id TEXT NOT NULL DEFAULT '',
          updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY,
          source_id TEXT NOT NULL,
          source_uri TEXT NOT NULL,
          external_id TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          version INTEGER NOT NULL,
          active_version_id TEXT NOT NULL,
          title TEXT NOT NULL,
          content_type TEXT NOT NULL,
          file_extension TEXT NOT NULL,
          size INTEGER NOT NULL,
          created_at REAL NOT NULL,
          modified_at REAL NOT NULL,
          ingested_at REAL NOT NULL,
          updated_at REAL NOT NULL,
          is_deleted INTEGER NOT NULL DEFAULT 0,
          permission_scope_id TEXT NOT NULL DEFAULT '',
          provenance TEXT NOT NULL DEFAULT '{}',
          cluster_id TEXT NOT NULL DEFAULT ''
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
          text,
          title,
          path,
          heading_path,
          chunk_id UNINDEXED,
          tokenize='porter unicode61'
        );

        CREATE TABLE IF NOT EXISTS vector_backend_metadata (
          id TEXT PRIMARY KEY,
          version TEXT NOT NULL,
          state TEXT NOT NULL,
          message TEXT NOT NULL,
          updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS policies (
          id TEXT PRIMARY KEY,
          version INTEGER NOT NULL,
          raw_retention TEXT NOT NULL,
          extractor_id TEXT NOT NULL,
          chunker_id TEXT NOT NULL,
          embedding_provider_id TEXT NOT NULL,
          vector_backend_id TEXT NOT NULL,
          updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS policy_component_resolutions (
          policy_id TEXT NOT NULL,
          component_id TEXT NOT NULL,
          component_kind TEXT NOT NULL,
          state TEXT NOT NULL,
          message TEXT NOT NULL,
          PRIMARY KEY(policy_id, component_id, component_kind)
        );

        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY,
          state TEXT NOT NULL,
          kind TEXT NOT NULL DEFAULT 'ingest',
          completed_unit_count INTEGER NOT NULL,
          total_unit_count INTEGER,
          message TEXT NOT NULL,
          updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS failures (
          id TEXT PRIMARY KEY,
          category TEXT NOT NULL,
          message TEXT NOT NULL,
          detail TEXT NOT NULL,
          source_id TEXT,
          document_id TEXT,
          source_uri TEXT,
          is_recoverable INTEGER NOT NULL,
          recoverability TEXT NOT NULL DEFAULT 'retryable',
          occurred_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS objects (
          id TEXT PRIMARY KEY,
          type TEXT NOT NULL,
          title TEXT NOT NULL,
          cluster_id TEXT,
          source_id TEXT,
          source_uri TEXT,
          policy_id TEXT,
          representation_id TEXT,
          embedding_space_id TEXT,
          model_id TEXT NOT NULL,
          updated_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
        CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(content_type);
        CREATE INDEX IF NOT EXISTS idx_documents_cluster ON documents(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_failures_time ON failures(occurred_at);
        CREATE INDEX IF NOT EXISTS idx_jobs_time ON jobs(updated_at);
        """)
        try createConstrainedTables(db: db)
        try db.exec("""
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(active);
        CREATE INDEX IF NOT EXISTS idx_embeddings_space ON embeddings(embedding_space_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings(chunk_id);
        """)
        try ensureIndexStoreColumn(db: db, table: "objects", name: "source_id", definition: "source_id TEXT")
        try ensureIndexStoreColumn(db: db, table: "objects", name: "source_uri", definition: "source_uri TEXT")
        try ensureIndexStoreColumn(db: db, table: "objects", name: "policy_id", definition: "policy_id TEXT")
        try ensureIndexStoreColumn(db: db, table: "objects", name: "representation_id", definition: "representation_id TEXT")
        // Tables that never earned readers (#66): declared in early schemas, written
        // never (or only with placeholder values). Dropped from existing stores too.
        try db.exec("""
        DROP TABLE IF EXISTS source_cursors;
        DROP TABLE IF EXISTS raw_blobs;
        DROP TABLE IF EXISTS relations;
        DROP TABLE IF EXISTS chunk_metadata;
        -- Same category, found by a later sweep: created by every prior schema, never once
        -- written to or read from.
        DROP TABLE IF EXISTS search_diagnostics;
        DROP TABLE IF EXISTS objects_fts;
        """)
        try ensureIndexStoreColumn(db: db, table: "objects", name: "embedding_space_id", definition: "embedding_space_id TEXT")
        // Left NULL for existing rows: a re-ingest fills them, and until then the read path
        // reports "unknown" rather than inventing a line range.
        try ensureIndexStoreColumn(db: db, table: "chunks", name: "line_start", definition: "line_start INTEGER")
        try ensureIndexStoreColumn(db: db, table: "chunks", name: "line_end", definition: "line_end INTEGER")
        try ensureIndexStoreColumn(db: db, table: "jobs", name: "kind", definition: "kind TEXT NOT NULL DEFAULT 'ingest'")
        try ensureIndexStoreColumn(db: db, table: "failures", name: "source_uri", definition: "source_uri TEXT")
        let addedRecoverabilityColumn = try ensureIndexStoreColumn(
            db: db,
            table: "failures",
            name: "recoverability",
            definition: "recoverability TEXT NOT NULL DEFAULT 'retryable'"
        )
        if addedRecoverabilityColumn {
            try db.exec("""
            UPDATE failures
            SET recoverability = CASE WHEN is_recoverable != 0 THEN 'retryable' ELSE 'unrecoverable' END;
            """)
        } else {
            try db.exec("""
            UPDATE failures
            SET recoverability = CASE WHEN is_recoverable != 0 THEN 'retryable' ELSE 'unrecoverable' END
            WHERE recoverability IS NULL OR recoverability = '';
            """)
        }
        try db.exec("""
        CREATE INDEX IF NOT EXISTS idx_objects_cluster ON objects(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_objects_source ON objects(source_id);
        CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(type);
        CREATE INDEX IF NOT EXISTS idx_objects_policy ON objects(policy_id);
        CREATE INDEX IF NOT EXISTS idx_objects_embedding_space ON objects(embedding_space_id);
        """)

        try record(migration: 1, name: "durable-retrieval-core", db: db)
        try migrateToReferentialIntegrity(db: db)
        try refuseStoreFromNewerBuild(db: db)

        let backend = try db.prepare("""
        INSERT INTO vector_backend_metadata(id,version,state,message,updated_at)
        VALUES(?1,?2,?3,?4,?5)
        ON CONFLICT(id) DO UPDATE SET version=excluded.version,
          state=excluded.state,
          message=excluded.message,
          updated_at=excluded.updated_at
        """)
        backend.bind(1, vectorBackendID)
        backend.bind(2, vectorBackendVersion)
        backend.bind(3, VectorStorageStatus.State.ready.rawValue)
        backend.bind(4, "Exact vector scan over SQLite-backed chunk embeddings")
        backend.bind(5, Date.now.timeIntervalSince1970)
        try backend.step()
    }
}
