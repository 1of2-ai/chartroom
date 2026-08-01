import Foundation
import SQLite3
import Testing
@testable import IndexEngine

@Suite("SQLite storage reliability")
struct StorageReliabilityTests {
    private enum ConfigurationFailure: Error {
        case injected
    }

    private func temporaryStorePath(_ label: String) -> String {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("\(label)-\(UUID().uuidString).sqlite")
            .path
    }

    @Test("a failed post-open configuration closes the SQLite handle")
    func failedConfigurationClosesHandle() throws {
        let path = temporaryStorePath("failed-open")
        defer { try? FileManager.default.removeItem(atPath: path) }

        #expect(throws: ConfigurationFailure.self) {
            _ = try SQLite(path: path) { handle in
                let beginResult = sqlite3_exec(handle, "BEGIN EXCLUSIVE", nil, nil, nil)
                #expect(beginResult == SQLITE_OK)
                throw ConfigurationFailure.injected
            }
        }

        var secondHandle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        #expect(sqlite3_open_v2(path, &secondHandle, flags, nil) == SQLITE_OK)
        let openedHandle = try #require(secondHandle)
        defer { sqlite3_close_v2(openedHandle) }
        sqlite3_busy_timeout(openedHandle, 50)

        #expect(sqlite3_exec(openedHandle, "BEGIN EXCLUSIVE", nil, nil, nil) == SQLITE_OK)
        #expect(sqlite3_exec(openedHandle, "ROLLBACK", nil, nil, nil) == SQLITE_OK)
    }

    @Test("selective failure deletion rolls back as one durable mutation")
    func selectiveFailureDeletionIsAtomic() async throws {
        let path = temporaryStorePath("failure-delete")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 4))

        for index in 0..<3 {
            try await store.recordFailure(.init(
                id: EngineID(rawValue: "failure-\(index)"),
                category: .storageFailure,
                message: "Failure \(index)",
                detail: "fixture",
                recoverability: .retryable,
                occurredAt: Date(timeIntervalSince1970: Double(index))
            ))
        }

        let db = try SQLite(path: path)
        try db.exec("""
        CREATE TABLE failure_delete_attempts (count INTEGER NOT NULL);
        INSERT INTO failure_delete_attempts VALUES(0);
        CREATE TRIGGER fail_second_failure_delete
        BEFORE DELETE ON failures
        BEGIN
          UPDATE failure_delete_attempts SET count = count + 1;
          SELECT CASE
            WHEN (SELECT count FROM failure_delete_attempts) > 1
            THEN RAISE(ABORT, 'injected second-delete failure')
          END;
        END;
        """)

        await #expect(throws: SQLiteError.self) {
            try await store.deleteFailures(ids: ["failure-0", "failure-1", "failure-2"])
        }
        #expect(try await store.failureSnapshots(limit: 10).count == 3)
    }

    @Test("an empty selective failure deletion does not acquire a write lock")
    func emptyFailureDeletionIsANoOp() async throws {
        let path = temporaryStorePath("empty-failure-delete")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let store = try IndexStore(path: path, embedder: HashingEmbedder(dimension: 4))
        let competingWriter = try SQLite(path: path)
        try competingWriter.exec("BEGIN IMMEDIATE")
        defer { try? competingWriter.exec("ROLLBACK") }

        try await store.deleteFailures(ids: [])
    }
}
