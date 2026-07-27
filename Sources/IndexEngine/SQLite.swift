import Accelerate
import Foundation
import SQLite3

/// SQLite wants the TRANSIENT destructor (it copies the bound bytes during the
/// call). It is not exported to Swift, so reconstruct it.
@inline(__always) private func sqliteTransient() -> sqlite3_destructor_type {
    unsafeBitCast(-1, to: sqlite3_destructor_type.self)
}

public struct SQLiteError: Error, CustomStringConvertible {
    public let code: Int32
    public let message: String
    public var description: String { "SQLite error \(code): \(message)" }
}

/// A thin, focused wrapper over the system SQLite C API. Used only inside the
/// `IndexStore` actor, so it does not need to be Sendable.
final class SQLite {
    let handle: OpaquePointer
    let path: String

    init(path: String) throws {
        self.path = path
        var h: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let rc = sqlite3_open_v2(path, &h, flags, nil)
        guard rc == SQLITE_OK, let h else {
            let msg = h.map { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
            if let h { sqlite3_close_v2(h) }
            throw SQLiteError(code: rc, message: msg)
        }
        handle = h
        let timeoutRC = sqlite3_busy_timeout(handle, 5_000)
        guard timeoutRC == SQLITE_OK else {
            throw SQLiteError(code: timeoutRC, message: String(cString: sqlite3_errmsg(handle)))
        }
        try exec("PRAGMA journal_mode=WAL")
        try exec("PRAGMA foreign_keys=ON")
    }

    deinit { sqlite3_close_v2(handle) }

    /// Total on-disk footprint of the database and its WAL/SHM sidecars in bytes, or
    /// nil for an in-memory store. WAL mode (enabled above) keeps `-wal` and `-shm`
    /// companion files next to the main database; that layout is this wrapper's concern,
    /// so callers ask the store for its size rather than knowing the file naming.
    var fileByteSize: Int64? {
        guard path != ":memory:", !path.isEmpty else { return nil }
        let total = [path, path + "-wal", path + "-shm"].reduce(Int64(0)) { sum, candidate in
            let size = (try? FileManager.default.attributesOfItem(atPath: candidate))?[.size] as? Int64 ?? 0
            return sum + size
        }
        return total > 0 ? total : nil
    }

    func exec(_ sql: String) throws {
        var err: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(handle, sql, nil, nil, &err)
        if rc != SQLITE_OK {
            let m = err.map { String(cString: $0) } ?? String(cString: sqlite3_errmsg(handle))
            sqlite3_free(err)
            throw SQLiteError(code: rc, message: m)
        }
    }

    func transaction<T>(_ body: () throws -> T) throws -> T {
        try exec("BEGIN IMMEDIATE TRANSACTION")
        do {
            let result = try body()
            try exec("COMMIT")
            return result
        } catch {
            try? exec("ROLLBACK")
            throw error
        }
    }

    func prepare(_ sql: String) throws -> Statement {
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(handle, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK, let stmt else {
            throw SQLiteError(code: rc, message: String(cString: sqlite3_errmsg(handle)))
        }
        return Statement(stmt: stmt, db: handle)
    }
}

final class Statement {
    private let stmt: OpaquePointer
    private let db: OpaquePointer
    /// The first failed bind since the last `reset()`, rethrown by `step()`.
    ///
    /// Binding is a `void` operation at every call site in this package, so making it throw
    /// would put `try` on several hundred lines to report a fault that only `step()` can act
    /// on anyway. Deferring instead keeps the call sites clean and still fails loudly: a
    /// statement that could not be bound never executes. Aborting the host process — which is
    /// what a `precondition` here did — is not an option a library gets to take.
    private var bindFailure: SQLiteError?

    init(stmt: OpaquePointer, db: OpaquePointer) { self.stmt = stmt; self.db = db }
    deinit { sqlite3_finalize(stmt) }

    func bind(_ i: Int32, _ text: String) { checkBind(sqlite3_bind_text(stmt, i, text, -1, sqliteTransient())) }
    func bind(_ i: Int32, _ value: Double) { checkBind(sqlite3_bind_double(stmt, i, value)) }
    func bind(_ i: Int32, _ value: Int) { checkBind(sqlite3_bind_int64(stmt, i, Int64(value))) }
    func bindNull(_ i: Int32) { checkBind(sqlite3_bind_null(stmt, i)) }
    func bindBlob(_ i: Int32, _ bytes: [UInt8]) {
        bytes.withUnsafeBytes { checkBind(sqlite3_bind_blob(stmt, i, $0.baseAddress, Int32($0.count), sqliteTransient())) }
    }

    /// Records the first failure only: later binds on an already-failed statement report the
    /// same fault, and the first one is the one that explains it.
    private func checkBind(_ rc: Int32) {
        guard rc != SQLITE_OK, bindFailure == nil else { return }
        bindFailure = SQLiteError(code: rc, message: String(cString: sqlite3_errmsg(db)))
    }

    /// Rewind for reuse: clears the row cursor and bindings so one prepared
    /// statement can run repeatedly inside a loop instead of re-preparing.
    func reset() {
        sqlite3_reset(stmt)
        sqlite3_clear_bindings(stmt)
        bindFailure = nil
    }

    /// True if a row is available, false when the statement is done.
    @discardableResult func step() throws -> Bool {
        if let bindFailure {
            self.bindFailure = nil
            throw bindFailure
        }
        let rc = sqlite3_step(stmt)
        if rc == SQLITE_ROW { return true }
        if rc == SQLITE_DONE { return false }
        throw SQLiteError(code: rc, message: String(cString: sqlite3_errmsg(db)))
    }

    func text(_ col: Int32) -> String? {
        guard let c = sqlite3_column_text(stmt, col) else { return nil }
        return String(cString: c)
    }
    func int(_ col: Int32) -> Int { Int(sqlite3_column_int64(stmt, col)) }
    func double(_ col: Int32) -> Double { sqlite3_column_double(stmt, col) }
    func null(_ col: Int32) -> Bool { sqlite3_column_type(stmt, col) == SQLITE_NULL }
    /// For nullable integer columns, where `int(_:)` would report a missing value as 0 and make
    /// "not recorded" indistinguishable from a real zero.
    func optionalInt(_ col: Int32) -> Int? { null(col) ? nil : int(col) }
    func blob(_ col: Int32) -> [UInt8] {
        guard let p = sqlite3_column_blob(stmt, col) else { return [] }
        let n = Int(sqlite3_column_bytes(stmt, col))
        return Array(UnsafeRawBufferPointer(start: p, count: n))
    }

    /// Reads a blob column without copying it. SQLite owns the bytes and invalidates them at the
    /// next `step()` or `reset()`, so `body` runs inside that window and must not escape the
    /// buffer. Used by the vector scan, where copying every stored embedding into an array was
    /// the scan's dominant cost.
    func withBlob<T>(_ col: Int32, _ body: (UnsafeRawBufferPointer) -> T) -> T {
        guard let pointer = sqlite3_column_blob(stmt, col) else {
            return body(UnsafeRawBufferPointer(start: nil, count: 0))
        }
        return body(UnsafeRawBufferPointer(start: pointer, count: Int(sqlite3_column_bytes(stmt, col))))
    }
}

/// Float vector <-> bytes (host-endian; the DB is local to one machine) and the
/// similarity used for ranking. Embeddings are L2-normalized, so cosine = dot.
enum Vector {
    static func toBytes(_ v: [Float]) -> [UInt8] { v.withUnsafeBytes { Array($0) } }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        let count = min(a.count, b.count)
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(count))
        return result
    }

    /// Cosine of `query` against a stored vector still sitting in SQLite's own buffer.
    ///
    /// Returns nil when the blob is not exactly `query.count` floats, which is the stored-vector
    /// dimension mismatch the caller reports rather than scoring past.
    ///
    /// SQLite gives no alignment guarantee for blob storage, and binding misaligned memory to
    /// `Float` is undefined behaviour, so the aligned case scores in place and the misaligned one
    /// pays for a single stack copy. Both avoid the two heap allocations per row — a `[UInt8]`
    /// and a `[Float]` — that the scan used to make for every embedding in the store.
    static func cosine(query: [Float], storedBytes: UnsafeRawBufferPointer) -> Float? {
        let dimension = query.count
        guard storedBytes.count == dimension * MemoryLayout<Float>.stride,
              let base = storedBytes.baseAddress
        else { return nil }

        if Int(bitPattern: base).isMultiple(of: MemoryLayout<Float>.alignment) {
            var result: Float = 0
            vDSP_dotpr(query, 1, base.assumingMemoryBound(to: Float.self), 1, &result, vDSP_Length(dimension))
            return result
        }

        return withUnsafeTemporaryAllocation(of: Float.self, capacity: dimension) { aligned in
            guard let alignedBase = aligned.baseAddress else { return 0 }
            memcpy(alignedBase, base, storedBytes.count)
            var result: Float = 0
            vDSP_dotpr(query, 1, alignedBase, 1, &result, vDSP_Length(dimension))
            return result
        }
    }
}

/// The best `capacity` vector hits seen so far, as a min-heap on similarity.
///
/// Exact cosine search has no way to skip a stored vector — the top-k is only knowable after
/// every candidate is scored — so the scan itself cannot be bounded. What *can* be bounded is
/// what it retains: this keeps `k` entries instead of one per corpus chunk, and because a row's
/// id is only read once the row is admitted, a large store no longer allocates a Swift `String`
/// for every chunk it rejects.
struct BoundedTopKHits {
    private let capacity: Int
    private var entries: [(similarity: Float, id: String)] = []

    init(capacity: Int) {
        self.capacity = max(0, capacity)
        entries.reserveCapacity(self.capacity)
    }

    /// `id` is an autoclosure so a rejected row never materializes one. On a large store almost
    /// every row is rejected, which is the whole point.
    mutating func insert(similarity: Float, id: @autoclosure () -> String?) {
        guard capacity > 0 else { return }
        if entries.count == capacity {
            guard similarity > entries[0].similarity, let id = id() else { return }
            entries[0] = (similarity, id)
            siftDown(from: 0)
        } else {
            guard let id = id() else { return }
            entries.append((similarity, id))
            siftUp(from: entries.count - 1)
        }
    }

    /// Best first, ties broken by id so repeated identical queries return a stable order.
    func sortedDescending() -> [(id: String, similarity: Float)] {
        entries
            .sorted { lhs, rhs in
                lhs.similarity == rhs.similarity ? lhs.id < rhs.id : lhs.similarity > rhs.similarity
            }
            .map { (id: $0.id, similarity: $0.similarity) }
    }

    private mutating func siftUp(from index: Int) {
        var child = index
        while child > 0 {
            let parent = (child - 1) / 2
            guard entries[child].similarity < entries[parent].similarity else { return }
            entries.swapAt(child, parent)
            child = parent
        }
    }

    private mutating func siftDown(from index: Int) {
        var parent = index
        while true {
            let left = parent * 2 + 1
            let right = left + 1
            var smallest = parent
            if left < entries.count, entries[left].similarity < entries[smallest].similarity {
                smallest = left
            }
            if right < entries.count, entries[right].similarity < entries[smallest].similarity {
                smallest = right
            }
            guard smallest != parent else { return }
            entries.swapAt(parent, smallest)
            parent = smallest
        }
    }
}
