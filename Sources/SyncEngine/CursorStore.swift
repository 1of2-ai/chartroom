import ConnectorEngine
import Foundation

/// Durable storage for connector sync cursors, keyed by source. The orchestrator reads the
/// cursor before requesting changes and writes it back only once every emitted mutation has
/// been durably accepted by the engine.
public protocol CursorStore: Sendable {
    func cursor(forKey key: String) -> SourceCursor?
    func setCursor(_ cursor: SourceCursor, forKey key: String)
}

/// Persists each cursor under an independent encoded defaults key, so updates for different
/// sources are single writes rather than a shared dictionary read-modify-write. Legacy
/// dictionary entries remain readable for stores created by earlier releases.
public final class UserDefaultsCursorStore: CursorStore, @unchecked Sendable {
    private let defaults: UserDefaults
    private let storageKey: String

    public init(defaults: UserDefaults = .standard, storageKey: String) {
        self.defaults = defaults
        self.storageKey = storageKey
    }

    public func cursor(forKey key: String) -> SourceCursor? {
        if let cursor = defaults.string(forKey: entryStorageKey(for: key)) {
            return cursor
        }
        return (defaults.dictionary(forKey: storageKey) as? [String: SourceCursor])?[key]
    }

    public func setCursor(_ cursor: SourceCursor, forKey key: String) {
        defaults.set(cursor, forKey: entryStorageKey(for: key))
    }

    private func entryStorageKey(for key: String) -> String {
        let encodedKey = Data(key.utf8).base64EncodedString()
        return "\(storageKey).cursor.v1.\(encodedKey)"
    }
}
