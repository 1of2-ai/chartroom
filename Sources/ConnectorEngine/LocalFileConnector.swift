import Foundation
import IndexEngine
import UniformTypeIdentifiers

public struct ConnectorEngineError: Error, Sendable, CustomStringConvertible {
    public enum Category: String, Codable, Sendable {
        case configurationInvalid
        case sourceUnavailable
        case permissionDenied
        case unsupportedSource
        case cursorInvalid
    }

    public var category: Category
    public var code: String
    public var summary: String
    public var detail: String

    public init(
        _ category: Category,
        code: String,
        summary: String,
        detail: String = ""
    ) {
        self.category = category
        self.code = code
        self.summary = summary
        self.detail = detail
    }

    public var description: String {
        detail.isEmpty ? "\(code): \(summary)" : "\(code): \(summary). \(detail)"
    }
}

public struct LocalFileConnectorOptions: Codable, Hashable, Sendable {
    public var allowedPathExtensions: Set<String>?
    public var maxFileSizeBytes: Int64
    public var includeHiddenFiles: Bool

    public init(
        allowedPathExtensions: Set<String>? = nil,
        maxFileSizeBytes: Int64 = 20 * 1024 * 1024,
        includeHiddenFiles: Bool = false
    ) {
        self.allowedPathExtensions = allowedPathExtensions.map { extensions in
            Set(extensions.map(Self.normalizedPathExtension))
        }
        self.maxFileSizeBytes = maxFileSizeBytes
        self.includeHiddenFiles = includeHiddenFiles
    }

    private static func normalizedPathExtension(_ pathExtension: String) -> String {
        pathExtension
            .lowercased()
            .trimmingCharacters(in: CharacterSet(charactersIn: "."))
    }
}

public struct LocalFileConnector: SourceConnector {
    fileprivate static let cursorPrefix = "local-file-v2:"
    fileprivate static let legacyCursorPrefix = "local-file-v1:"
    private static let scanQueue = DispatchQueue(
        label: "com.chartroom.local-file-scan",
        qos: .utility,
        attributes: .concurrent
    )

    public let id: ConnectorID
    public let rootURL: URL
    public let options: LocalFileConnectorOptions
    public let capabilities = ConnectorCapabilities(supportsIncrementalSync: true, supportsRuntimeTools: false)

    public init(
        rootURL: URL,
        id: ConnectorID = "local-files",
        options: LocalFileConnectorOptions = .init()
    ) {
        self.rootURL = rootURL.standardizedFileURL
        self.id = id
        self.options = options
    }

    public func validate() async throws {
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: rootURL.path, isDirectory: &isDirectory)
        guard exists else {
            throw ConnectorEngineError(
                .sourceUnavailable,
                code: "connector.local.missing-root",
                summary: "The selected local source does not exist.",
                detail: rootURL.path
            )
        }
    }

    public func changes(since cursor: SourceCursor?) async throws -> AsyncThrowingStream<SourceEvent, Error> {
        // Cheap checks stay eager so callers get typed errors at the call site; the
        // expensive directory walk runs inside the stream, during iteration.
        try await validate()
        let previousCursor = try decodedCursor(cursor)

        return AsyncThrowingStream { continuation in
            let cancellation = LocalFileScanCancellation()
            Self.scanQueue.async {
                do {
                    try emitChanges(
                        previousCursor: previousCursor,
                        checkCancellation: cancellation.check,
                        into: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in cancellation.cancel() }
        }
    }

    private func emitChanges(
        previousCursor: LocalFileCursor?,
        checkCancellation: () throws -> Void,
        into continuation: AsyncThrowingStream<SourceEvent, Error>.Continuation
    ) throws {
        try checkCancellation()
        let scan = try scanPayloads(checkCancellation: checkCancellation)
        try checkCancellation()
        let payloads = scan.payloads

        // Unreadable subtrees fail a FIRST sync loudly: nothing is at risk of deletion
        // yet, and a partially indexed source must not look complete. On later syncs the
        // walk continues; previously indexed files under unreadable paths are shielded
        // from delete detection and surfaced as `.permissionChanged` events instead, so
        // one bad directory cannot block syncing or purge valid documents.
        if previousCursor == nil, let failure = scan.unreadable.first {
            throw ConnectorEngineError(
                .sourceUnavailable,
                code: "connector.local.enumeration-incomplete",
                summary: "The selected directory could not be fully enumerated.",
                detail: "\(failure.url.path): \(failure.reason)"
            )
        }

        var currentFingerprints: [String: String] = [:]
        var currentDocumentIDs: Set<String> = []
        for payload in payloads {
            try checkCancellation()
            currentFingerprints[payload.documentID.rawValue] = fileFingerprint(for: payload)
            currentDocumentIDs.insert(payload.documentID.rawValue)
        }
        var missingDocumentIDs: [String] = []
        for documentID in previousCursor?.documentIDs ?? [] {
            try checkCancellation()
            if !currentDocumentIDs.contains(documentID) {
                missingDocumentIDs.append(documentID)
            }
        }
        // `relativePath(for:)` leaves a non-descendant path absolute, so the root
        // itself — the enumerator's own failure URL when the walk cannot start —
        // must not become a prefix: an absolute prefix matches no document-relative
        // path and would reclassify every shielded document as deleted.
        var unreadablePrefixes: [String] = []
        var rootIsUnreadable = false
        for failure in scan.unreadable {
            try checkCancellation()
            let failurePath = failure.url.standardizedFileURL.path
            if failurePath == rootURL.path || !failurePath.hasPrefix(rootURL.path + "/") {
                rootIsUnreadable = true
            } else {
                unreadablePrefixes.append(relativePath(for: failure.url))
            }
        }
        let (shieldedDocumentIDs, deletedDocumentIDs): ([String], [String])
        if rootIsUnreadable {
            (shieldedDocumentIDs, deletedDocumentIDs) = (missingDocumentIDs, [])
        } else {
            (shieldedDocumentIDs, deletedDocumentIDs) = try partitionShielded(
                missingDocumentIDs,
                unreadablePrefixes: unreadablePrefixes,
                checkCancellation: checkCancellation
            )
        }

        let previousFingerprints = previousCursor?.fileFingerprints ?? [:]
        // Shielded documents stay in the cursor with their last known fingerprints:
        // when the subtree becomes readable again, unchanged files do not re-upsert
        // and files deleted in the interim are still detected.
        var cursorFingerprints = currentFingerprints
        for documentID in shieldedDocumentIDs {
            try checkCancellation()
            // An unknown previous fingerprint still keeps the document tracked (empty
            // sentinel): it re-upserts once the subtree is readable again instead of
            // silently dropping out of delete detection.
            cursorFingerprints[documentID] = previousFingerprints[documentID] ?? ""
        }
        let currentCursor = LocalFileCursor(
            rootPath: rootURL.path,
            fingerprint: try fingerprint(for: payloads, checkCancellation: checkCancellation),
            fileFingerprints: cursorFingerprints
        )
        let nextCursor = try currentCursor.encoded()
        try checkCancellation()
        let rootMoved = previousCursor.map { $0.rootPath != rootURL.path } ?? false
        let changedPayloads: [SourcePayload]
        if previousCursor == nil || rootMoved || previousFingerprints.isEmpty {
            changedPayloads = payloads
        } else {
            var changed: [SourcePayload] = []
            for payload in payloads {
                try checkCancellation()
                if previousFingerprints[payload.documentID.rawValue] != currentFingerprints[payload.documentID.rawValue] {
                    changed.append(payload)
                }
            }
            changedPayloads = changed
        }
        let hasChanges = !deletedDocumentIDs.isEmpty || !changedPayloads.isEmpty || !shieldedDocumentIDs.isEmpty
        // Every unreadable path is reported, even when it shields no indexed
        // documents (e.g. a subtree created unreadable after the last sync) —
        // a partial walk must stay visible, not just non-destructive.
        var unavailablePaths: [(path: String, reason: String)] = []
        for failure in scan.unreadable {
            try checkCancellation()
            unavailablePaths.append((path: relativePath(for: failure.url), reason: failure.reason))
        }
        unavailablePaths.sort { $0.path < $1.path }

        for entry in unavailablePaths {
            try checkCancellation()
            continuation.yield(.pathUnavailable(path: entry.path, reason: entry.reason))
        }
        if hasChanges {
            for documentID in deletedDocumentIDs.sorted() {
                try checkCancellation()
                continuation.yield(.delete(documentID: EngineID(rawValue: documentID)))
            }
            for documentID in shieldedDocumentIDs.sorted() {
                try checkCancellation()
                continuation.yield(.permissionChanged(documentID: EngineID(rawValue: documentID)))
            }
            for payload in changedPayloads {
                try checkCancellation()
                continuation.yield(.upsert(payload))
            }
        }

        try checkCancellation()
        continuation.yield(.checkpoint(nextCursor))
    }

    /// Split documents that vanished from the walk into those under an unreadable
    /// path (kept, reported as permission changes) and true deletions.
    private func partitionShielded(
        _ missingDocumentIDs: [String],
        unreadablePrefixes: [String],
        checkCancellation: () throws -> Void
    ) throws -> (shielded: [String], deleted: [String]) {
        guard !unreadablePrefixes.isEmpty else { return ([], missingDocumentIDs) }
        let idPrefix = "\(id.rawValue):"
        var shielded: [String] = []
        var deleted: [String] = []
        for documentID in missingDocumentIDs {
            try checkCancellation()
            let relative = documentID.hasPrefix(idPrefix)
                ? String(documentID.dropFirst(idPrefix.count))
                : documentID
            let isShielded = unreadablePrefixes.contains { prefix in
                relative == prefix || relative.hasPrefix(prefix + "/")
            }
            if isShielded {
                shielded.append(documentID)
            } else {
                deleted.append(documentID)
            }
        }
        return (shielded, deleted)
    }

    public func fetch(_ reference: SourceReference) async throws -> SourcePayload {
        guard reference.connectorID == id else {
            throw ConnectorEngineError(
                .configurationInvalid,
                code: "connector.local.wrong-connector",
                summary: "The source reference belongs to a different connector.",
                detail: reference.connectorID.rawValue
            )
        }

        let url = reference.uri.standardizedFileURL
        let resolvedURL = url.resolvingSymlinksInPath()
        let resolvedRoot = rootURL.resolvingSymlinksInPath()
        guard resolvedURL.path == resolvedRoot.path || resolvedURL.path.hasPrefix(resolvedRoot.path + "/") else {
            throw ConnectorEngineError(
                .permissionDenied,
                code: "connector.local.outside-root",
                summary: "The requested file is outside the connector root.",
                detail: url.path
            )
        }

        return try payload(for: url)
    }

    struct ScanResult {
        var payloads: [SourcePayload]
        var unreadable: [(url: URL, reason: String)]
    }

    private struct FileInspectionFailure: Error {
        var url: URL
        var reason: String
    }

    func scanPayloads(checkCancellation: () throws -> Void) throws -> ScanResult {
        try checkCancellation()
        let values = try rootURL.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey])
        if values.isRegularFile == true {
            let payload = try payload(for: rootURL)
            try checkCancellation()
            return ScanResult(payloads: [payload], unreadable: [])
        }

        guard values.isDirectory == true else {
            throw ConnectorEngineError(
                .unsupportedSource,
                code: "connector.local.unsupported-root",
                summary: "The selected local source is not a regular file or directory.",
                detail: rootURL.path
            )
        }

        var options: FileManager.DirectoryEnumerationOptions = [.skipsPackageDescendants]
        if !self.options.includeHiddenFiles {
            options.insert(.skipsHiddenFiles)
        }

        var unreadable: [(url: URL, reason: String)] = []
        guard let enumerator = FileManager.default.enumerator(
            at: rootURL,
            includingPropertiesForKeys: Self.inspectionKeys,
            options: options,
            errorHandler: { url, error in
                unreadable.append((url, error.localizedDescription))
                return true
            }
        ) else {
            throw ConnectorEngineError(
                .sourceUnavailable,
                code: "connector.local.enumeration-failed",
                summary: "The selected directory could not be enumerated.",
                detail: rootURL.path
            )
        }

        var payloads: [SourcePayload] = []
        for case let url as URL in enumerator {
            try checkCancellation()
            do {
                guard let payload = try payloadIfSupported(url.standardizedFileURL) else { continue }
                payloads.append(payload)
            } catch let failure as FileInspectionFailure {
                unreadable.append((failure.url, failure.reason))
            }
        }

        try checkCancellation()
        payloads.sort { $0.documentID.rawValue < $1.documentID.rawValue }
        return ScanResult(payloads: payloads, unreadable: unreadable)
    }

    /// Every resource key a payload needs, shared by the walk and per-file inspection.
    private static let inspectionKeys: [URLResourceKey] = [
        .isRegularFileKey,
        .isHiddenKey,
        .fileSizeKey,
        .contentModificationDateKey,
        .attributeModificationDateKey,
        .contentTypeKey
    ]

    private func payloadIfSupported(_ url: URL) throws -> SourcePayload? {
        let values: URLResourceValues
        do {
            values = try url.resourceValues(forKeys: Set(Self.inspectionKeys))
        } catch {
            // A file that vanished mid-walk is a deletion the cursor diff will pick up;
            // one that exists but cannot be inspected is reported as unreadable so the
            // walk can continue and the sync stays visible instead of aborting.
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            throw FileInspectionFailure(url: url, reason: error.localizedDescription)
        }

        guard values.isRegularFile == true else { return nil }
        // A hidden file explicitly chosen as the connector root was deliberately
        // selected; the hidden-file policy only filters files discovered by the walk.
        let isExplicitRoot = url.standardizedFileURL.path == rootURL.path
        if !options.includeHiddenFiles, values.isHidden == true, !isExplicitRoot { return nil }

        let pathExtension = url.pathExtension.lowercased()
        if let allowed = options.allowedPathExtensions, !allowed.contains(pathExtension) {
            return nil
        }

        if let size = values.fileSize, Int64(size) > options.maxFileSizeBytes {
            return nil
        }

        return makePayload(for: url, values: values)
    }

    private func payload(for url: URL) throws -> SourcePayload {
        let supported: SourcePayload?
        do {
            supported = try payloadIfSupported(url)
        } catch let failure as FileInspectionFailure {
            // The walk tolerates unreadable entries, but a directly requested file
            // (single-file root, fetch) must fail with the typed connector error.
            throw ConnectorEngineError(
                .sourceUnavailable,
                code: "connector.local.stat-failed",
                summary: "The selected file could not be inspected.",
                detail: "\(failure.url.path): \(failure.reason)"
            )
        }
        guard let payload = supported else {
            throw ConnectorEngineError(
                .unsupportedSource,
                code: "connector.local.unsupported-file",
                summary: "The selected file is outside the connector policy.",
                detail: url.path
            )
        }

        return payload
    }

    private func makePayload(for url: URL, values: URLResourceValues) -> SourcePayload {
        let relativePath = relativePath(for: url)
        var metadata: [String: MetadataValue] = [
            "relativePath": .string(relativePath),
            "pathExtension": .string(url.pathExtension.lowercased())
        ]

        if let size = values.fileSize {
            metadata["fileSize"] = .integer(Int64(size))
        }
        if let modified = values.contentModificationDate {
            metadata["modifiedAt"] = .double(modified.timeIntervalSince1970)
        }
        if let attributesModified = values.attributeModificationDate {
            metadata["attributesModifiedAt"] = .double(attributesModified.timeIntervalSince1970)
        }

        return SourcePayload(
            documentID: EngineID(rawValue: "\(id.rawValue):\(relativePath)"),
            sourceID: id,
            sourceURI: url,
            displayName: url.lastPathComponent,
            contentType: values.contentType?.identifier ?? UTType.data.identifier,
            body: .binaryReference(url),
            metadata: metadata
        )
    }

    private func relativePath(for url: URL) -> String {
        let rootPath = rootURL.path
        var path = url.standardizedFileURL.path
        if path.hasPrefix(rootPath + "/") {
            path.removeFirst(rootPath.count + 1)
        }
        return path
    }

    private func decodedCursor(_ cursor: SourceCursor?) throws -> LocalFileCursor? {
        guard let cursor else { return nil }
        let data: Data
        if cursor.hasPrefix(Self.cursorPrefix) {
            let encodedPayload = String(cursor.dropFirst(Self.cursorPrefix.count))
            guard let compressed = Data(base64URLEncoded: encodedPayload) else {
                throw invalidCursor("The cursor payload is not valid base64.", detail: cursor)
            }
            do {
                data = try (compressed as NSData).decompressed(using: .zlib) as Data
            } catch {
                throw invalidCursor("The cursor payload could not be decompressed.", detail: error.localizedDescription)
            }
        } else if cursor.hasPrefix(Self.legacyCursorPrefix) {
            let encodedPayload = String(cursor.dropFirst(Self.legacyCursorPrefix.count))
            guard let decoded = Data(base64URLEncoded: encodedPayload) else {
                throw invalidCursor("The cursor payload is not valid base64.", detail: cursor)
            }
            data = decoded
        } else {
            throw invalidCursor("The cursor does not belong to the local file connector.", detail: cursor)
        }
        let decoded: LocalFileCursor
        do {
            decoded = try JSONDecoder().decode(LocalFileCursor.self, from: data)
        } catch {
            throw invalidCursor("The cursor payload could not be decoded.", detail: error.localizedDescription)
        }
        guard (1...LocalFileCursor.currentVersion).contains(decoded.version) else {
            throw invalidCursor(
                "The cursor version is not supported.",
                detail: "Expected at most \(LocalFileCursor.currentVersion), got \(decoded.version)."
            )
        }
        return decoded
    }

    private func invalidCursor(_ summary: String, detail: String) -> ConnectorEngineError {
        ConnectorEngineError(
            .cursorInvalid,
            code: "connector.local.cursor-invalid",
            summary: summary,
            detail: detail
        )
    }

    private func fileFingerprint(for payload: SourcePayload) -> String {
        var hasher = StableFNV1A()
        hasher.update("local-file-v3")
        hasher.update(payload.documentID.rawValue)
        if case let .integer(size)? = payload.metadata["fileSize"] {
            hasher.update(String(size))
        }
        if case let .double(modifiedAt)? = payload.metadata["modifiedAt"] {
            hasher.update(String(modifiedAt.bitPattern))
        }
        if case let .double(attributesModifiedAt)? = payload.metadata["attributesModifiedAt"] {
            hasher.update(String(attributesModifiedAt.bitPattern))
        }
        return String(hasher.value, radix: 16)
    }

    private func fingerprint(
        for payloads: [SourcePayload],
        checkCancellation: () throws -> Void
    ) throws -> String {
        var hasher = StableFNV1A()
        hasher.update(rootURL.path)
        // `scanPayloads` establishes document-ID order before returning.
        for payload in payloads {
            try checkCancellation()
            hasher.update(fileFingerprint(for: payload))
        }
        return "\(payloads.count)|\(String(hasher.value, radix: 16))"
    }
}

final class LocalFileScanCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var isCancelled = false

    func cancel() {
        lock.withLock {
            isCancelled = true
        }
    }

    func check() throws {
        if lock.withLock({ isCancelled }) {
            throw CancellationError()
        }
    }
}

private struct LocalFileCursor: Codable, Hashable, Sendable {
    static let currentVersion = 3

    var version: Int = currentVersion
    var rootPath: String
    var fingerprint: String
    var fileFingerprints: [String: String]

    /// Tracked membership is the fingerprint key set; version 1 stored it redundantly.
    var documentIDs: [String] { fileFingerprints.keys.sorted() }

    init(
        version: Int = currentVersion,
        rootPath: String,
        fingerprint: String,
        fileFingerprints: [String: String]
    ) {
        self.version = version
        self.rootPath = rootPath
        self.fingerprint = fingerprint
        self.fileFingerprints = fileFingerprints
    }

    private enum CodingKeys: String, CodingKey {
        case version
        case rootPath
        case fingerprint
        case documentIDs
        case fileFingerprints
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(Int.self, forKey: .version)
        rootPath = try container.decode(String.self, forKey: .rootPath)
        fingerprint = try container.decode(String.self, forKey: .fingerprint)
        var fingerprints = try container.decodeIfPresent([String: String].self, forKey: .fileFingerprints) ?? [:]
        // Version-1 cursors tracked membership in a separate ID list; fold IDs without
        // a fingerprint in as empty sentinels so delete detection still covers them.
        let legacyDocumentIDs = try container.decodeIfPresent([String].self, forKey: .documentIDs) ?? []
        for documentID in legacyDocumentIDs where fingerprints[documentID] == nil {
            fingerprints[documentID] = ""
        }
        fileFingerprints = fingerprints
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(version, forKey: .version)
        try container.encode(rootPath, forKey: .rootPath)
        try container.encode(fingerprint, forKey: .fingerprint)
        try container.encode(fileFingerprints, forKey: .fileFingerprints)
    }

    func encoded() throws -> SourceCursor {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        let compressed = try (data as NSData).compressed(using: .zlib) as Data
        return LocalFileConnector.cursorPrefix + compressed.base64URLEncodedString()
    }
}

private extension Data {
    init?(base64URLEncoded value: String) {
        var base64 = value
            .replacing("-", with: "+")
            .replacing("_", with: "/")
        let remainder = base64.count % 4
        if remainder > 0 {
            base64 += String(repeating: "=", count: 4 - remainder)
        }
        self.init(base64Encoded: base64)
    }

    func base64URLEncodedString() -> String {
        base64EncodedString()
            .replacing("+", with: "-")
            .replacing("/", with: "_")
            .replacing("=", with: "")
    }
}
