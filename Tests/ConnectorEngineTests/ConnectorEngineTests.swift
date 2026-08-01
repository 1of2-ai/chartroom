import Darwin
import Foundation
import IndexEngine
import Testing
@testable import ConnectorEngine

@Suite("ConnectorEngine contracts")
struct ConnectorEngineTests {
    private struct ScanCheckpointCancellation: Error {}

    @Test("connectors emit IndexEngine payloads without owning indexing")
    func connectorEmitsEnginePayloads() async throws {
        let connector = FixtureConnector()
        try await connector.validate()

        var events: [SourceEvent] = []
        for try await event in try await connector.changes(since: nil) {
            events.append(event)
        }

        #expect(events.count == 2)
        guard case let .upsert(payload) = events.first else {
            Issue.record("Expected first event to be an upsert")
            return
        }
        #expect(payload.documentID == "fixture-doc")
        #expect(payload.sourceID == connector.id)

        guard case let .checkpoint(cursor) = events.last else {
            Issue.record("Expected final event to be a checkpoint")
            return
        }
        #expect(cursor == "cursor-1")
    }

    @Test("local file connector walks directories into source payload events")
    func localFileConnectorWalksDirectory() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let noteURL = root.appending(path: "Notes/Retrieval.md")
        try FileManager.default.createDirectory(at: noteURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try "IndexEngine source payload".write(to: noteURL, atomically: true, encoding: .utf8)
        try "hidden".write(to: root.appending(path: ".ignored.md"), atomically: true, encoding: .utf8)
        try Data(repeating: 1, count: 32).write(to: root.appending(path: "image.bin"))

        let connector = LocalFileConnector(
            rootURL: root,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: [".md"])
        )

        var events: [SourceEvent] = []
        for try await event in try await connector.changes(since: nil) {
            events.append(event)
        }

        #expect(events.count == 2)

        guard case let .upsert(payload) = events.first else {
            Issue.record("Expected an upsert for the markdown file")
            return
        }

        #expect(payload.documentID == "local-test:Notes/Retrieval.md")
        #expect(payload.sourceID == "local-test")
        #expect(payload.sourceURI == noteURL.standardizedFileURL)
        #expect(payload.displayName == "Retrieval.md")
        #expect(payload.metadata["relativePath"] == .string("Notes/Retrieval.md"))

        guard case let .checkpoint(cursor) = events.last else {
            Issue.record("Expected a checkpoint")
            return
        }

        var repeatEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: cursor) {
            repeatEvents.append(event)
        }

        #expect(repeatEvents == [.checkpoint(cursor)])

        try "IndexEngine source payload with changed searchable content".write(to: noteURL, atomically: true, encoding: .utf8)

        var changedEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: cursor) {
            changedEvents.append(event)
        }

        #expect(changedEvents.count == 2)
        guard case let .upsert(changedPayload) = changedEvents.first else {
            Issue.record("Expected an upsert after the file changed")
            return
        }
        #expect(changedPayload.documentID == "local-test:Notes/Retrieval.md")
    }

    @Test("local file scanning checks cancellation between directory entries")
    func localFileScanningChecksCancellationBetweenEntries() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for index in 0..<5 {
            try "entry \(index)".write(
                to: root.appending(path: "\(index).md"),
                atomically: true,
                encoding: .utf8
            )
        }

        let connector = LocalFileConnector(rootURL: root, id: "local-test")
        var checkpointCount = 0

        #expect(throws: ScanCheckpointCancellation.self) {
            _ = try connector.scanPayloads {
                checkpointCount += 1
                if checkpointCount == 3 {
                    throw ScanCheckpointCancellation()
                }
            }
        }
        #expect(checkpointCount == 3)
    }

    @Test("local file connector emits deletes for files removed since the prior cursor")
    func localFileConnectorEmitsDeletesForRemovedFiles() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let keptURL = root.appending(path: "Kept.md")
        let removedURL = root.appending(path: "Removed.md")
        try "kept source payload".write(to: keptURL, atomically: true, encoding: .utf8)
        try "removed source payload".write(to: removedURL, atomically: true, encoding: .utf8)

        let connector = LocalFileConnector(
            rootURL: root,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )

        var initialEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: nil) {
            initialEvents.append(event)
        }

        guard case let .checkpoint(cursor) = initialEvents.last else {
            Issue.record("Expected an initial checkpoint")
            return
        }

        try FileManager.default.removeItem(at: removedURL)

        var removalEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: cursor) {
            removalEvents.append(event)
        }

        #expect(removalEvents.count == 2)
        #expect(removalEvents.contains(.delete(documentID: "local-test:Removed.md")))
        #expect(!removalEvents.contains { event in
            guard case .upsert = event else { return false }
            return true
        })

        guard case .checkpoint = removalEvents.last else {
            Issue.record("Expected a checkpoint after removal")
            return
        }
    }

    @Test("local file connector only re-upserts files whose fingerprints changed")
    func localFileConnectorOnlyUpsertsChangedFiles() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let changedURL = root.appending(path: "Changed.md")
        let unchangedURL = root.appending(path: "Unchanged.md")
        try "first body".write(to: changedURL, atomically: true, encoding: .utf8)
        try "stable body".write(to: unchangedURL, atomically: true, encoding: .utf8)

        let connector = LocalFileConnector(
            rootURL: root,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )

        var initialEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: nil) {
            initialEvents.append(event)
        }
        guard case let .checkpoint(cursor) = initialEvents.last else {
            Issue.record("Expected an initial checkpoint")
            return
        }

        try await Task.sleep(for: .milliseconds(20))
        try "second body".write(to: changedURL, atomically: true, encoding: .utf8)

        var changedEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: cursor) {
            changedEvents.append(event)
        }

        let upsertedIDs = changedEvents.compactMap { event -> DocumentID? in
            guard case let .upsert(payload) = event else { return nil }
            return payload.documentID
        }
        #expect(upsertedIDs == ["local-test:Changed.md"])
    }

    @Test("local file connector detects same-size rewrites whose mtime is restored")
    func localFileConnectorDetectsPreservedMtimeRewrite() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let noteURL = root.appending(path: "Restored.md")
        try Data("alpha".utf8).write(to: noteURL)
        let connector = LocalFileConnector(
            rootURL: root,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )

        var initialEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: nil) {
            initialEvents.append(event)
        }
        guard case let .checkpoint(cursor) = initialEvents.last else {
            Issue.record("Expected an initial checkpoint")
            return
        }

        var before = stat()
        #expect(Darwin.lstat(noteURL.path, &before) == 0)
        try await Task.sleep(for: .milliseconds(20))
        try Data("bravo".utf8).write(to: noteURL)
        let originalTimes = [before.st_atimespec, before.st_mtimespec]
        let restoreResult = originalTimes.withUnsafeBufferPointer { times in
            Darwin.utimensat(AT_FDCWD, noteURL.path, times.baseAddress, 0)
        }
        #expect(restoreResult == 0)

        var after = stat()
        #expect(Darwin.lstat(noteURL.path, &after) == 0)
        #expect(after.st_size == before.st_size)
        #expect(after.st_mtimespec.tv_sec == before.st_mtimespec.tv_sec)
        #expect(after.st_mtimespec.tv_nsec == before.st_mtimespec.tv_nsec)
        #expect(
            after.st_ctimespec.tv_sec != before.st_ctimespec.tv_sec
                || after.st_ctimespec.tv_nsec != before.st_ctimespec.tv_nsec
        )

        var restoredEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: cursor) {
            restoredEvents.append(event)
        }

        let upsertedIDs = restoredEvents.compactMap { event -> DocumentID? in
            guard case let .upsert(payload) = event else { return nil }
            return payload.documentID
        }
        #expect(upsertedIDs == ["local-test:Restored.md"])
    }

    @Test("local file connector rejects undecodable cursors instead of silently disabling deletes")
    func localFileConnectorRejectsInvalidCursor() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let connector = LocalFileConnector(rootURL: root, id: "local-test")

        do {
            _ = try await connector.changes(since: "not-a-local-file-cursor")
            Issue.record("Expected an invalid cursor error")
        } catch let error as ConnectorEngineError {
            #expect(error.category == .cursorInvalid)
            #expect(error.code == "connector.local.cursor-invalid")
        }
    }

    @Test("local file connector preserves delete detection when the root folder moves")
    func localFileConnectorDetectsDeletesAfterRootMove() async throws {
        let oldRoot = FileManager.default.temporaryDirectory.appending(path: "local-connector-old-\(UUID().uuidString)", directoryHint: .isDirectory)
        let newRoot = FileManager.default.temporaryDirectory.appending(path: "local-connector-new-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: oldRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: newRoot, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: oldRoot)
            try? FileManager.default.removeItem(at: newRoot)
        }

        try "kept body".write(to: oldRoot.appending(path: "Kept.md"), atomically: true, encoding: .utf8)
        try "removed body".write(to: oldRoot.appending(path: "Removed.md"), atomically: true, encoding: .utf8)
        try "kept body".write(to: newRoot.appending(path: "Kept.md"), atomically: true, encoding: .utf8)

        let oldConnector = LocalFileConnector(
            rootURL: oldRoot,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )
        var oldEvents: [SourceEvent] = []
        for try await event in try await oldConnector.changes(since: nil) {
            oldEvents.append(event)
        }
        guard case let .checkpoint(cursor) = oldEvents.last else {
            Issue.record("Expected an initial checkpoint")
            return
        }

        let newConnector = LocalFileConnector(
            rootURL: newRoot,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )
        var movedEvents: [SourceEvent] = []
        for try await event in try await newConnector.changes(since: cursor) {
            movedEvents.append(event)
        }

        #expect(movedEvents.contains(.delete(documentID: "local-test:Removed.md")))
        #expect(movedEvents.contains { event in
            guard case let .upsert(payload) = event else { return false }
            return payload.documentID == "local-test:Kept.md"
        })
    }

    @Test("unreadable subtrees shield indexed files instead of deleting them or blocking the sync")
    func localFileConnectorShieldsUnreadableSubtrees() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        let protectedDirectory = root.appending(path: "Protected", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: protectedDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: NSNumber(value: Int16(0o700))], ofItemAtPath: protectedDirectory.path)
            try? FileManager.default.removeItem(at: root)
        }

        try "visible".write(to: root.appending(path: "Visible.md"), atomically: true, encoding: .utf8)
        try "hidden".write(to: protectedDirectory.appending(path: "Hidden.md"), atomically: true, encoding: .utf8)

        let connector = LocalFileConnector(
            rootURL: root,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )
        var initialEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: nil) {
            initialEvents.append(event)
        }
        guard case let .checkpoint(cursor) = initialEvents.last else {
            Issue.record("Expected an initial checkpoint")
            return
        }

        try FileManager.default.setAttributes([.posixPermissions: NSNumber(value: Int16(0o000))], ofItemAtPath: protectedDirectory.path)

        var shieldedEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: cursor) {
            shieldedEvents.append(event)
        }
        #expect(!shieldedEvents.contains(.delete(documentID: "local-test:Protected/Hidden.md")))
        #expect(shieldedEvents.contains(.permissionChanged(documentID: "local-test:Protected/Hidden.md")))
        #expect(shieldedEvents.contains { event in
            guard case let .pathUnavailable(path, _) = event else { return false }
            return path == "Protected"
        })
        #expect(!shieldedEvents.contains { event in
            guard case .upsert = event else { return false }
            return true
        })
        guard case let .checkpoint(shieldedCursor) = shieldedEvents.last else {
            Issue.record("Expected a checkpoint after the shielded sync")
            return
        }

        // Once readable again, the carried fingerprint means the unchanged file is
        // neither deleted nor re-upserted.
        try FileManager.default.setAttributes([.posixPermissions: NSNumber(value: Int16(0o700))], ofItemAtPath: protectedDirectory.path)

        var restoredEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: shieldedCursor) {
            restoredEvents.append(event)
        }
        #expect(restoredEvents.count == 1)
        guard case .checkpoint = restoredEvents.last else {
            Issue.record("Expected only a checkpoint after permissions were restored")
            return
        }
    }

    @Test("an unreadable subtree with no indexed files is still reported as unavailable")
    func localFileConnectorReportsUnreadablePathsWithNothingToShield() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let protectedDirectory = root.appending(path: "Protected", directoryHint: .isDirectory)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: NSNumber(value: Int16(0o700))], ofItemAtPath: protectedDirectory.path)
            try? FileManager.default.removeItem(at: root)
        }

        try "visible".write(to: root.appending(path: "Visible.md"), atomically: true, encoding: .utf8)

        let connector = LocalFileConnector(
            rootURL: root,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )
        var initialEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: nil) {
            initialEvents.append(event)
        }
        guard case let .checkpoint(cursor) = initialEvents.last else {
            Issue.record("Expected an initial checkpoint")
            return
        }

        // The subtree appears after the first sync and was never readable, so there
        // is nothing to shield — the partial walk must still be visible.
        try FileManager.default.createDirectory(at: protectedDirectory, withIntermediateDirectories: true)
        try "hidden".write(to: protectedDirectory.appending(path: "Hidden.md"), atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: NSNumber(value: Int16(0o000))], ofItemAtPath: protectedDirectory.path)

        var events: [SourceEvent] = []
        for try await event in try await connector.changes(since: cursor) {
            events.append(event)
        }
        #expect(events.contains { event in
            guard case let .pathUnavailable(path, _) = event else { return false }
            return path == "Protected"
        })
        #expect(!events.contains { event in
            if case .delete = event { return true }
            if case .permissionChanged = event { return true }
            if case .upsert = event { return true }
            return false
        })
        guard case .checkpoint = events.last else {
            Issue.record("Expected a checkpoint after the degraded sync")
            return
        }
    }

    @Test("an unreadable root shields every indexed file instead of purging the source")
    func localFileConnectorShieldsUnreadableRoot() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        let subdirectory = root.appending(path: "Notes", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: subdirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: NSNumber(value: Int16(0o700))], ofItemAtPath: root.path)
            try? FileManager.default.removeItem(at: root)
        }

        try "top".write(to: root.appending(path: "Top.md"), atomically: true, encoding: .utf8)
        try "nested".write(to: subdirectory.appending(path: "Nested.md"), atomically: true, encoding: .utf8)

        let connector = LocalFileConnector(
            rootURL: root,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )
        var initialEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: nil) {
            initialEvents.append(event)
        }
        guard case let .checkpoint(cursor) = initialEvents.last else {
            Issue.record("Expected an initial checkpoint")
            return
        }

        // The whole root becomes unreadable — an unmounted volume or revoked
        // permission. This is the worst case shielding exists for: nothing may
        // be deleted.
        try FileManager.default.setAttributes([.posixPermissions: NSNumber(value: Int16(0o000))], ofItemAtPath: root.path)

        var shieldedEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: cursor) {
            shieldedEvents.append(event)
        }
        #expect(!shieldedEvents.contains { event in
            if case .delete = event { return true }
            return false
        })
        #expect(shieldedEvents.contains(.permissionChanged(documentID: "local-test:Top.md")))
        #expect(shieldedEvents.contains(.permissionChanged(documentID: "local-test:Notes/Nested.md")))
        guard case let .checkpoint(shieldedCursor) = shieldedEvents.last else {
            Issue.record("Expected a checkpoint after the shielded sync")
            return
        }

        // Once the root is readable again the carried fingerprints mean unchanged
        // files are neither deleted nor re-upserted.
        try FileManager.default.setAttributes([.posixPermissions: NSNumber(value: Int16(0o700))], ofItemAtPath: root.path)

        var restoredEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: shieldedCursor) {
            restoredEvents.append(event)
        }
        #expect(restoredEvents.count == 1)
        guard case .checkpoint = restoredEvents.last else {
            Issue.record("Expected only a checkpoint after the root became readable again")
            return
        }
    }

    @Test("a first sync with an unreadable subtree fails loudly instead of indexing a partial tree")
    func localFileConnectorFailsFirstSyncOnUnreadableSubtree() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        let protectedDirectory = root.appending(path: "Protected", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: protectedDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: NSNumber(value: Int16(0o700))], ofItemAtPath: protectedDirectory.path)
            try? FileManager.default.removeItem(at: root)
        }

        try "visible".write(to: root.appending(path: "Visible.md"), atomically: true, encoding: .utf8)
        try "hidden".write(to: protectedDirectory.appending(path: "Hidden.md"), atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: NSNumber(value: Int16(0o000))], ofItemAtPath: protectedDirectory.path)

        let connector = LocalFileConnector(
            rootURL: root,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )
        do {
            // The walk is lazy: the failure surfaces during iteration, not at the call.
            for try await _ in try await connector.changes(since: nil) {}
            Issue.record("Expected an incomplete first enumeration to throw")
        } catch let error as ConnectorEngineError {
            #expect(error.category == .sourceUnavailable)
            #expect(error.code == "connector.local.enumeration-incomplete")
        }
    }

    @Test("local file connector emits a selected file as one payload")
    func localFileConnectorEmitsSingleFile() async throws {
        let fileURL = FileManager.default.temporaryDirectory.appending(path: "local-file-\(UUID().uuidString).md")
        try "single file source payload".write(to: fileURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let connector = LocalFileConnector(
            rootURL: fileURL,
            id: "single-file",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )

        var events: [SourceEvent] = []
        for try await event in try await connector.changes(since: nil) {
            events.append(event)
        }

        #expect(events.count == 2)
        guard case let .upsert(payload) = events.first else {
            Issue.record("Expected one upsert for the selected file")
            return
        }

        #expect(payload.documentID == EngineID(rawValue: "single-file:\(fileURL.path)"))
        #expect(payload.sourceID == "single-file")
        #expect(payload.sourceURI == fileURL.standardizedFileURL)
        #expect(payload.metadata["relativePath"] == .string(fileURL.path))
    }

    @Test("local file connector rejects fetches outside its root")
    func localFileConnectorRejectsOutsideRootFetch() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let outside = FileManager.default.temporaryDirectory.appending(path: "outside-\(UUID().uuidString).md")
        try "outside".write(to: outside, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: outside) }

        let connector = LocalFileConnector(rootURL: root, id: "local-test")

        do {
            _ = try await connector.fetch(SourceReference(connectorID: "local-test", uri: outside))
            #expect(Bool(false), "Expected outside-root fetch to fail")
        } catch let error as ConnectorEngineError {
            #expect(error.category == .permissionDenied)
            #expect(error.code == "connector.local.outside-root")
        }
    }

    @Test("local file connector resolves symlinks before fetch containment checks")
    func localFileConnectorRejectsSymlinkEscapedFetch() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let outsideDirectory = FileManager.default.temporaryDirectory.appending(path: "outside-dir-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: outsideDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outsideDirectory) }

        let outsideFile = outsideDirectory.appending(path: "secret.md")
        try "outside through symlink".write(to: outsideFile, atomically: true, encoding: .utf8)
        let link = root.appending(path: "linked")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outsideDirectory)

        let connector = LocalFileConnector(rootURL: root, id: "local-test")
        let escapedURL = link.appending(path: "secret.md")

        do {
            _ = try await connector.fetch(SourceReference(connectorID: "local-test", uri: escapedURL))
            Issue.record("Expected symlink-escaped fetch to fail")
        } catch let error as ConnectorEngineError {
            #expect(error.category == .permissionDenied)
            #expect(error.code == "connector.local.outside-root")
        }
    }

    @Test("legacy v1 cursors still decode and drive delete detection")
    func localFileConnectorDecodesLegacyV1Cursor() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let keptURL = root.appending(path: "Kept.md")
        try "kept source payload".write(to: keptURL, atomically: true, encoding: .utf8)

        // A version-1 cursor as the previous connector release wrote it: plain
        // base64url JSON with a redundant documentIDs list and no compression.
        let legacyJSON = """
        {"version":1,"rootPath":"\(root.path)","fingerprint":"stale",\
        "documentIDs":["local-test:Kept.md","local-test:Removed.md"],"fileFingerprints":{}}
        """
        let legacyPayload = Data(legacyJSON.utf8).base64EncodedString()
            .replacing("+", with: "-")
            .replacing("/", with: "_")
            .replacing("=", with: "")
        let legacyCursor = "local-file-v1:" + legacyPayload

        let connector = LocalFileConnector(
            rootURL: root,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )

        var events: [SourceEvent] = []
        for try await event in try await connector.changes(since: legacyCursor) {
            events.append(event)
        }

        #expect(events.contains(.delete(documentID: "local-test:Removed.md")))
        guard case let .checkpoint(nextCursor) = events.last else {
            Issue.record("Expected a checkpoint after the legacy cursor sync")
            return
        }
        #expect(nextCursor.hasPrefix("local-file-v2:"))

        // The upgraded cursor keeps working: no deletes, no upserts on a quiet resync.
        var resyncEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: nextCursor) {
            resyncEvents.append(event)
        }
        #expect(resyncEvents.count == 1)
        guard case .checkpoint = resyncEvents.first else {
            Issue.record("Expected only a checkpoint on a quiet resync")
            return
        }
    }

    @Test("cursors stay compact for large trees")
    func localFileConnectorCursorIsCompact() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        var totalIdentifierBytes = 0
        for index in 0..<200 {
            let name = "Deeply/Nested/Notes/Folder-\(index / 20)/Meeting Notes \(index).md"
            let url = root.appending(path: name)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try "body \(index)".write(to: url, atomically: true, encoding: .utf8)
            totalIdentifierBytes += "local-test:\(name)".utf8.count
        }

        let connector = LocalFileConnector(
            rootURL: root,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )

        var events: [SourceEvent] = []
        for try await event in try await connector.changes(since: nil) {
            events.append(event)
        }
        guard case let .checkpoint(cursor) = events.last else {
            Issue.record("Expected a checkpoint")
            return
        }

        // The compressed cursor must undercut even the raw document-ID bytes it
        // tracks; the old JSON encoding stored each ID twice plus fingerprints.
        #expect(cursor.utf8.count < totalIdentifierBytes)
    }

    @Test("re-syncing from a stale cursor replays the same changes")
    func localFileConnectorReplaysFromStaleCursor() async throws {
        let root = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let keptURL = root.appending(path: "Kept.md")
        let removedURL = root.appending(path: "Removed.md")
        try "kept".write(to: keptURL, atomically: true, encoding: .utf8)
        try "removed".write(to: removedURL, atomically: true, encoding: .utf8)

        let connector = LocalFileConnector(
            rootURL: root,
            id: "local-test",
            options: LocalFileConnectorOptions(allowedPathExtensions: ["md"])
        )

        var initialEvents: [SourceEvent] = []
        for try await event in try await connector.changes(since: nil) {
            initialEvents.append(event)
        }
        guard case let .checkpoint(staleCursor) = initialEvents.last else {
            Issue.record("Expected an initial checkpoint")
            return
        }

        try FileManager.default.removeItem(at: removedURL)
        try await Task.sleep(for: .milliseconds(20))
        try "kept, edited".write(to: keptURL, atomically: true, encoding: .utf8)

        // A consumer that crashed before persisting the next checkpoint re-syncs
        // from the stale cursor; the connector must replay identical changes.
        func collect() async throws -> [SourceEvent] {
            var events: [SourceEvent] = []
            for try await event in try await connector.changes(since: staleCursor) {
                events.append(event)
            }
            return events
        }
        let firstReplay = try await collect()
        let secondReplay = try await collect()

        #expect(firstReplay == secondReplay)
        #expect(firstReplay.contains(.delete(documentID: "local-test:Removed.md")))
        #expect(firstReplay.contains { event in
            guard case let .upsert(payload) = event else { return false }
            return payload.documentID == "local-test:Kept.md"
        })
    }

    @Test("a hidden file explicitly chosen as the root is accepted")
    func localFileConnectorAcceptsHiddenRootFile() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "local-connector-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let hiddenURL = directory.appending(path: ".hidden-notes.md")
        try "hidden but deliberately selected".write(to: hiddenURL, atomically: true, encoding: .utf8)

        let connector = LocalFileConnector(rootURL: hiddenURL, id: "local-test")

        var events: [SourceEvent] = []
        for try await event in try await connector.changes(since: nil) {
            events.append(event)
        }

        let upserted = events.compactMap { event -> SourcePayload? in
            guard case let .upsert(payload) = event else { return nil }
            return payload
        }
        #expect(upserted.count == 1)
        #expect(upserted.first?.displayName == ".hidden-notes.md")
    }
}

private struct FixtureConnector: SourceConnector {
    let id: ConnectorID = "fixture"
    let capabilities = ConnectorCapabilities()

    func validate() async throws {}

    func changes(since cursor: SourceCursor?) async throws -> AsyncThrowingStream<SourceEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(
                .upsert(
                    SourcePayload(
                        documentID: "fixture-doc",
                        sourceID: id,
                        sourceURI: URL(filePath: "/tmp/fixture.md"),
                        displayName: "Fixture",
                        body: .text("connector payload for index engine ingestion")
                    )
                )
            )
            continuation.yield(.checkpoint("cursor-1"))
            continuation.finish()
        }
    }

    func fetch(_ reference: SourceReference) async throws -> SourcePayload {
        SourcePayload(
            documentID: EngineID(rawValue: reference.externalID ?? reference.uri.lastPathComponent),
            sourceID: id,
            sourceURI: reference.uri,
            displayName: reference.uri.lastPathComponent,
            body: .binaryReference(reference.uri)
        )
    }
}
