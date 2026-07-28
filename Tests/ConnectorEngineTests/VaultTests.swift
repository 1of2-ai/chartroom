import Foundation
import Testing
import IndexEngine
@testable import ConnectorEngine

@Suite("Vault ingestion (Obsidian markdown)")
struct VaultTests {
    @Test("parses frontmatter and body")
    func parsesFrontmatter() {
        let md = """
        ---
        id: n-capture
        type: note
        title: Capture Plan
        cluster: c1
        ---
        # Heading
        thermal rig notes here
        """
        let o = Vault.parse(markdown: md, relativePath: "Projects/Capture.md")
        #expect(o.id == "n-capture")
        #expect(o.type == "note")
        #expect(o.title == "Capture Plan")
        #expect(o.clusterID == "c1")
        #expect(o.body.contains("thermal rig notes"))
    }

    @Test("derives defaults without frontmatter")
    func derivesDefaults() {
        let o = Vault.parse(markdown: "# Budget Ask\nsitting with finance", relativePath: "Inbox/budget.md")
        #expect(o.id == "Inbox/budget.md")
        #expect(o.type == "note")
        #expect(o.title == "Budget Ask")
        #expect(o.clusterID == nil)
    }

    @Test("falls back to the file stem when there is no heading")
    func fileStemTitle() {
        let o = Vault.parse(markdown: "just some body text", relativePath: "Notes/thermal-pad.md")
        #expect(o.title == "thermal-pad")
    }

    @Test("treats unclosed frontmatter as body text")
    func unclosedFrontmatterIsBodyText() {
        let o = Vault.parse(markdown: "---\ntitle: Not Frontmatter\n# Actual Heading\nbody", relativePath: "Notes/broken.md")
        #expect(o.id == "Notes/broken.md")
        #expect(o.title == "Actual Heading")
        #expect(o.body.contains("title: Not Frontmatter"))
    }

    @Test("ingests a directory of markdown and it becomes searchable")
    func ingestsAndSearches() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("vault-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try "---\ntitle: Thermal\n---\nthe m-series rig is thermal throttling"
            .write(to: dir.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
        try "# Budget\nthe studio budget ask is with finance"
            .write(to: dir.appendingPathComponent("b.md"), atomically: true, encoding: .utf8)

        let store = try IndexStore(path: ":memory:", embedder: HashingEmbedder(dimension: 256))
        let n = try await store.ingestVault(at: dir)
        #expect(n == 2)
        #expect(try await store.count() == 2)

        let hits = try await store.search("thermal throttling", limit: 5)
        #expect(hits.first?.title == "Thermal")
    }

    @Test("bad vault files record failures without aborting the walk")
    func badVaultFilesRecordFailuresWithoutAborting() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("vault-bad-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try "# Good\nsearchable thermal note"
            .write(to: dir.appendingPathComponent("good.md"), atomically: true, encoding: .utf8)
        try Data([0xFF, 0xFE, 0x00])
            .write(to: dir.appendingPathComponent("bad.md"))

        let store = try IndexStore(path: ":memory:", embedder: HashingEmbedder(dimension: 256))
        let count = try await store.ingestVault(at: dir)
        #expect(count == 1)
        #expect(try await store.count() == 1)

        let failures = try await store.failureSnapshots(limit: 10)
        let badFile = try #require(failures.first { $0.documentID == "bad.md" })
        #expect(badFile.category == .extractionFailure)
        #expect(badFile.recoverability == .needsUserAction)

        let hits = try await store.search("thermal", limit: 5)
        #expect(hits.first?.documentID == "good.md")
    }

    @Test("vault ingestion never follows a markdown symlink outside the resolved root")
    func skipsOutsideSymlink() throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("vault-link-\(UUID().uuidString)")
        let vault = base.appendingPathComponent("Vault")
        try FileManager.default.createDirectory(at: vault, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }

        try "# Inside\nsearchable local note"
            .write(to: vault.appendingPathComponent("inside.md"), atomically: true, encoding: .utf8)
        let outside = base.appendingPathComponent("outside.md")
        try "# Secret\noutside vault content"
            .write(to: outside, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            at: vault.appendingPathComponent("linked.md"),
            withDestinationURL: outside
        )

        #expect(Vault.markdownFiles(in: vault).map(\.lastPathComponent) == ["inside.md"])
    }

    @Test("vault ingestion skips hidden files and package descendants")
    func skipsHiddenFilesAndPackages() throws {
        let vault = FileManager.default.temporaryDirectory.appendingPathComponent("vault-hidden-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: vault, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: vault) }

        try "# Visible\nincluded"
            .write(to: vault.appendingPathComponent("visible.md"), atomically: true, encoding: .utf8)
        try "# Hidden\nexcluded"
            .write(to: vault.appendingPathComponent(".hidden.md"), atomically: true, encoding: .utf8)
        let package = vault.appendingPathComponent("Archived.bundle")
        try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
        try "# Packaged\nexcluded"
            .write(to: package.appendingPathComponent("packaged.md"), atomically: true, encoding: .utf8)

        #expect(Vault.markdownFiles(in: vault).map(\.lastPathComponent) == ["visible.md"])
    }

    @Test("vault ingestion ignores non-regular markdown entries")
    func skipsNonRegularMarkdownEntries() throws {
        let vault = FileManager.default.temporaryDirectory.appendingPathComponent("vault-regular-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: vault, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: vault) }
        try FileManager.default.createDirectory(
            at: vault.appendingPathComponent("directory.md"),
            withIntermediateDirectories: true
        )

        #expect(Vault.markdownFiles(in: vault).isEmpty)
    }

    @Test("vault ingestion enforces the local-file 20 MiB bound")
    func skipsOversizedMarkdown() throws {
        let vault = FileManager.default.temporaryDirectory.appendingPathComponent("vault-large-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: vault, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: vault) }
        try Data(repeating: 0x61, count: 20 * 1024 * 1024 + 1)
            .write(to: vault.appendingPathComponent("oversized.md"))

        #expect(Vault.markdownFiles(in: vault).isEmpty)
    }

    @Test("secure vault reads reject a discovered file replaced by an outside symlink")
    func secureReadRejectsAcceptedFileReplacement() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("vault-replaced-file-\(UUID().uuidString)")
        let vault = base.appendingPathComponent("Vault")
        try FileManager.default.createDirectory(at: vault, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }

        let acceptedPath = vault.appendingPathComponent("accepted.md")
        try "# Accepted\ninside content"
            .write(to: acceptedPath, atomically: true, encoding: .utf8)
        let outsidePath = base.appendingPathComponent("outside.md")
        try "# Outside\nmust never be read"
            .write(to: outsidePath, atomically: true, encoding: .utf8)

        let reader = try VaultFileReader(rootURL: vault)
        let acceptedURL = try #require(Vault.markdownFiles(in: reader.rootURL).first)
        try FileManager.default.removeItem(at: acceptedPath)
        try FileManager.default.createSymbolicLink(at: acceptedPath, withDestinationURL: outsidePath)

        #expect(throws: VaultFileReadError.self) {
            _ = try reader.readMarkdown(at: acceptedURL)
        }
    }

    @Test("secure vault reads reject a discovered file whose ancestor becomes a symlink")
    func secureReadRejectsAcceptedAncestorReplacement() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("vault-replaced-ancestor-\(UUID().uuidString)")
        let vault = base.appendingPathComponent("Vault")
        let notes = vault.appendingPathComponent("Notes")
        try FileManager.default.createDirectory(at: notes, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }

        let acceptedPath = notes.appendingPathComponent("accepted.md")
        try "# Accepted\ninside content"
            .write(to: acceptedPath, atomically: true, encoding: .utf8)
        let outsideDirectory = base.appendingPathComponent("Outside")
        try FileManager.default.createDirectory(at: outsideDirectory, withIntermediateDirectories: true)
        try "# Outside\nmust never be read"
            .write(
                to: outsideDirectory.appendingPathComponent("accepted.md"),
                atomically: true,
                encoding: .utf8
            )

        let reader = try VaultFileReader(rootURL: vault)
        let acceptedURL = try #require(Vault.markdownFiles(in: reader.rootURL).first)
        try FileManager.default.removeItem(at: notes)
        try FileManager.default.createSymbolicLink(at: notes, withDestinationURL: outsideDirectory)

        #expect(throws: VaultFileReadError.self) {
            _ = try reader.readMarkdown(at: acceptedURL)
        }
    }

    @Test("secure vault reads accept a markdown file exactly 20 MiB long")
    func secureReadAcceptsExactSizeBoundary() throws {
        let vault = FileManager.default.temporaryDirectory
            .appendingPathComponent("vault-size-boundary-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: vault, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: vault) }

        let boundarySize = 20 * 1024 * 1024
        let file = vault.appendingPathComponent("boundary.md")
        try Data(repeating: 0x61, count: boundarySize).write(to: file)

        let reader = try VaultFileReader(rootURL: vault)
        let acceptedURL = try #require(Vault.markdownFiles(in: reader.rootURL).first)
        #expect(try reader.readMarkdown(at: acceptedURL).utf8.count == boundarySize)
    }
}
