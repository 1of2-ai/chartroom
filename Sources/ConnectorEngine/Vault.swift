import Darwin
import Foundation
import IndexEngine

enum VaultFileReadError: Error, CustomStringConvertible {
    case invalidRoot(URL, errno: Int32)
    case outsideRoot(URL)
    case invalidRelativePath(URL)
    case unsafePathComponent(String, errno: Int32)
    case notRegularFile(URL)
    case fileTooLarge(URL, actualBytes: Int64)
    case readFailed(URL, errno: Int32)
    case invalidUTF8(URL)

    var description: String {
        switch self {
        case let .invalidRoot(url, code):
            "Could not securely open vault root \(url.path): \(String(cString: strerror(code)))"
        case let .outsideRoot(url):
            "Vault file is outside the opened root: \(url.path)"
        case let .invalidRelativePath(url):
            "Vault file has an invalid relative path: \(url.path)"
        case let .unsafePathComponent(component, code):
            "Could not securely open vault path component \(component): \(String(cString: strerror(code)))"
        case let .notRegularFile(url):
            "Vault entry is not a regular file: \(url.path)"
        case let .fileTooLarge(url, actualBytes):
            "Vault file exceeds 20 MiB (\(actualBytes) bytes): \(url.path)"
        case let .readFailed(url, code):
            "Could not read vault file \(url.path): \(String(cString: strerror(code)))"
        case let .invalidUTF8(url):
            "Vault file is not valid UTF-8: \(url.path)"
        }
    }
}

final class VaultFileReader {
    static let maximumMarkdownSize = 20 * 1024 * 1024

    let rootURL: URL
    private let rootDescriptor: Int32

    init(rootURL: URL) throws {
        let canonicalRoot = try Self.canonicalURL(for: rootURL)
        self.rootURL = canonicalRoot
        self.rootDescriptor = try Self.openCanonicalDirectory(canonicalRoot)
    }

    deinit {
        Darwin.close(rootDescriptor)
    }

    func relativePath(for url: URL) throws -> String {
        try relativeComponents(for: url).joined(separator: "/")
    }

    func readMarkdown(at url: URL) throws -> String {
        let components = try relativeComponents(for: url)
        let descriptor = try openFile(components: components, sourceURL: url)
        defer { Darwin.close(descriptor) }

        var status = stat()
        guard Darwin.fstat(descriptor, &status) == 0 else {
            throw VaultFileReadError.readFailed(url, errno: Darwin.errno)
        }
        guard (status.st_mode & S_IFMT) == S_IFREG else {
            throw VaultFileReadError.notRegularFile(url)
        }

        let measuredSize = Int64(status.st_size)
        guard measuredSize >= 0, measuredSize <= Int64(Self.maximumMarkdownSize) else {
            throw VaultFileReadError.fileTooLarge(url, actualBytes: measuredSize)
        }

        let data = try readBounded(descriptor, sourceURL: url, measuredSize: Int(measuredSize))
        guard let text = String(data: data, encoding: .utf8) else {
            throw VaultFileReadError.invalidUTF8(url)
        }
        return text
    }

    private func relativeComponents(for url: URL) throws -> [String] {
        let rootComponents = rootURL.pathComponents
        let candidateComponents = url.pathComponents
        guard candidateComponents.count > rootComponents.count,
              Array(candidateComponents.prefix(rootComponents.count)) == rootComponents else {
            throw VaultFileReadError.outsideRoot(url)
        }

        let relative = Array(candidateComponents.dropFirst(rootComponents.count))
        guard !relative.isEmpty,
              relative.allSatisfy({ !$0.isEmpty && $0 != "/" && $0 != "." && $0 != ".." }) else {
            throw VaultFileReadError.invalidRelativePath(url)
        }
        return relative
    }

    private func openFile(components: [String], sourceURL: URL) throws -> Int32 {
        var parentDescriptor = rootDescriptor
        var ownedDirectoryDescriptor: Int32?
        defer {
            if let ownedDirectoryDescriptor {
                Darwin.close(ownedDirectoryDescriptor)
            }
        }

        for component in components.dropLast() {
            let nextDescriptor = component.withCString {
                Darwin.openat(
                    parentDescriptor,
                    $0,
                    O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW
                )
            }
            guard nextDescriptor >= 0 else {
                throw VaultFileReadError.unsafePathComponent(component, errno: Darwin.errno)
            }
            if let ownedDirectoryDescriptor {
                Darwin.close(ownedDirectoryDescriptor)
            }
            ownedDirectoryDescriptor = nextDescriptor
            parentDescriptor = nextDescriptor
        }

        guard let filename = components.last else {
            throw VaultFileReadError.invalidRelativePath(sourceURL)
        }
        let descriptor = filename.withCString {
            Darwin.openat(
                parentDescriptor,
                $0,
                O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK
            )
        }
        guard descriptor >= 0 else {
            throw VaultFileReadError.unsafePathComponent(filename, errno: Darwin.errno)
        }
        return descriptor
    }

    private func readBounded(
        _ descriptor: Int32,
        sourceURL: URL,
        measuredSize: Int
    ) throws -> Data {
        var data = Data()
        data.reserveCapacity(measuredSize)
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)

        while data.count <= Self.maximumMarkdownSize {
            let remaining = Self.maximumMarkdownSize + 1 - data.count
            let requestedCount = min(buffer.count, remaining)
            let bytesRead = buffer.withUnsafeMutableBytes { bytes in
                Darwin.read(descriptor, bytes.baseAddress, requestedCount)
            }
            if bytesRead < 0 {
                if Darwin.errno == EINTR {
                    continue
                }
                throw VaultFileReadError.readFailed(sourceURL, errno: Darwin.errno)
            }
            guard bytesRead > 0 else { break }
            data.append(buffer, count: bytesRead)
            if data.count > Self.maximumMarkdownSize {
                throw VaultFileReadError.fileTooLarge(
                    sourceURL,
                    actualBytes: Int64(data.count)
                )
            }
        }
        return data
    }

    private static func openCanonicalDirectory(_ url: URL) throws -> Int32 {
        var descriptor = Darwin.open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw VaultFileReadError.invalidRoot(url, errno: Darwin.errno)
        }

        for component in url.pathComponents where component != "/" {
            let nextDescriptor = component.withCString {
                Darwin.openat(
                    descriptor,
                    $0,
                    O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW
                )
            }
            guard nextDescriptor >= 0 else {
                let code = Darwin.errno
                Darwin.close(descriptor)
                throw VaultFileReadError.invalidRoot(url, errno: code)
            }
            Darwin.close(descriptor)
            descriptor = nextDescriptor
        }
        return descriptor
    }

    static func canonicalURL(for url: URL) throws -> URL {
        let resolvedPath = url.path.withCString { Darwin.realpath($0, nil) }
        guard let resolvedPath else {
            throw VaultFileReadError.invalidRoot(url, errno: Darwin.errno)
        }
        defer { Darwin.free(resolvedPath) }
        return URL(filePath: String(cString: resolvedPath), directoryHint: .isDirectory)
    }
}

/// Parses Obsidian-style markdown into the index's search projection. The `.md`
/// on disk stays canonical (the Obsidian exception); what we store is metadata +
/// body text for retrieval, keyed by the vault-relative path so a hit points back
/// at the file.
public enum Vault {
    private static let inspectionKeys: Set<URLResourceKey> = [
        .isRegularFileKey,
        .isSymbolicLinkKey,
        .fileSizeKey
    ]

    /// Frontmatter (`--- ... ---`) supplies id/type/title/cluster when present;
    /// otherwise they are derived (id = relative path, title = first `#` heading
    /// or the file stem, type = "note").
    public static func parse(markdown text: String, relativePath: String) -> IndexedObject {
        let (front, body) = splitFrontmatter(text)
        let trimmedBody = body.trimmingCharacters(in: .whitespacesAndNewlines)
        return IndexedObject(
            id: front["id"] ?? relativePath,
            type: front["type"] ?? "note",
            title: front["title"] ?? firstHeading(body) ?? fileStem(relativePath),
            body: trimmedBody,
            clusterID: front["cluster"]
        )
    }

    static func splitFrontmatter(_ text: String) -> (front: [String: String], body: String) {
        let lines = text.components(separatedBy: "\n")
        guard lines.first?.trimmingCharacters(in: .whitespaces) == "---" else { return ([:], text) }
        var front: [String: String] = [:]
        var i = 1
        var foundClosingDelimiter = false
        while i < lines.count {
            let line = lines[i]
            i += 1
            if line.trimmingCharacters(in: .whitespaces) == "---" {
                foundClosingDelimiter = true
                break
            }
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = String(line[..<colon]).trimmingCharacters(in: .whitespaces)
            let value = String(line[line.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
            if !key.isEmpty { front[key] = value }
        }
        guard foundClosingDelimiter else { return ([:], text) }
        return (front, lines[i...].joined(separator: "\n"))
    }

    static func firstHeading(_ body: String) -> String? {
        for line in body.components(separatedBy: "\n") {
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("# ") { return String(t.dropFirst(2)).trimmingCharacters(in: .whitespaces) }
        }
        return nil
    }

    static func fileStem(_ path: String) -> String {
        (((path as NSString).lastPathComponent) as NSString).deletingPathExtension
    }

    /// All `.md` files under `directory`, gathered synchronously (the directory
    /// enumerator's iterator is unavailable from async contexts).
    static func markdownFiles(in directory: URL) -> [URL] {
        guard let root = try? VaultFileReader.canonicalURL(for: directory) else { return [] }
        guard let walker = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: Array(inspectionKeys),
            options: [.skipsHiddenFiles, .skipsPackageDescendants],
            errorHandler: { _, _ in true }
        ) else {
            return []
        }

        var out: [URL] = []
        while let object = walker.nextObject() {
            guard let url = object as? URL, url.pathExtension.lowercased() == "md" else {
                continue
            }

            let values: URLResourceValues
            do {
                values = try url.resourceValues(forKeys: inspectionKeys)
            } catch {
                continue
            }
            guard values.isSymbolicLink != true,
                  values.isRegularFile == true,
                  let fileSize = values.fileSize,
                  fileSize <= VaultFileReader.maximumMarkdownSize else {
                continue
            }

            guard let resolvedURL = try? VaultFileReader.canonicalURL(for: url) else {
                continue
            }
            guard contains(resolvedURL, within: root) else { continue }
            out.append(resolvedURL)
        }
        return out
    }

    private static func contains(_ url: URL, within root: URL) -> Bool {
        let rootPath = root.path
        let prefix = rootPath.hasSuffix("/") ? rootPath : rootPath + "/"
        return url.path.hasPrefix(prefix)
    }
}

public extension IndexStore {
    /// Ingest every `.md` file under `directory` (an Obsidian vault) into the
    /// index. The markdown stays canonical on disk; the stored Object is a search
    /// projection keyed by the vault-relative path. Returns the count ingested.
    ///
    /// Note: file IO runs on the actor here for simplicity; a later pass can read
    /// off-actor and batch the upserts.
    @discardableResult
    func ingestVault(at directory: URL) async throws -> Int {
        let reader = try VaultFileReader(rootURL: directory)
        var count = 0
        for url in Vault.markdownFiles(in: reader.rootURL) {
            let rel = (try? reader.relativePath(for: url)) ?? url.lastPathComponent
            let text: String
            do {
                text = try reader.readMarkdown(at: url)
            } catch {
                try recordFailure(
                    FailureSnapshot(
                        id: EngineID(rawValue: UUID().uuidString),
                        category: .extractionFailure,
                        message: "Could not ingest vault file \(rel)",
                        detail: String(describing: error),
                        documentID: EngineID(rawValue: rel),
                        recoverability: .needsUserAction,
                        occurredAt: Date.now
                    )
                )
                continue
            }

            do {
                try await upsert(Vault.parse(markdown: text, relativePath: rel))
                count += 1
            } catch {
                try recordFailure(
                    FailureSnapshot(
                        id: EngineID(rawValue: UUID().uuidString),
                        category: FailureSnapshot.Category(error),
                        message: "Could not ingest vault file \(rel)",
                        detail: String(describing: error),
                        documentID: EngineID(rawValue: rel),
                        recoverability: IndexEngineError.Recoverability(error),
                        occurredAt: Date.now
                    )
                )
            }
        }
        return count
    }
}
