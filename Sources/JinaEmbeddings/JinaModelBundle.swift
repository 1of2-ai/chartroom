import Foundation
import CryptoKit

/// Versioned Core ML artifact bundle consumed by the Swift inference API.
///
/// Conversion tools produce a directory with `manifest.json` at its root. Inference code reads that
/// manifest and resolves artifact paths from it, instead of depending on converter script layout.
public struct JinaModelBundle: Sendable {
    public static let defaultManifestFilename = "manifest.json"

    public let rootDirectory: URL
    public let manifest: Manifest
    public let manifestFingerprint: String

    /// Create a bundle from an already-known manifest. This is also used by the legacy directory path,
    /// where the package assumes the v1 default filenames without requiring `manifest.json`.
    public init(rootDirectory: URL, manifest: Manifest = .default, manifestFingerprint: String? = nil) {
        self.rootDirectory = rootDirectory
        self.manifest = manifest
        self.manifestFingerprint = manifestFingerprint ?? Self.fingerprint(for: manifest)
    }

    /// Read and validate `manifest.json` from a converted model bundle directory.
    public init(url: URL, manifestFilename: String = Self.defaultManifestFilename) throws {
        let manifestURL = url.appendingPathComponent(manifestFilename)
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
        guard manifest.formatVersion == Manifest.currentFormatVersion else {
            throw JinaModelBundleError.unsupportedFormatVersion(
                expected: Manifest.currentFormatVersion,
                actual: manifest.formatVersion
            )
        }
        // Fingerprint the decoded manifest, not the file's bytes: formatting is not
        // identity, and the manifest-constructed initializers must land in the same
        // embedding space as the file-loaded one.
        self.init(rootDirectory: url, manifest: manifest, manifestFingerprint: Self.fingerprint(for: manifest))
    }

    /// Compatibility constructor for directories that use the default artifact names but do not yet
    /// contain a manifest.
    public static func legacy(directory: URL) -> JinaModelBundle {
        JinaModelBundle(rootDirectory: directory, manifest: .default)
    }

    /// Resolve a manifest path. Relative paths are interpreted inside `rootDirectory`; absolute paths
    /// are preserved so advanced callers can keep artifacts outside one directory.
    public func resolve(_ path: String) -> URL {
        Self.resolve(rootDirectory: rootDirectory, path: path)
    }

    public static func resolve(rootDirectory: URL, path: String) -> URL {
        path.hasPrefix("/") ? URL(fileURLWithPath: path) : rootDirectory.appendingPathComponent(path)
    }

    public var embeddingSpaceID: String {
        "\(manifest.modelID):\(manifest.embeddingDimension):manifest-\(manifestFingerprint)"
    }

    private static func fingerprint(for manifest: Manifest) -> String {
        // Sorted keys make the encode deterministic across processes; a plain
        // JSONEncoder's key order is not, which turned the space ID into a coin flip.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(manifest) else {
            // Unreachable for a plain Codable struct; a sentinel beats hashing empty
            // Data, which would masquerade as a legitimate fingerprint.
            return "unencodable"
        }
        return fingerprint(for: data)
    }

    private static func fingerprint(for data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return String(digest.map { String(format: "%02x", $0) }.joined().prefix(12))
    }

    public struct Manifest: Codable, Equatable, Sendable {
        public static let currentFormatVersion = 1
        public static let `default` = Manifest()

        public var formatVersion: Int
        public var modelID: String
        public var embeddingDimension: Int
        public var minimumDeployment: MinimumDeployment
        public var text: TextArtifacts
        public var image: ImageArtifacts
        public var audio: AudioArtifacts
        public var video: VideoArtifacts
        public var decoder: DecoderArtifacts

        public init(
            formatVersion: Int = currentFormatVersion,
            modelID: String = "jinaai/jina-embeddings-v5-omni-small",
            embeddingDimension: Int = 1024,
            minimumDeployment: MinimumDeployment = .default,
            text: TextArtifacts = .default,
            image: ImageArtifacts = .default,
            audio: AudioArtifacts = .default,
            video: VideoArtifacts = .default,
            decoder: DecoderArtifacts = .default
        ) {
            self.formatVersion = formatVersion
            self.modelID = modelID
            self.embeddingDimension = embeddingDimension
            self.minimumDeployment = minimumDeployment
            self.text = text
            self.image = image
            self.audio = audio
            self.video = video
            self.decoder = decoder
        }
    }

    public struct MinimumDeployment: Codable, Equatable, Sendable {
        public static let `default` = MinimumDeployment()

        public var macOS: String
        public var iOS: String

        public init(macOS: String = "15.0", iOS: String = "18.0") {
            self.macOS = macOS
            self.iOS = iOS
        }
    }

    public struct TextArtifacts: Codable, Equatable, Sendable {
        public static let `default` = TextArtifacts()

        public var model: String
        public var tokenizer: String
        public var buckets: [Int]

        public init(
            model: String = "text_multifunc.mlpackage",
            tokenizer: String = "jina-v5-omni-small",
            buckets: [Int] = [32, 64, 128, 256, 512]
        ) {
            self.model = model
            self.tokenizer = tokenizer
            self.buckets = buckets
        }
    }

    public struct ImageArtifacts: Codable, Equatable, Sendable {
        public static let `default` = ImageArtifacts()

        public var encoder: String
        public var resources: String
        public var patchBuckets: [Int]

        public init(
            encoder: String = "vision_tower_masked_multifunc.mlpackage",
            resources: String = "vision_swift",
            patchBuckets: [Int] = [1024, 1600, 2304, 3072, 4032]
        ) {
            self.encoder = encoder
            self.resources = resources
            self.patchBuckets = patchBuckets
        }
    }

    public struct AudioArtifacts: Codable, Equatable, Sendable {
        public static let `default` = AudioArtifacts()

        public var encoder: String
        public var frameBuckets: [Int]

        public init(
            encoder: String = "audio_tower_masked_multifunc.mlpackage",
            frameBuckets: [Int] = [200, 400, 800, 1600, 3200]
        ) {
            self.encoder = encoder
            self.frameBuckets = frameBuckets
        }
    }

    public struct VideoArtifacts: Codable, Equatable, Sendable {
        public static let `default` = VideoArtifacts()

        public var encoder: String
        public var patchBuckets: [Int]

        public init(
            encoder: String = "vision_tower_video_multifunc.mlpackage",
            patchBuckets: [Int] = [256, 512, 1024, 2048]
        ) {
            self.encoder = encoder
            self.patchBuckets = patchBuckets
        }
    }

    public struct DecoderArtifacts: Codable, Equatable, Sendable {
        public static let `default` = DecoderArtifacts()

        public var embed: String
        public var model: String
        public var sequenceBuckets: [Int]

        public init(
            embed: String = "embed_multifunc.mlpackage",
            model: String = "decoder_embeds_multifunc.mlpackage",
            sequenceBuckets: [Int] = [128, 256, 512, 1024]
        ) {
            self.embed = embed
            self.model = model
            self.sequenceBuckets = sequenceBuckets
        }
    }
}

public enum JinaModelBundleError: Error, CustomStringConvertible {
    case unsupportedFormatVersion(expected: Int, actual: Int)

    public var description: String {
        switch self {
        case let .unsupportedFormatVersion(expected, actual):
            return "JinaModelBundle: unsupported manifest formatVersion \(actual). Expected \(expected)."
        }
    }
}
