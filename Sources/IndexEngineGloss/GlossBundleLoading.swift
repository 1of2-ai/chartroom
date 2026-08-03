import CryptoKit
import Foundation
import Glossematics

/// A loaded Glossematics model bundle plus the engine-facing identity of its vector space.
///
/// `GlossModelBundle` deliberately carries no space identity — the SDK's job ends at producing
/// vectors. The engine's job starts at never mixing two models' vectors in one index, so the
/// adapter derives the `embeddingSpaceID` here, from the manifest that describes the weights.
public struct LoadedGlossBundle: Sendable {
    public let bundle: GlossModelBundle
    /// Stable identity for one semantic vector contract; equal IDs authorize reuse of
    /// persisted vectors (see `Embedder.embeddingSpaceID`).
    public let embeddingSpaceID: String
}

/// Loads a model bundle directory for the engine, accepting both manifest generations.
///
/// Glossematics reads only manifest v2. The shipped `JinaV5OmniSmall.bundle` — and every copy
/// already staged on user machines — carries a v1 manifest, so this loader migrates v1 in
/// memory instead of demanding the artifact be regenerated. The migration constants are the
/// jina-v5-omni-small identity that v1 kept hardcoded in the runtime and v2 moved into the
/// manifest; they are valid for that model only, and the loader refuses any other v1 modelID
/// rather than guessing.
///
/// **Space-ID continuity.** The v1 adapter fingerprinted the *decoded v1 manifest* (sorted-keys
/// JSON, SHA-256, 12 hex chars) into `modelID:dimension:manifest-<fingerprint>`. A migrated
/// bundle keeps that exact derivation — same bytes fingerprinted, same ID — so indexes built
/// before this migration keep their vectors. A native v2 manifest fingerprints its own decoded
/// form; a bundle whose manifest genuinely changed gets a new space, which is the point.
public enum GlossBundleLoader {
    /// The one v1 model this loader knows how to migrate.
    static let v1MigratableModelID = "jinaai/jina-embeddings-v5-omni-small"

    public static func load(url: URL) throws -> LoadedGlossBundle {
        let manifestURL = url.appendingPathComponent(GlossModelBundle.defaultManifestFilename)
        let data: Data
        do {
            data = try Data(contentsOf: manifestURL)
        } catch {
            throw GlossBundleLoaderError.missingManifest(url: manifestURL)
        }

        switch try JSONDecoder().decode(FormatVersionProbe.self, from: data).formatVersion {
        case GlossModelBundle.Manifest.currentFormatVersion:
            let bundle = try GlossModelBundle(url: url)
            return LoadedGlossBundle(
                bundle: bundle,
                embeddingSpaceID: embeddingSpaceID(
                    modelID: bundle.manifest.modelID,
                    dimension: bundle.manifest.embeddingDimension,
                    fingerprintOf: bundle.manifest
                )
            )
        case 1:
            let v1 = try JSONDecoder().decode(V1Manifest.self, from: data)
            guard v1.modelID == v1MigratableModelID else {
                throw GlossBundleLoaderError.unmigratableV1Model(modelID: v1.modelID)
            }
            let bundle = GlossModelBundle(rootDirectory: url, manifest: migrate(v1))
            return LoadedGlossBundle(
                bundle: bundle,
                // Fingerprint the v1 manifest, not the migrated one: the weights did not
                // change, so the space must not either.
                embeddingSpaceID: embeddingSpaceID(
                    modelID: v1.modelID,
                    dimension: v1.embeddingDimension,
                    fingerprintOf: v1
                )
            )
        case let other:
            throw GlossModelBundleError.unsupportedFormatVersion(
                expected: GlossModelBundle.Manifest.currentFormatVersion, actual: other
            )
        }
    }

    // MARK: - v1 → v2 migration

    /// Fill the v2 fields v1 kept in code, from the jina-v5-omni-small contract
    /// (`GlossematicsSDK/models/jina-v5-omni-small/model.yaml` and the SDK's `MediaTokens`
    /// presets, which the smoke tests pin against the model's tokenizer).
    static func migrate(_ v1: V1Manifest) -> GlossModelBundle.Manifest {
        GlossModelBundle.Manifest(
            modelID: v1.modelID,
            embeddingDimension: v1.embeddingDimension,
            matryoshkaDimensions: [32, 64, 128, 256, 512, 1024],
            minimumDeployment: .init(
                macOS: v1.minimumDeployment.macOS, iOS: v1.minimumDeployment.iOS),
            prompts: .init(query: "Query: ", document: "Document: "),
            tokens: .init(
                padID: 151643,  // <|endoftext|>
                image: mediaTokenIDs(.jinaV5OmniSmallImage),
                video: mediaTokenIDs(.jinaV5OmniSmallVideo),
                audio: mediaTokenIDs(.jinaV5OmniSmallAudio)
            ),
            text: .init(
                model: v1.text.model,
                tokenizer: v1.text.tokenizer,
                buckets: v1.text.buckets,
                requiresAttentionMask: false  // Qwen3 causal tower
            ),
            image: .init(
                encoder: v1.image.encoder,
                resources: v1.image.resources,
                patchBuckets: v1.image.patchBuckets,
                preprocess: .init(patch: 16, merge: 2, minPixels: 65536, maxPixels: 16_777_216)
            ),
            audio: .init(encoder: v1.audio.encoder, frameBuckets: v1.audio.frameBuckets),
            video: .init(encoder: v1.video.encoder, patchBuckets: v1.video.patchBuckets),
            decoder: .init(
                embed: v1.decoder.embed,
                model: v1.decoder.model,
                sequenceBuckets: v1.decoder.sequenceBuckets
            )
        )
    }

    private static func mediaTokenIDs(_ tokens: MediaTokens) -> GlossModelBundle.MediaTokenIDs {
        .init(prefixIDs: tokens.prefix, suffixIDs: tokens.suffix, placeholderID: tokens.placeholder)
    }

    // MARK: - Space identity

    /// `modelID:dimension:manifest-<sha256-prefix>` over the decoded manifest with sorted keys —
    /// byte-for-byte the derivation the v1 adapter used, so existing indexes keep their spaces.
    /// Formatting on disk is not identity; the decoded value is.
    static func embeddingSpaceID(modelID: String, dimension: Int, fingerprintOf manifest: some Encodable) -> String {
        "\(modelID):\(dimension):manifest-\(fingerprint(of: manifest))"
    }

    private static func fingerprint(of manifest: some Encodable) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(manifest) else {
            // Unreachable for a plain Codable struct; a sentinel beats hashing empty
            // Data, which would masquerade as a legitimate fingerprint.
            return "unencodable"
        }
        let digest = SHA256.hash(data: data)
        return String(digest.map { String(format: "%02x", $0) }.joined().prefix(12))
    }

    private struct FormatVersionProbe: Decodable {
        let formatVersion: Int
    }

    /// The v1 manifest shape, mirrored field-for-field from the retired `JinaModelBundle.Manifest`
    /// so the fingerprint of a decoded v1 manifest is byte-identical to what that adapter computed.
    /// Do not add, remove, or rename fields: any change here silently re-keys every v1 bundle's
    /// embedding space.
    struct V1Manifest: Codable, Equatable {
        var formatVersion: Int
        var modelID: String
        var embeddingDimension: Int
        var minimumDeployment: MinimumDeployment
        var text: TextArtifacts
        var image: ImageArtifacts
        var audio: AudioArtifacts
        var video: VideoArtifacts
        var decoder: DecoderArtifacts

        struct MinimumDeployment: Codable, Equatable {
            var macOS: String
            var iOS: String
        }

        struct TextArtifacts: Codable, Equatable {
            var model: String
            var tokenizer: String
            var buckets: [Int]
        }

        struct ImageArtifacts: Codable, Equatable {
            var encoder: String
            var resources: String
            var patchBuckets: [Int]
        }

        struct AudioArtifacts: Codable, Equatable {
            var encoder: String
            var frameBuckets: [Int]
        }

        struct VideoArtifacts: Codable, Equatable {
            var encoder: String
            var patchBuckets: [Int]
        }

        struct DecoderArtifacts: Codable, Equatable {
            var embed: String
            var model: String
            var sequenceBuckets: [Int]
        }
    }
}

public enum GlossBundleLoaderError: Error, CustomStringConvertible {
    case missingManifest(url: URL)
    /// A v1 manifest names a model this loader has no migration constants for. v1 kept the
    /// model's identity in runtime code, so migrating an unknown v1 model would mean inventing
    /// its prompts and token ids.
    case unmigratableV1Model(modelID: String)

    public var description: String {
        switch self {
        case let .missingManifest(url):
            return "GlossBundleLoader: no manifest.json at \(url.path). Point at a converted model bundle."
        case let .unmigratableV1Model(modelID):
            return "GlossBundleLoader: cannot migrate v1 manifest for \(modelID); only "
                + "\(GlossBundleLoader.v1MigratableModelID) has known migration constants. "
                + "Regenerate the bundle with a v2 manifest."
        }
    }
}
