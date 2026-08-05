import Foundation
import Glossematics

/// Resolves the on-disk location of a Gloss model bundle.
///
/// Bundles are large (multi-gigabyte) and distributed as release assets rather than
/// through the package checkout (`.lfsconfig` excludes them from LFS smudge, so SwiftPM
/// checkouts contain pointer files only). Callers use `locate` to discover a staged copy
/// and fall back to the mock embedder when it is absent (see `IndexEngineGloss.resolveEmbedder`).
///
/// Resolution order, first valid wins:
/// 1. The profile's environment override (absolute path to the bundle).
/// 2. A bundle resource copied into the host app bundle, under the profile's name.
/// 3. Any caller-supplied `additionalCandidates` — the host supplies its own locations,
///    e.g. an Application Support install directory or a repo-local development path.
/// 4. The `IndexEngineGloss` SwiftPM package resource.
public enum GlossModelBundleLocator {
    /// The per-model constants of bundle discovery. Validation is manifest-driven and needs no
    /// profile — any directory that holds a complete Gloss bundle passes `isValidBundle` — so a
    /// profile exists only where a name must be known *before* anything is on disk: the
    /// environment key deployments export and the resource name hosts embed.
    public struct Profile: Sendable {
        public let bundleName: String
        public let environmentKey: String

        /// `bundleName` without its `.bundle` extension, as `Bundle.url(forResource:withExtension:)`
        /// wants it.
        var resourceName: String {
            (bundleName as NSString).deletingPathExtension
        }

        public init(bundleName: String, environmentKey: String) {
            self.bundleName = bundleName
            self.environmentKey = environmentKey
        }

        /// The shipped default. Both names predate the Glossematics migration and stay,
        /// because deployments already set the key and embed the resource.
        public static let jinaV5OmniSmall = Profile(
            bundleName: "JinaV5OmniSmall.bundle",
            environmentKey: "JINA_MODEL_BUNDLE"
        )
    }

    public static func locate(
        profile: Profile = .jinaV5OmniSmall,
        additionalCandidates: [URL] = []
    ) -> URL? {
        candidates(profile: profile, additionalCandidates: additionalCandidates)
            .first(where: isValidBundle)
    }

    static func candidates(profile: Profile, additionalCandidates: [URL]) -> [URL] {
        var candidates: [URL] = []

        if let override = ProcessInfo.processInfo.environment[profile.environmentKey],
           !override.isEmpty {
            candidates.append(URL(fileURLWithPath: override))
        }

        if let resource = Bundle.main.url(forResource: profile.resourceName, withExtension: "bundle") {
            candidates.append(resource)
        }

        candidates.append(contentsOf: additionalCandidates)

        if let packageResource = Bundle.module.url(forResource: profile.resourceName, withExtension: "bundle") {
            candidates.append(packageResource)
        }
        return candidates
    }

    /// A bundle is usable only if its manifest decodes (v1 manifests migrate in memory) and
    /// every artifact its manifest declares is present. A manifest-only directory must not be
    /// considered model-backed because the first embed would fail later.
    public static func isValidBundle(_ url: URL) -> Bool {
        do {
            let loaded = try GlossBundleLoader.load(url: url)
            return requiredArtifactURLs(for: loaded.bundle).allSatisfy { artifactURL in
                FileManager.default.fileExists(atPath: artifactURL.path)
            }
        } catch {
            return false
        }
    }

    /// Everything the manifest's present tower sections load. v2 tower sections are optional
    /// (a text-only bundle is valid); a section that is present must be complete.
    private static func requiredArtifactURLs(for bundle: GlossModelBundle) -> [URL] {
        let manifest = bundle.manifest
        let tokenizerFolder = bundle.resolve(manifest.text.tokenizer)
        var urls = [
            bundle.resolve(manifest.text.model),
            tokenizerFolder.appendingPathComponent("tokenizer.json"),
            tokenizerFolder.appendingPathComponent("tokenizer_config.json"),
        ]
        if let image = manifest.image {
            urls.append(bundle.resolve(image.encoder))
            // The files the image/video towers load by name, not just their folder —
            // a folder that exists but is missing one would validate as model-backed
            // and then fail on the first image embed, far from the misconfiguration.
            let visionResources = bundle.resolve(image.resources)
            urls.append(visionResources.appendingPathComponent("meta.json"))
            urls.append(visionResources.appendingPathComponent("pos_embed_table.f32"))
            urls.append(visionResources.appendingPathComponent("rope_inv_freq.f32"))
        }
        if let audio = manifest.audio {
            urls.append(bundle.resolve(audio.encoder))
        }
        if let video = manifest.video {
            urls.append(bundle.resolve(video.encoder))
        }
        if let decoder = manifest.decoder {
            urls.append(bundle.resolve(decoder.embed))
            urls.append(bundle.resolve(decoder.model))
        }
        return urls
    }
}
