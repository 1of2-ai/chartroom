import Foundation
import JinaEmbeddings

/// Resolves the on-disk location of the `JinaV5OmniSmall.bundle` Core ML artifact.
///
/// The bundle is large (multi-gigabyte) and distributed as release assets rather than
/// through the package checkout (`.lfsconfig` excludes it from LFS smudge, so SwiftPM
/// checkouts contain pointer files only). Callers use `locate` to discover a staged copy
/// and fall back to the mock embedder when it is absent (see `IndexEngineJina.resolveEmbedder`).
///
/// Resolution order, first valid wins:
/// 1. `JINA_MODEL_BUNDLE` environment override (absolute path to the bundle).
/// 2. A `JinaV5OmniSmall.bundle` resource copied into the host app bundle.
/// 3. Any caller-supplied `additionalCandidates` — the host supplies its own locations,
///    e.g. an Application Support install directory or a repo-local development path.
/// 4. The `IndexEngineJina` SwiftPM package resource.
public enum JinaModelBundleLocator {
    public static let bundleName = "JinaV5OmniSmall.bundle"
    public static let environmentKey = "JINA_MODEL_BUNDLE"

    public static func locate(additionalCandidates: [URL] = []) -> URL? {
        candidates(additionalCandidates: additionalCandidates).first(where: isValidBundle)
    }

    static func candidates(additionalCandidates: [URL]) -> [URL] {
        var candidates: [URL] = []

        if let override = ProcessInfo.processInfo.environment[environmentKey], !override.isEmpty {
            candidates.append(URL(fileURLWithPath: override))
        }

        if let resource = Bundle.main.url(forResource: "JinaV5OmniSmall", withExtension: "bundle") {
            candidates.append(resource)
        }

        candidates.append(contentsOf: additionalCandidates)

        if let packageResource = Bundle.module.url(forResource: "JinaV5OmniSmall", withExtension: "bundle") {
            candidates.append(packageResource)
        }
        return candidates
    }

    /// A bundle is usable only if its manifest decodes and every artifact path the
    /// omni provider can load is present. A manifest-only directory must not be
    /// considered model-backed because the first embed would fail later.
    public static func isValidBundle(_ url: URL) -> Bool {
        do {
            let bundle = try JinaModelBundle(url: url)
            return requiredArtifactURLs(for: bundle).allSatisfy { artifactURL in
                FileManager.default.fileExists(atPath: artifactURL.path)
            }
        } catch {
            return false
        }
    }

    private static func requiredArtifactURLs(for bundle: JinaModelBundle) -> [URL] {
        let manifest = bundle.manifest
        let tokenizerFolder = bundle.resolve(manifest.text.tokenizer)
        let visionResources = bundle.resolve(manifest.image.resources)
        return [
            bundle.resolve(manifest.text.model),
            tokenizerFolder.appendingPathComponent("tokenizer.json"),
            tokenizerFolder.appendingPathComponent("tokenizer_config.json"),
            bundle.resolve(manifest.image.encoder),
            // The files the image/video towers load by name, not just their folder —
            // a folder that exists but is missing one would validate as model-backed
            // and then fail on the first image embed, far from the misconfiguration.
            visionResources.appendingPathComponent("meta.json"),
            visionResources.appendingPathComponent("pos_embed_table.f32"),
            visionResources.appendingPathComponent("rope_inv_freq.f32"),
            bundle.resolve(manifest.audio.encoder),
            bundle.resolve(manifest.video.encoder),
            bundle.resolve(manifest.decoder.embed),
            bundle.resolve(manifest.decoder.model)
        ]
    }
}
