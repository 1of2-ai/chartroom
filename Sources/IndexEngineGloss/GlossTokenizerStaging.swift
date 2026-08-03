import Foundation
import Glossematics

/// Resolves a tokenizer folder that `swift-transformers` can load.
///
/// `AutoTokenizer.from(modelFolder:)` requires a `config.json` to *exist* even though it
/// builds the BPE tokenizer purely from `tokenizer.json` + `tokenizer_config.json`. The
/// model bundle ships only those two files. Rather than mutate the (git-ignored,
/// externally managed) bundle, this stages a sibling folder that symlinks the real
/// tokenizer files and adds a minimal `config.json`. Shared by every Gloss-backed provider.
enum GlossTokenizerStagingError: Error, Equatable {
    case missingTokenizerArtifact(path: String)
}

enum GlossTokenizerStaging {
    private static let lock = NSLock()

    /// The folder to hand to a Gloss text embedder: the bundle's own tokenizer folder when it
    /// already has `config.json`, otherwise a staged folder that adds one.
    static func resolvedFolder(for bundle: GlossModelBundle) throws -> URL {
        let original = bundle.resolve(bundle.manifest.text.tokenizer)
        let fm = FileManager.default
        if fm.fileExists(atPath: original.appendingPathComponent("config.json").path) {
            return original
        }

        // Caches, not temporaryDirectory: the tmp cleaner can purge a staged folder
        // between provider init and the tokenizer's lazy first load.
        let cachesRoot = (try? fm.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? fm.temporaryDirectory
        let staged = cachesRoot
            .appendingPathComponent("IndexEngineGloss-tokenizer", isDirectory: true)
            .appendingPathComponent(stableKey(for: original.path), isDirectory: true)

        // Concurrent loaders of the same bundle would otherwise race on the shared staged
        // directory; serialize and make the work idempotent.
        lock.lock()
        defer { lock.unlock() }

        let names = ["tokenizer.json", "tokenizer_config.json", "config.json"]
        if names.allSatisfy({ fm.fileExists(atPath: staged.appendingPathComponent($0).path) }) {
            return staged
        }

        try fm.createDirectory(at: staged, withIntermediateDirectories: true)
        for file in ["tokenizer.json", "tokenizer_config.json"] {
            let target = original.appendingPathComponent(file)
            // `createSymbolicLink` happily creates dangling links; a missing original
            // must fail here, not later as an opaque AutoTokenizer load error.
            guard fm.fileExists(atPath: target.path) else {
                throw GlossTokenizerStagingError.missingTokenizerArtifact(path: target.path)
            }
            let link = staged.appendingPathComponent(file)
            try? fm.removeItem(at: link)
            try fm.createSymbolicLink(at: link, withDestinationURL: target)
        }
        try Data(#"{"model_type":"qwen2"}"#.utf8)
            .write(to: staged.appendingPathComponent("config.json"))
        return staged
    }

    /// Compact, deterministic directory name from the source path so distinct bundles stage
    /// to distinct folders and re-runs reuse the same one. FNV-1a (Swift's `Hasher` is
    /// per-process seeded and unsuitable for a stable on-disk name).
    private static func stableKey(for path: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in path.utf8 { hash = (hash ^ UInt64(byte)) &* 0x100000001b3 }
        return String(hash, radix: 16)
    }
}
