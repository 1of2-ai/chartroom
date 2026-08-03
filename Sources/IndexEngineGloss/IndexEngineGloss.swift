import CoreML
import Foundation
import IndexEngine

/// Where the text tower runs. Measured on Apple silicon, the Neural Engine embeds at a
/// fraction of the GPU's power (≈5 W vs ≈67 W → ~6.7× less energy per embedding) at roughly
/// half the throughput, so `efficiency` is the default and the right choice on battery.
public enum TextComputePreference: String, CaseIterable, Sendable {
    case efficiency  // Neural Engine — lowest energy, cool and quiet
    case speed       // GPU — fastest per item, much higher power

    public var computeUnits: MLComputeUnits {
        switch self {
        case .efficiency: .cpuAndNeuralEngine
        case .speed: .cpuAndGPU
        }
    }

    public var displayName: String {
        switch self {
        case .efficiency: "Efficiency (Neural Engine)"
        case .speed: "Speed (GPU)"
        }
    }
}

public extension TextComputePreference {
    /// The one defaults key for the compute preference, shared by every host storage site
    /// (Settings pickers, engine opening).
    static let defaultsKey = "textComputeMode"

    /// The compute placement an index open should actually use: the stored preference, but
    /// hard overridden to the Neural Engine whenever Low Power Mode is on — the GPU's power
    /// draw is the opposite of what that mode promises.
    static func effective(
        stored: TextComputePreference?,
        isLowPowerModeEnabled: Bool
    ) -> TextComputePreference {
        isLowPowerModeEnabled ? .efficiency : (stored ?? .efficiency)
    }

    /// Convenience over `effective(stored:isLowPowerModeEnabled:)` reading the shared
    /// defaults key and the live Low Power Mode state.
    static func effective(defaults: UserDefaults = .standard) -> TextComputePreference {
        effective(
            stored: defaults.string(forKey: defaultsKey).flatMap(TextComputePreference.init(rawValue:)),
            isLowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled
        )
    }
}

/// Entry point for resolving the engine's embedding provider with a graceful fallback.
///
/// Apps call `resolveEmbedder` at startup and hand the result to
/// `IndexEngineConfiguration`. When the model bundle is present the engine gets
/// real semantic vectors through Glossematics; when it is absent (CI, a fresh checkout,
/// a machine without the multi-gigabyte artifact) the engine transparently falls back to
/// the dependency-free `HashingEmbedder` so the app still opens and search still runs in
/// a degraded-but-honest mode.
public enum IndexEngineGloss {
    /// The embedder chosen for an index, plus enough provenance for the host app to
    /// surface which path is active in its model-status UI.
    public struct ResolvedEmbedder: Sendable {
        public let embedder: any Embedder
        /// True when the real model loaded; false when falling back to the mock.
        public let isModelBacked: Bool
        /// Human-readable detail for diagnostics and model-status display.
        public let detail: String

        public init(embedder: any Embedder, isModelBacked: Bool, detail: String) {
            self.embedder = embedder
            self.isModelBacked = isModelBacked
            self.detail = detail
        }
    }

    /// Resolve the best available embedder. Never throws: a missing bundle or a failed
    /// model load both resolve to the mock embedder with explanatory detail.
    public static func resolveEmbedder(
        additionalCandidates: [URL] = [],
        compute: TextComputePreference = .efficiency
    ) async -> ResolvedEmbedder {
        guard let bundleURL = GlossModelBundleLocator.locate(additionalCandidates: additionalCandidates) else {
            return ResolvedEmbedder(
                embedder: HashingEmbedder(),
                isModelBacked: false,
                detail: "Model bundle not found; using hashing mock embedder."
            )
        }

        do {
            let provider = try await GlossOmniEmbeddingProvider.load(bundleURL: bundleURL, compute: compute)
            let modalities = provider.supportsImageEmbedding ? "text + image" : "text"
            return ResolvedEmbedder(
                embedder: provider,
                isModelBacked: true,
                detail: "Glossematics embeddings (\(provider.modelID), \(provider.dimension)d, \(modalities)) on \(compute.displayName)."
            )
        } catch {
            return ResolvedEmbedder(
                embedder: HashingEmbedder(),
                isModelBacked: false,
                detail: "Model load failed (\(error)); using hashing mock embedder."
            )
        }
    }
}
