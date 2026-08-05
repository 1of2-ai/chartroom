import Foundation

/// Per-model weak-similarity thresholds for Gloss-backed embedders.
///
/// The threshold is a property of a model's similarity distribution, so it can only be
/// *measured*, never assumed — `GlossSimilarity.weakThreshold` documents that measurement
/// for jina-v5-omni-small. Models without a measured value get `conservativeDefault`:
/// deliberately low, because a threshold that is too high silently discards genuine
/// matches while one that is too low merely lets tail into the weak section, which is a
/// visible failure the UI already handles.
public enum GlossWeakSimilarityThreshold {
    /// Matches `Embedder`'s documented near-orthogonal floor. Used until a model has a
    /// calibration entry below.
    public static let conservativeDefault: Float = 0.15

    /// Measured values, keyed by manifest `modelID`. Extending this table (with the
    /// measurement, in `GlossSimilarity`-style rationale) is part of onboarding a model.
    static let calibrated: [String: Float] = [
        "jinaai/jina-embeddings-v5-omni-small": GlossSimilarity.weakThreshold,
    ]

    public static func threshold(forModelID modelID: String, override: Float? = nil) -> Float {
        override ?? calibrated[modelID] ?? conservativeDefault
    }
}
