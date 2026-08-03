import Foundation

/// Similarity calibration for the jina-embeddings-v5-omni-small space.
///
/// Both providers embed text through the same tower into the same 1024-d space, so the
/// distribution — and therefore the threshold — is a property of the model, not of the provider
/// that wraps it. One constant is what stops the two drifting apart.
public enum GlossSimilarity {
    /// Cosine below which a vector-only hit is tail rather than an answer.
    ///
    /// Measured on `jinaai/jina-embeddings-v5-omni-small` at 1024-d, query/document prompts,
    /// with `Query: ` / `Document: ` prefixes applied as in normal retrieval:
    ///
    /// | Pair kind                   | n  | min    | median | max   |
    /// | --------------------------- | -- | ------ | ------ | ----- |
    /// | Topically unrelated         | 12 | −0.014 | 0.034  | 0.161 |
    /// | Paraphrase / restatement    |  6 | 0.532  | 0.694  | 0.801 |
    ///
    /// The two populations separate cleanly, so the threshold's only job is clearing the noise
    /// ceiling. It sits above the observed unrelated maximum with roughly 55% headroom and far
    /// below the weakest true match, deliberately biased low: the partition exists to tell
    /// "nothing matched" from "here are answers", and a document that is genuinely relevant
    /// without being a paraphrase lands between these populations. Cutting those would trade a
    /// visible failure for a silent one.
    ///
    /// Small samples, so this is a calibration point rather than a tuned value.
    /// `GlossSimilarityCalibrationTests` re-derives both populations against the live model and
    /// fails if this constant stops separating them.
    public static let weakThreshold: Float = 0.25
}
