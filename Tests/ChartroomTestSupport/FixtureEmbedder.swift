import Foundation
import IndexEngine

/// Conformance shorthand for the stub embedders tests use to drive the engine.
///
/// `Embedder` deliberately gives `isModelBacked` and `weakSimilarityThreshold` no defaults: a real
/// model inheriting either one is the bug that removing them fixed. Test fixtures are the one
/// population where a shared answer *is* correct — none of them is a model, and none of them is
/// exercising the weak/strong partition — so they get the defaults here, scoped to test support,
/// instead of on the product protocol where a shipping embedder could pick them up.
public protocol FixtureEmbedder: Embedder {}

public extension FixtureEmbedder {
    var isModelBacked: Bool { false }

    /// Matches the threshold these fixtures were written against, so adopting this protocol
    /// changes no test's expectations. Fixtures that exercise weak results state their own.
    var weakSimilarityThreshold: Float { 0.15 }
}
