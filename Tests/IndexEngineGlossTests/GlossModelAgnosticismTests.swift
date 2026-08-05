import Foundation
import Testing
@testable import IndexEngineGloss
import IndexEngine

/// The per-model seams introduced for the multi-model batch: neither suite needs a real
/// bundle, because both test the *selection* of per-model constants, not inference.
@Suite("Weak-similarity threshold table")
struct GlossWeakSimilarityThresholdTests {
    @Test("calibrated model gets its measured threshold")
    func calibratedModel() {
        #expect(
            GlossWeakSimilarityThreshold.threshold(forModelID: "jinaai/jina-embeddings-v5-omni-small")
                == GlossSimilarity.weakThreshold
        )
    }

    @Test("unknown model falls back to the conservative default")
    func unknownModel() {
        #expect(
            GlossWeakSimilarityThreshold.threshold(forModelID: "jinaai/jina-embeddings-v5-omni-nano")
                == GlossWeakSimilarityThreshold.conservativeDefault
        )
    }

    @Test("explicit override beats both the table and the default")
    func overrideWins() {
        #expect(GlossWeakSimilarityThreshold.threshold(
            forModelID: "jinaai/jina-embeddings-v5-omni-small", override: 0.4) == 0.4)
        #expect(GlossWeakSimilarityThreshold.threshold(
            forModelID: "unknown/model", override: 0.4) == 0.4)
    }

    @Test("conservative default stays below the calibrated value")
    func defaultIsConservative() {
        // The fallback must never be *stricter* than a measured threshold: an uncalibrated
        // model should err toward showing weak matches, not silently discarding them.
        #expect(GlossWeakSimilarityThreshold.conservativeDefault < GlossSimilarity.weakThreshold)
    }
}

@Suite("Bundle locator profiles")
struct GlossModelBundleLocatorProfileTests {
    @Test("default profile preserves the shipped constants")
    func defaultProfile() {
        let profile = GlossModelBundleLocator.Profile.jinaV5OmniSmall
        #expect(profile.bundleName == "JinaV5OmniSmall.bundle")
        #expect(profile.environmentKey == "JINA_MODEL_BUNDLE")
        #expect(profile.resourceName == "JinaV5OmniSmall")
    }

    @Test("candidates honor a custom profile's environment key")
    func customEnvironmentKey() {
        // Distinct keys must not observe each other's overrides: a JINA_MODEL_BUNDLE export
        // pointing at the small bundle must not leak into a nano profile's resolution.
        let key = "GLOSS_TEST_BUNDLE_\(UInt32.random(in: 0..<UInt32.max))"
        let custom = GlossModelBundleLocator.Profile(
            bundleName: "JinaV5OmniNano.bundle", environmentKey: key)
        let withoutOverride = GlossModelBundleLocator.candidates(
            profile: custom, additionalCandidates: [])
        #expect(!withoutOverride.contains { $0.path.hasSuffix("from-env") })

        setenv(key, "/tmp/from-env", 1)
        defer { unsetenv(key) }
        let withOverride = GlossModelBundleLocator.candidates(
            profile: custom, additionalCandidates: [])
        #expect(withOverride.first?.path == "/tmp/from-env")
    }

    @Test("caller candidates rank between host resources and package resources")
    func candidateOrder() {
        let extra = URL(fileURLWithPath: "/tmp/app-support-bundle")
        let candidates = GlossModelBundleLocator.candidates(
            profile: .jinaV5OmniSmall, additionalCandidates: [extra])
        #expect(candidates.contains(extra))
        // The package resource (when staged) is the last resort, after caller candidates.
        if let packageIndex = candidates.lastIndex(where: { $0.path.contains("IndexEngineGloss") }),
           let extraIndex = candidates.firstIndex(of: extra) {
            #expect(extraIndex < packageIndex)
        }
    }
}
