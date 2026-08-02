import Testing
import Glossematics

/// GlossematicsSDK 0.2.0 (vendored as a submodule) smoke test: proves the dependency
/// resolves, compiles, and exposes its public API without requiring a model bundle.
@Suite("GlossematicsSDK")
struct GlossematicsSDKSmokeTests {
    @Test("media token presets match the jina-v5-omni-small contract")
    func mediaTokenPresets() {
        #expect(MediaTokens.jinaV5OmniSmallImage.placeholder == 151655)
        #expect(MediaTokens.jinaV5OmniSmallVideo.placeholder == 151656)
        #expect(MediaTokens.jinaV5OmniSmallAudio.placeholder == 151669)
        #expect(MediaTokens.jinaV5OmniSmallImage.prefix == [151644, 872, 198, 151652])
    }
}
