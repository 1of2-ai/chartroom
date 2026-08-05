import ChartroomTestSupport
import Foundation
import Testing
@testable import IndexEngine

/// The shipped MCP server ran the hashing stand-in while reporting "Embedding provider is
/// available" — an index with no semantic channel presenting as a healthy model. These pin the
/// signal that makes that state visible instead of inferable.
@Suite("Model status honesty")
struct ModelStatusHonestyTests {
    private struct ModelBackedFixture: Embedder {
        let modelID = "fixture-model"
        let dimension = 8
        let isModelBacked = true
        let weakSimilarityThreshold: Float = 0.4

        func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
            [Float](repeating: 1 / Float(dimension).squareRoot(), count: dimension)
        }
    }

    @Test("a hashing store reports itself as not model-backed while still being available")
    func hashingStoreIsReportedAsFallback() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 64))
        )
        let status = await engine.modelStatus()

        // Availability and model-backing are separate claims: the stand-in answers probes fine.
        #expect(status.isAvailable)
        #expect(status.isModelBacked == false)
        // A host that renders only the message must still learn the truth from it.
        #expect(status.message.contains("Not model-backed"))
        #expect(status.message.contains("lexical only"))
    }

    @Test("a model-backed store says so without the degradation notice")
    func modelBackedStoreIsReportedPlainly() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: ModelBackedFixture())
        )
        let status = await engine.modelStatus()

        #expect(status.isAvailable)
        #expect(status.isModelBacked)
        #expect(status.message.contains("Not model-backed") == false)
    }

    /// `isModelBacked` has no protocol default, so a stand-in cannot inherit `true` — the compiler
    /// makes every conformance state which it is.
    @Test("the built-in stand-in declares itself a stand-in")
    func hashingEmbedderDeclaresItself() {
        #expect(HashingEmbedder().isModelBacked == false)
        #expect(ModelBackedFixture().isModelBacked)
    }

    private struct ImageCapableFixture: Embedder {
        let modelID = "fixture-omni"
        let dimension = 8
        let isModelBacked = true
        let weakSimilarityThreshold: Float = 0.4
        let supportsImageEmbedding = true

        func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
            [Float](repeating: 1 / Float(dimension).squareRoot(), count: dimension)
        }

        func embedImage(at url: URL) async throws -> [Float] {
            try await embed(url.lastPathComponent, kind: .document)
        }
    }

    /// Text-only models embed an image's *filename*, not its pixels; the status must say which
    /// an index delivers so hosts don't present name-only image search as content search.
    @Test("status reports whether images embed by content")
    func imageCapabilityIsReported() async throws {
        let textOnly = try await IndexEngine.openInMemory(
            configuration: .init(embedder: ModelBackedFixture())
        )
        #expect(await textOnly.modelStatus().supportsImageEmbedding == false)

        let omni = try await IndexEngine.openInMemory(
            configuration: .init(embedder: ImageCapableFixture())
        )
        #expect(await omni.modelStatus().supportsImageEmbedding)
    }
}
