import ChartroomTestSupport
import Foundation
import Testing
@testable import IndexEngine

/// Image ingestion was covered only by the Core ML end-to-end test, which takes minutes and is
/// skipped entirely in checkouts without the model payloads. A stub embedder exercises the same
/// branch — extract, single chunk, image-modality vector — in milliseconds and on every machine.
@Suite("Image ingestion")
struct ImageIngestTests {
    private struct ImageCapableEmbedder: FixtureEmbedder {
        let modelID = "image-stub-v1"
        let dimension = 8

        func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
            var vector = [Float](repeating: 0, count: dimension)
            vector[0] = 1
            return vector
        }

        var supportsImageEmbedding: Bool { true }

        func embedImage(at url: URL) async throws -> [Float] {
            var vector = [Float](repeating: 0, count: dimension)
            vector[1] = 1
            return vector
        }
    }

    private func makeImageFile() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "image-ingest-\(UUID().uuidString).png")
        FileManager.default.createFile(atPath: url.path, contents: Data([0x89, 0x50, 0x4E, 0x47]))
        return url
    }

    @Test("an image document is stored as one chunk with one embedding")
    func imageProducesChunkAndEmbedding() async throws {
        let store = try IndexStore(path: ":memory:", embedder: ImageCapableEmbedder())
        let imageURL = makeImageFile()
        defer { try? FileManager.default.removeItem(at: imageURL) }

        try await store.upsert(
            IndexedObject(id: "img-1", type: "public.png", title: "blue.png", body: "", sourceURI: imageURL)
        )

        let counts = try await store.counts()
        #expect(counts.documentCount == 1)
        #expect(counts.chunkCount == 1)
        #expect(counts.embeddingCount == 1)
    }

    @Test("an image ingested through the facade reports acceptance, not a silent zero")
    func imageIngestThroughFacadeSucceeds() async throws {
        let engine = try await IndexEngine.openInMemory(
            configuration: IndexEngineConfiguration(embedder: ImageCapableEmbedder())
        )
        let imageURL = makeImageFile()
        defer { try? FileManager.default.removeItem(at: imageURL) }

        let summary = try await engine.ingest(IngestRequest(payloads: [
            SourcePayload(
                documentID: "img-1",
                sourceURI: imageURL,
                displayName: "blue.png",
                contentType: "public.png",
                body: .binaryReference(imageURL)
            )
        ]))

        #expect(
            summary.failedCount == 0,
            "ingest failed: \(summary.failures.map { "\($0.category): \($0.message)" })"
        )
        #expect(summary.acceptedCount == 1)
        #expect(await engine.snapshot().embeddingCount == 1)
    }

    /// An image chunk's vector comes from pixels, so a rebuild cannot regenerate it from stored
    /// text. It has to be reported as skipped rather than quietly re-embedded as text.
    @Test("a rebuild skips image chunks instead of re-embedding them as text")
    func rebuildSkipsImageChunks() async throws {
        let path = FileManager.default.temporaryDirectory
            .appending(path: "image-rebuild-\(UUID().uuidString).sqlite").path
        defer { try? FileManager.default.removeItem(atPath: path) }

        let original = try IndexStore(path: path, embedder: ImageCapableEmbedder())
        let imageURL = makeImageFile()
        defer { try? FileManager.default.removeItem(at: imageURL) }
        try await original.upsert(
            IndexedObject(id: "img-1", type: "public.png", title: "blue.png", body: "", sourceURI: imageURL)
        )

        struct SwappedEmbedder: FixtureEmbedder {
            let modelID = "image-stub-v2"
            let dimension = 8
            func embed(_ text: String, kind: EmbedKind) async throws -> [Float] {
                var vector = [Float](repeating: 0, count: dimension)
                vector[2] = 1
                return vector
            }
            var supportsImageEmbedding: Bool { true }
            func embedImage(at url: URL) async throws -> [Float] {
                var vector = [Float](repeating: 0, count: dimension)
                vector[3] = 1
                return vector
            }
        }

        let swapped = try IndexStore(path: path, embedder: SwappedEmbedder())
        let summary = try await swapped.rebuildActiveEmbeddingSpace()

        #expect(summary.rebuiltChunkCount == 0)
        #expect(summary.skippedNonTextChunkCount == 1)
        #expect(summary.isComplete == false, "the caller must be told a re-ingest is still needed")
    }
}
