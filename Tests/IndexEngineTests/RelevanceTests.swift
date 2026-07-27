import Foundation
import Testing
@testable import IndexEngine

/// Vector search always returns its top-N by cosine, so a query matching nothing still comes
/// back full. These tests pin the signal that lets a client tell those two cases apart.
@Suite("Retrieval relevance")
struct RelevanceTests {
    private static let corpus = [
        ("doc-atlas", "Atlas Routing", "atlas routing tables converge on the shortest path"),
        ("doc-policy", "Policy Registry", "semantic retrieval with policy registry and vector search"),
        ("doc-ledger", "Ledger Sync", "ledger synchronization reconciles pending transactions"),
        ("doc-mel", "Mel Frontend", "mel spectrogram frontend windows the audio signal"),
    ]

    private func engineWithCorpus() async throws -> IndexEngine {
        let engine = try await IndexEngine.openInMemory(
            configuration: .init(embedder: HashingEmbedder(dimension: 256))
        )
        let payloads = Self.corpus.map { id, title, body in
            SourcePayload(
                documentID: EngineID(rawValue: id),
                sourceID: "local-files",
                sourceURI: URL(filePath: "/tmp/\(id).md"),
                displayName: title,
                contentType: "net.daringfireball.markdown",
                body: .text(body)
            )
        }
        let summary = try await engine.ingest(.init(payloads: payloads))
        #expect(summary.acceptedCount == Self.corpus.count)
        return engine
    }

    /// The C1 reproduction: nonsense against a four-document index returned all four, each
    /// carrying only a fused rank. The corpus still comes back — that is how top-N works —
    /// but every row is now marked weak and carries the similarity that says why.
    @Test("a query matching nothing returns only weak results")
    func gibberishQueryIsWeak() async throws {
        let engine = try await engineWithCorpus()

        let response = try await engine.search(.init(query: "zzqqxx nonexistent term", limit: 10))

        #expect(!response.results.isEmpty, "top-N vector search still returns the tail")
        for result in response.results {
            let similarity = try #require(result.similarity, "vector hits must carry similarity")
            #expect(similarity < 0.15)
            #expect(result.isWeak, "\(result.documentID) is tail, not an answer")
        }
        #expect(response.results.allSatisfy { $0.diagnostics.vectorRank != nil })
        #expect(response.results.allSatisfy { $0.diagnostics.ftsRank == nil })
    }

    @Test("a genuine match is not weak and scores well above the floor")
    func genuineMatchIsStrong() async throws {
        let engine = try await engineWithCorpus()

        let response = try await engine.search(.init(query: "policy registry", limit: 10))

        let best = try #require(response.results.first)
        #expect(best.documentID == "doc-policy")
        #expect(best.isWeak == false)
        let similarity = try #require(best.similarity)
        #expect(similarity > 0.15)
    }

    /// A literal title hit is evidence on its own. Without the lexical guard the floor would
    /// discard an exact title match whose body shares no vocabulary with the query.
    @Test("a title match survives the floor even with low similarity")
    func lexicalEvidenceOverridesLowSimilarity() async throws {
        let engine = try await engineWithCorpus()

        let response = try await engine.search(.init(query: "Ledger Sync", limit: 10))

        let ledger = try #require(response.results.first { $0.documentID == "doc-ledger" })
        #expect(ledger.diagnostics.exactRank != nil || ledger.diagnostics.ftsRank != nil)
        #expect(ledger.isWeak == false)
    }

    /// The fused score is a position artifact. A result matching nothing still earns the exact
    /// RRF constant for its rank — the value the C1 audit observed as 1/(60+4) at rank 4 — which
    /// is why no floor can be built on it. Similarity is the value that actually varies with
    /// match quality.
    @Test("the fused score of a non-matching result is a pure rank artifact")
    func fusedScoreCarriesNoSimilarityInformation() async throws {
        let engine = try await engineWithCorpus()

        let weak = try await engine.search(.init(query: "zzqqxx nonexistent term", limit: 4))
        let strong = try await engine.search(.init(query: "policy registry", limit: 1))

        // Only the vector channel contributes, so each score is exactly 1/(k + rank).
        for (index, result) in weak.results.enumerated() {
            #expect(result.diagnostics.vectorRank == index + 1)
            #expect(abs(result.score - 1 / (60 + Double(index + 1))) < 1e-12)
        }

        let strongResult = try #require(strong.results.first)
        let strongSimilarity = try #require(strongResult.similarity)
        let weakSimilarity = try #require(weak.results.first?.similarity)
        #expect(strongSimilarity > weakSimilarity)
    }

    @Test("keyword hits carry a BM25 score oriented so larger is better")
    func keywordScoreIsExposed() async throws {
        let engine = try await engineWithCorpus()

        let response = try await engine.search(.init(query: "spectrogram", limit: 10))

        let mel = try #require(response.results.first { $0.documentID == "doc-mel" })
        #expect(mel.diagnostics.ftsRank != nil)
        let keywordScore = try #require(mel.diagnostics.keywordScore)
        #expect(keywordScore > 0, "SQLite BM25 is negated on the way out")
    }
}
