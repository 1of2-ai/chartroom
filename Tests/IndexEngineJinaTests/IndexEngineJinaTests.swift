import CoreGraphics
import CoreML
import Foundation
import ImageIO
import IndexEngine
import JinaEmbeddings
import Testing
import UniformTypeIdentifiers
@testable import IndexEngineJina

@Suite("IndexEngineJina — Jina text embedding adapter")
struct IndexEngineJinaTests {
    /// The bundled Core ML resource, or nil in checkouts where Git LFS did not pull the
    /// real payloads. Model-backed tests are gated on this so pointer-only checkouts skip
    /// rather than failing during Core ML load.
    static let repoBundle: URL? = JinaModelBundleLocator.locate()

    @Test("locator rejects a directory without a manifest")
    func locatorRejectsNonBundle() {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        #expect(JinaModelBundleLocator.isValidBundle(tmp) == false)
    }

    @Test("locator rejects manifest-only bundles before they are treated as model-backed")
    func locatorRejectsManifestOnlyBundle() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("manifest-only-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let data = try JSONEncoder().encode(JinaModelBundle.Manifest.default)
        try data.write(to: tmp.appendingPathComponent("manifest.json"))

        #expect(JinaModelBundleLocator.isValidBundle(tmp) == false)
    }

    @Test(
        "locator discovers the bundled model resource when LFS payloads are present",
        .enabled(if: IndexEngineJinaTests.repoBundle != nil)
    )
    func locatorDiscoversBundledModelResource() throws {
        let bundle = try #require(Self.repoBundle)
        #expect(JinaModelBundleLocator.isValidBundle(bundle))
    }

    @Test("manifest changes produce distinct embedding spaces")
    func manifestFingerprintParticipatesInEmbeddingSpace() throws {
        let first = try Self.writeManifestBundle(
            JinaModelBundle.Manifest(modelID: "fixture-model", embeddingDimension: 32)
        )
        let second = try Self.writeManifestBundle(
            JinaModelBundle.Manifest(
                modelID: "fixture-model",
                embeddingDimension: 32,
                text: .init(model: "text-v2.mlpackage", tokenizer: "tok", buckets: [16, 32])
            )
        )
        defer {
            try? FileManager.default.removeItem(at: first)
            try? FileManager.default.removeItem(at: second)
        }

        let firstBundle = try JinaModelBundle(url: first)
        let secondBundle = try JinaModelBundle(url: second)

        #expect(firstBundle.embeddingSpaceID.hasPrefix("fixture-model:32:manifest-"))
        #expect(secondBundle.embeddingSpaceID.hasPrefix("fixture-model:32:manifest-"))
        #expect(firstBundle.embeddingSpaceID != secondBundle.embeddingSpaceID)
    }

    @Test("a missing bundle resolves to the hashing mock embedder")
    func fallbackWhenMissing() async {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let resolved = await IndexEngineJina.resolveEmbedder(additionalCandidates: [missing])
        // Only assert the fallback when this machine has no real bundle installed,
        // otherwise the locator would (correctly) find the development bundle.
        if Self.repoBundle == nil {
            #expect(resolved.isModelBacked == false)
            #expect(resolved.embedder.modelID == HashingEmbedder().modelID)
        }
    }

    private static func writeManifestBundle(_ manifest: JinaModelBundle.Manifest) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("jina-manifest-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    @Test("tokenizer staging adds config.json outside the purgeable tmp directory and is idempotent")
    func tokenizerStagingResolvesDurableFolder() throws {
        let bundleURL = try Self.writeManifestBundle(
            JinaModelBundle.Manifest(
                modelID: "fixture-model",
                embeddingDimension: 32,
                text: .init(model: "text.mlpackage", tokenizer: "tok", buckets: [16])
            )
        )
        defer { try? FileManager.default.removeItem(at: bundleURL) }

        // A tokenizer folder as the model bundle ships it: no config.json.
        let tokenizerDirectory = bundleURL.appendingPathComponent("tok", isDirectory: true)
        try FileManager.default.createDirectory(at: tokenizerDirectory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: tokenizerDirectory.appendingPathComponent("tokenizer.json"))
        try Data("{}".utf8).write(to: tokenizerDirectory.appendingPathComponent("tokenizer_config.json"))

        let bundle = try JinaModelBundle(url: bundleURL)
        let staged = try JinaTokenizerStaging.resolvedFolder(for: bundle)

        for name in ["tokenizer.json", "tokenizer_config.json", "config.json"] {
            #expect(
                FileManager.default.fileExists(atPath: staged.appendingPathComponent(name).path),
                "staged folder must contain \(name)"
            )
        }
        // Not under temporaryDirectory — the tmp cleaner could purge it before first use.
        #expect(!staged.path.hasPrefix(FileManager.default.temporaryDirectory.path))

        // Re-resolving the same bundle reuses the same staged folder.
        #expect(try JinaTokenizerStaging.resolvedFolder(for: bundle) == staged)

        // A tokenizer folder that already ships config.json is used in place.
        try Data("{}".utf8).write(to: tokenizerDirectory.appendingPathComponent("config.json"))
        #expect(try JinaTokenizerStaging.resolvedFolder(for: bundle) == tokenizerDirectory)
    }

    @Test(
        "Jina text embeddings are 1024-d and rank a paraphrase above an unrelated sentence",
        .enabled(if: IndexEngineJinaTests.repoBundle != nil)
    )
    func semanticVectors() async throws {
        let bundle = try #require(Self.repoBundle)
        let provider = try await JinaTextEmbeddingProvider.load(bundleURL: bundle)
        #expect(provider.dimension == 1024)
        #expect(provider.modelID == "jinaai/jina-embeddings-v5-omni-small")

        let query = try await provider.embed("how do I cool an overheating laptop", kind: .query)
        let related = try await provider.embed("tips to stop a computer from thermal throttling", kind: .document)
        let unrelated = try await provider.embed("a recipe for sourdough bread", kind: .document)

        #expect(query.count == 1024)
        // The hashing mock cannot do this: a paraphrase with no shared content words
        // must still score above an unrelated sentence under real semantic embeddings.
        #expect(cosine(query, related) > cosine(query, unrelated))
    }

    @Test(
        "end to end: a content-word-free query retrieves the semantically relevant doc through the engine",
        .enabled(if: IndexEngineJinaTests.repoBundle != nil)
    )
    func endToEndSemanticRetrieval() async throws {
        let bundle = try #require(Self.repoBundle)
        let provider = try await JinaTextEmbeddingProvider.load(bundleURL: bundle)
        let engine = try await IndexEngine.openInMemory(
            configuration: IndexEngineConfiguration(embedder: provider)
        )

        _ = try await engine.ingest(IngestRequest(payloads: [
            SourcePayload(
                documentID: "doc-thermal",
                displayName: "Thermal",
                body: .text("Tips to stop a notebook PC from thermal throttling under sustained GPU load.")
            ),
            SourcePayload(
                documentID: "doc-bread",
                displayName: "Bread",
                body: .text("A traditional sourdough recipe with a long overnight fermentation.")
            ),
        ]))

        // Shares no meaningful tokens with either document, so the lexical/BM25
        // channels cannot rank the thermal doc — only the Jina vector channel can.
        let response = try await engine.search(
            SearchRequest(query: "how do I cool an overheating laptop", mode: .diagnostic)
        )

        let top = try #require(response.results.first)
        #expect(top.documentID == "doc-thermal")
        // The vector channel must have actually run (real embeddings present).
        #expect(response.diagnostics.degraded == false)
    }

    @Test(
        "omni embeds text and an image into the same 1024-d space",
        .enabled(if: IndexEngineJinaTests.repoBundle != nil)
    )
    func omniSharedSpace() async throws {
        let bundle = try #require(Self.repoBundle)
        let provider = try await JinaOmniEmbeddingProvider.load(bundleURL: bundle)
        #expect(provider.supportsImageEmbedding)

        let imageURL = try Self.makePNG(red: 0.85, green: 0.1, blue: 0.1)
        defer { try? FileManager.default.removeItem(at: imageURL) }

        let textVector = try await provider.embed("a solid red square", kind: .query)
        let imageVector = try await provider.embedImage(at: imageURL)

        #expect(textVector.count == 1024)
        #expect(imageVector.count == 1024)
        #expect(imageVector.allSatisfy { $0.isFinite })
        #expect(textVector != imageVector)
    }

    @Test(
        "end to end: an image ingested through the engine is embedded by content and retrievable",
        .enabled(if: IndexEngineJinaTests.repoBundle != nil)
    )
    func imageEndToEnd() async throws {
        let bundle = try #require(Self.repoBundle)
        let provider = try await JinaOmniEmbeddingProvider.load(bundleURL: bundle)
        let engine = try await IndexEngine.openInMemory(
            configuration: IndexEngineConfiguration(embedder: provider)
        )

        let imageURL = try Self.makePNG(red: 0.1, green: 0.2, blue: 0.9)
        defer { try? FileManager.default.removeItem(at: imageURL) }

        let ingestSummary = try await engine.ingest(IngestRequest(payloads: [
            SourcePayload(
                documentID: "img-1",
                sourceURI: imageURL,
                displayName: "blue.png",
                contentType: "public.png",
                body: .binaryReference(imageURL)
            )
        ]))
        // Ingestion failures were previously invisible here: the summary was discarded, so a
        // failed embed surfaced only as a confusing zero count further down.
        #expect(
            ingestSummary.failedCount == 0,
            "ingest failed: \(ingestSummary.failures.map { "\($0.category): \($0.message) — \($0.detail)" })"
        )
        #expect(ingestSummary.acceptedCount == 1)

        // The image was embedded by its pixels (the vector channel), not its filename: a
        // text query lands in the same space and retrieves it without sharing any words.
        let snapshot = await engine.snapshot()
        #expect(snapshot.embeddingCount >= 1)

        let response = try await engine.search(SearchRequest(query: "a photograph", mode: .diagnostic))
        #expect(response.results.contains { $0.documentID == "img-1" })
        #expect(response.diagnostics.degraded == false)
    }

    /// Write a solid-color PNG to a temp file via Core Graphics + ImageIO (thread-safe; no
    /// AppKit main-thread requirement).
    static func makePNG(red: CGFloat, green: CGFloat, blue: CGFloat, side: Int = 256) throws -> URL {
        guard let context = CGContext(
            data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw CocoaError(.fileWriteUnknown) }
        context.setFillColor(red: red, green: green, blue: blue, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        guard let image = context.makeImage() else { throw CocoaError(.fileWriteUnknown) }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("png")
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
        return url
    }

    @Test(
        "batched text embeddings match single embeddings",
        .enabled(if: IndexEngineJinaTests.repoBundle != nil)
    )
    func batchMatchesSingle() async throws {
        let bundle = try #require(Self.repoBundle)
        let provider = try await JinaOmniEmbeddingProvider.load(bundleURL: bundle)

        // Mixed lengths so the batch spans more than one sequence-length bucket.
        let texts = [
            "navigation",
            "minimalist navigation patterns for web usability",
            "the design of everyday things explains affordances and signifiers in great depth",
        ]
        let batched = try await provider.embed(texts, kind: .document)
        #expect(batched.count == texts.count)

        for (index, text) in texts.enumerated() {
            let single = try await provider.embed(text, kind: .document)
            #expect(batched[index].count == 1024)
            // Same model, same input → effectively identical vectors.
            #expect(cosine(batched[index], single) > 0.999)
        }
    }

    @Test(
        "benchmark: ANE vs GPU text inference (set JINA_BENCH=1)",
        .enabled(if: IndexEngineJinaTests.repoBundle != nil && ProcessInfo.processInfo.environment["JINA_BENCH"] != nil)
    )
    func computeUnitBenchmark() async throws {
        let bundle = try JinaModelBundle(url: #require(Self.repoBundle))
        let modelURL = bundle.resolve(bundle.manifest.text.model)
        let compiled = modelURL.pathExtension == "mlmodelc" ? modelURL : try await MLModel.compileModel(at: modelURL)

        let units: [(String, MLComputeUnits)] = [("ANE", .cpuAndNeuralEngine), ("GPU", .cpuAndGPU)]
        let buckets = [32, 128, 512]
        let batchSizes = [1, 4, 16]
        let iterations = 6

        func pad(_ s: String, _ w: Int) -> String { s.count >= w ? s : s + String(repeating: " ", count: w - s.count) }
        func padL(_ s: String, _ w: Int) -> String { s.count >= w ? s : String(repeating: " ", count: w - s.count) + s }
        func milliseconds(_ block: () throws -> Void) rethrows -> Double {
            let start = Date(); try block(); return Date().timeIntervalSince(start) * 1000
        }

        print("\n=== Jina text tower — per-text latency (ms), lower is better ===")
        var header = pad("unit", 6) + pad("bucket", 8)
        for b in batchSizes { header += padL("b\(b)/txt", 11) }
        print(header)

        for (name, unit) in units {
            for bucket in buckets {
                let encoder = try CoreMLTextEncoder(modelURL: compiled, computeUnits: unit, functionName: "bucket_\(bucket)")
                let ids = [Int32](repeating: 100, count: bucket)
                var row = pad(name, 6) + pad("\(bucket)", 8)
                for batch in batchSizes {
                    let inputs = Array(repeating: ids, count: batch)
                    _ = try encoder.encode(batch: inputs)   // warm up
                    var total = 0.0
                    for _ in 0 ..< iterations { total += try milliseconds { _ = try encoder.encode(batch: inputs) } }
                    let perText = total / Double(iterations) / Double(batch)
                    row += padL(String(format: "%.1f", perText), 11)
                }
                print(row)
            }
        }
        #expect(Bool(true))
    }

    @Test(
        "benchmark: power per compute unit (run via Scripts/bench_power.sh)",
        .enabled(if: IndexEngineJinaTests.repoBundle != nil && ProcessInfo.processInfo.environment["JINA_BENCH_POWER"] != nil)
    )
    func powerBenchmark() async throws {
        let bundle = try JinaModelBundle(url: #require(Self.repoBundle))
        let modelURL = bundle.resolve(bundle.manifest.text.model)
        let compiled = modelURL.pathExtension == "mlmodelc" ? modelURL : try await MLModel.compileModel(at: modelURL)

        let units: [(String, MLComputeUnits)] = [("ANE", .cpuAndNeuralEngine), ("GPU", .cpuAndGPU)]
        let buckets = [32, 128, 512]
        let batch = 16
        let window = Double(ProcessInfo.processInfo.environment["JINA_BENCH_SECONDS"] ?? "") ?? 6

        func mark(_ phase: String, _ label: String, _ extra: String = "") {
            print("MARK \(String(format: "%.3f", Date().timeIntervalSince1970)) \(phase) \(label) \(extra)")
            fflush(stdout)
        }

        // Idle baseline so the parser can report marginal (over-idle) power per config.
        try await Task.sleep(for: .seconds(2))
        mark("BEGIN", "idle")
        try await Task.sleep(for: .seconds(window))
        mark("END", "idle")

        for (name, unit) in units {
            for bucket in buckets {
                let encoder = try CoreMLTextEncoder(modelURL: compiled, computeUnits: unit, functionName: "bucket_\(bucket)")
                let inputs = Array(repeating: [Int32](repeating: 100, count: bucket), count: batch)
                _ = try encoder.encode(batch: inputs)   // warm

                try await Task.sleep(for: .seconds(1.5))   // let power settle before the window
                let label = "\(name)_\(bucket)"
                mark("BEGIN", label)
                let start = Date()
                var count = 0
                while Date().timeIntervalSince(start) < window {
                    _ = try encoder.encode(batch: inputs)
                    count += batch
                }
                let perText = Date().timeIntervalSince(start) / Double(count) * 1000
                mark("END", label, "perText=\(String(format: "%.2f", perText)) n=\(count)")
            }
        }
        #expect(Bool(true))
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in a.indices {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denom = na.squareRoot() * nb.squareRoot()
        return denom > 0 ? dot / denom : 0
    }
}
