import CoreGraphics
import CoreML
import Foundation
import Glossematics
import ImageIO
import IndexEngine
import Testing
import UniformTypeIdentifiers
@testable import IndexEngineGloss

@Suite("IndexEngineGloss — Glossematics embedding adapter")
struct IndexEngineGlossTests {
    /// The bundled Core ML resource, or nil in checkouts where Git LFS did not pull the
    /// real payloads. Model-backed tests are gated on this so pointer-only checkouts skip
    /// rather than failing during Core ML load.
    static let repoBundle: URL? = GlossModelBundleLocator.locate()

    // MARK: - Fixtures

    /// A v2 manifest with every tower section, artifact names matching the shipped bundle.
    static func omniManifestFixture(
        modelID: String = "fixture-model", dimension: Int = 32
    ) -> GlossModelBundle.Manifest {
        GlossModelBundle.Manifest(
            modelID: modelID,
            embeddingDimension: dimension,
            matryoshkaDimensions: [dimension],
            prompts: .init(query: "Query: ", document: "Document: "),
            tokens: .init(
                padID: 151643,
                image: .init(prefixIDs: [1], suffixIDs: [2], placeholderID: 3),
                video: .init(prefixIDs: [1], suffixIDs: [2], placeholderID: 4),
                audio: .init(prefixIDs: [1], suffixIDs: [2], placeholderID: 5)
            ),
            text: .init(
                model: "text_multifunc.mlpackage", tokenizer: "tok",
                buckets: [16, 32], requiresAttentionMask: false
            ),
            image: .init(
                encoder: "vision_tower_masked_multifunc.mlpackage", resources: "vision_swift",
                patchBuckets: [1024],
                preprocess: .init(patch: 16, merge: 2, minPixels: 65536, maxPixels: 16_777_216)
            ),
            audio: .init(encoder: "audio_tower_masked_multifunc.mlpackage", frameBuckets: [200]),
            video: .init(encoder: "vision_tower_video_multifunc.mlpackage", patchBuckets: [256]),
            decoder: .init(
                embed: "embed_multifunc.mlpackage", model: "decoder_embeds_multifunc.mlpackage",
                sequenceBuckets: [128]
            )
        )
    }

    /// A v2 manifest with only the (mandatory) text section — the smallest valid bundle.
    static func textOnlyManifestFixture(tokenizer: String = "tok") -> GlossModelBundle.Manifest {
        GlossModelBundle.Manifest(
            modelID: "fixture-model",
            embeddingDimension: 32,
            matryoshkaDimensions: [32],
            prompts: .init(query: "Query: ", document: "Document: "),
            tokens: .init(padID: 0),
            text: .init(model: "text.mlpackage", tokenizer: tokenizer, buckets: [16], requiresAttentionMask: false)
        )
    }

    private static func writeManifestBundle(_ manifest: GlossModelBundle.Manifest) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gloss-manifest-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    // MARK: - Locator

    @Test("locator rejects a directory without a manifest")
    func locatorRejectsNonBundle() {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        #expect(GlossModelBundleLocator.isValidBundle(tmp) == false)
    }

    @Test("locator rejects manifest-only bundles before they are treated as model-backed")
    func locatorRejectsManifestOnlyBundle() throws {
        let tmp = try Self.writeManifestBundle(Self.omniManifestFixture())
        defer { try? FileManager.default.removeItem(at: tmp) }

        #expect(GlossModelBundleLocator.isValidBundle(tmp) == false)
    }

    @Test("locator rejects a bundle whose vision resources folder is missing the files the towers load")
    func locatorRejectsIncompleteVisionResources() throws {
        let manifest = Self.omniManifestFixture()
        let tmp = try Self.writeManifestBundle(manifest)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let bundle = GlossModelBundle(rootDirectory: tmp, manifest: manifest)

        let touch = { (url: URL) throws in
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true
            )
            try Data().write(to: url)
        }
        let tokenizerFolder = bundle.resolve(manifest.text.tokenizer)
        for artifact in [
            bundle.resolve(manifest.text.model),
            tokenizerFolder.appendingPathComponent("tokenizer.json"),
            tokenizerFolder.appendingPathComponent("tokenizer_config.json"),
            bundle.resolve(manifest.image!.encoder),
            bundle.resolve(manifest.audio!.encoder),
            bundle.resolve(manifest.video!.encoder),
            bundle.resolve(manifest.decoder!.embed),
            bundle.resolve(manifest.decoder!.model),
        ] {
            try touch(artifact)
        }
        let resources = bundle.resolve(manifest.image!.resources)
        for file in ["meta.json", "pos_embed_table.f32", "rope_inv_freq.f32"] {
            try touch(resources.appendingPathComponent(file))
        }

        #expect(GlossModelBundleLocator.isValidBundle(tmp))

        // A resources folder that exists but lacks a file the image tower loads by
        // name must not validate as model-backed.
        try FileManager.default.removeItem(at: resources.appendingPathComponent("meta.json"))
        #expect(GlossModelBundleLocator.isValidBundle(tmp) == false)
    }

    @Test(
        "locator discovers the bundled model resource when LFS payloads are present",
        .enabled(if: IndexEngineGlossTests.repoBundle != nil)
    )
    func locatorDiscoversBundledModelResource() throws {
        let bundle = try #require(Self.repoBundle)
        #expect(GlossModelBundleLocator.isValidBundle(bundle))
    }

    // MARK: - Space identity

    @Test("manifest formatting does not change the embedding space")
    func manifestFormattingDoesNotChangeEmbeddingSpace() throws {
        let manifest = Self.omniManifestFixture()
        let compact = FileManager.default.temporaryDirectory
            .appendingPathComponent("fingerprint-compact-\(UUID().uuidString)", isDirectory: true)
        let pretty = FileManager.default.temporaryDirectory
            .appendingPathComponent("fingerprint-pretty-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: compact, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: pretty, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: compact)
            try? FileManager.default.removeItem(at: pretty)
        }

        let compactEncoder = JSONEncoder()
        try compactEncoder.encode(manifest).write(to: compact.appendingPathComponent("manifest.json"))
        let prettyEncoder = JSONEncoder()
        prettyEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try prettyEncoder.encode(manifest).write(to: pretty.appendingPathComponent("manifest.json"))

        let compactLoaded = try GlossBundleLoader.load(url: compact)
        let prettyLoaded = try GlossBundleLoader.load(url: pretty)
        // Same decoded manifest, different bytes on disk: one embedding space, not two.
        #expect(compactLoaded.embeddingSpaceID == prettyLoaded.embeddingSpaceID)
    }

    @Test("manifest changes produce distinct embedding spaces")
    func manifestFingerprintParticipatesInEmbeddingSpace() throws {
        var changed = Self.omniManifestFixture()
        changed.text.model = "text-v2.mlpackage"

        let first = try Self.writeManifestBundle(Self.omniManifestFixture())
        let second = try Self.writeManifestBundle(changed)
        defer {
            try? FileManager.default.removeItem(at: first)
            try? FileManager.default.removeItem(at: second)
        }

        let firstLoaded = try GlossBundleLoader.load(url: first)
        let secondLoaded = try GlossBundleLoader.load(url: second)

        #expect(firstLoaded.embeddingSpaceID.hasPrefix("fixture-model:32:manifest-"))
        #expect(secondLoaded.embeddingSpaceID.hasPrefix("fixture-model:32:manifest-"))
        #expect(firstLoaded.embeddingSpaceID != secondLoaded.embeddingSpaceID)
    }

    // MARK: - v1 migration

    /// The shipped `JinaV5OmniSmall.bundle` manifest, verbatim. Also the shape of every copy
    /// already staged on user machines.
    static let shippedV1ManifestJSON = """
    {
      "formatVersion": 1,
      "modelID": "jinaai/jina-embeddings-v5-omni-small",
      "embeddingDimension": 1024,
      "minimumDeployment": { "macOS": "15.0", "iOS": "18.0" },
      "text": {
        "model": "text_multifunc.mlpackage",
        "tokenizer": "jina-v5-omni-small",
        "buckets": [32, 64, 128, 256, 512]
      },
      "image": {
        "encoder": "vision_tower_masked_multifunc.mlpackage",
        "resources": "vision_swift",
        "patchBuckets": [1024, 1600, 2304, 3072, 4032]
      },
      "audio": {
        "encoder": "audio_tower_masked_multifunc.mlpackage",
        "frameBuckets": [200, 400, 800, 1600, 3200]
      },
      "video": {
        "encoder": "vision_tower_video_multifunc.mlpackage",
        "patchBuckets": [256, 512, 1024, 2048]
      },
      "decoder": {
        "embed": "embed_multifunc.mlpackage",
        "model": "decoder_embeds_multifunc.mlpackage",
        "sequenceBuckets": [128, 256, 512, 1024]
      }
    }
    """

    private static func writeV1Bundle() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gloss-v1-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(shippedV1ManifestJSON.utf8).write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    @Test("a v1 manifest keeps the embedding space the retired Jina adapter derived for it")
    func v1MigrationPreservesEmbeddingSpace() throws {
        let bundleURL = try Self.writeV1Bundle()
        defer { try? FileManager.default.removeItem(at: bundleURL) }

        let loaded = try GlossBundleLoader.load(url: bundleURL)
        // Golden value computed with the retired `JinaModelBundle` derivation against the
        // shipped manifest. If this changes, every existing index re-embeds from scratch —
        // that must be a decision, never a side effect.
        #expect(loaded.embeddingSpaceID == "jinaai/jina-embeddings-v5-omni-small:1024:manifest-0bde6f92cfda")
    }

    @Test("v1 migration fills the v2 identity fields with the omni-small contract")
    func v1MigrationFillsIdentityFields() throws {
        let bundleURL = try Self.writeV1Bundle()
        defer { try? FileManager.default.removeItem(at: bundleURL) }

        let manifest = try GlossBundleLoader.load(url: bundleURL).bundle.manifest
        #expect(manifest.formatVersion == GlossModelBundle.Manifest.currentFormatVersion)
        #expect(manifest.prompts.query == "Query: ")
        #expect(manifest.prompts.document == "Document: ")
        #expect(manifest.tokens.padID == 151643)
        #expect(manifest.text.requiresAttentionMask == false)
        #expect(manifest.matryoshkaDimensions == [32, 64, 128, 256, 512, 1024])
        #expect(manifest.tokens.image?.mediaTokens == MediaTokens.jinaV5OmniSmallImage)
        #expect(manifest.tokens.video?.mediaTokens == MediaTokens.jinaV5OmniSmallVideo)
        #expect(manifest.tokens.audio?.mediaTokens == MediaTokens.jinaV5OmniSmallAudio)
        #expect(manifest.image?.preprocess.minPixels == 65536)
        #expect(manifest.image?.preprocess.maxPixels == 16_777_216)
        // Artifact paths carry over untouched.
        #expect(manifest.text.model == "text_multifunc.mlpackage")
        #expect(manifest.decoder?.model == "decoder_embeds_multifunc.mlpackage")
    }

    @Test("a v1 manifest for an unknown model is refused, not guessed at")
    func v1MigrationRefusesUnknownModel() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gloss-v1-unknown-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let json = Self.shippedV1ManifestJSON.replacingOccurrences(
            of: "jinaai/jina-embeddings-v5-omni-small", with: "someone/other-model")
        try Data(json.utf8).write(to: directory.appendingPathComponent("manifest.json"))

        #expect(throws: GlossBundleLoaderError.self) {
            _ = try GlossBundleLoader.load(url: directory)
        }
    }

    // MARK: - Tokenizer staging

    @Test("staging a tokenizer folder with a missing artifact throws instead of creating dangling links")
    func stagingRejectsMissingTokenizerArtifacts() throws {
        let manifest = Self.textOnlyManifestFixture()
        let bundleURL = try Self.writeManifestBundle(manifest)
        defer { try? FileManager.default.removeItem(at: bundleURL) }
        let tokenizerDirectory = bundleURL.appendingPathComponent("tok", isDirectory: true)
        try FileManager.default.createDirectory(at: tokenizerDirectory, withIntermediateDirectories: true)
        // tokenizer.json exists; tokenizer_config.json is missing.
        try Data("{}".utf8).write(to: tokenizerDirectory.appendingPathComponent("tokenizer.json"))

        let bundle = try GlossModelBundle(url: bundleURL)
        #expect(throws: (any Error).self) {
            _ = try GlossTokenizerStaging.resolvedFolder(for: bundle)
        }
    }

    @Test("tokenizer staging adds config.json outside the purgeable tmp directory and is idempotent")
    func tokenizerStagingResolvesDurableFolder() throws {
        let manifest = Self.textOnlyManifestFixture()
        let bundleURL = try Self.writeManifestBundle(manifest)
        defer { try? FileManager.default.removeItem(at: bundleURL) }

        // A tokenizer folder as the model bundle ships it: no config.json.
        let tokenizerDirectory = bundleURL.appendingPathComponent("tok", isDirectory: true)
        try FileManager.default.createDirectory(at: tokenizerDirectory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: tokenizerDirectory.appendingPathComponent("tokenizer.json"))
        try Data("{}".utf8).write(to: tokenizerDirectory.appendingPathComponent("tokenizer_config.json"))

        let bundle = try GlossModelBundle(url: bundleURL)
        let staged = try GlossTokenizerStaging.resolvedFolder(for: bundle)

        for name in ["tokenizer.json", "tokenizer_config.json", "config.json"] {
            #expect(
                FileManager.default.fileExists(atPath: staged.appendingPathComponent(name).path),
                "staged folder must contain \(name)"
            )
        }
        // Not under temporaryDirectory — the tmp cleaner could purge it before first use.
        #expect(!staged.path.hasPrefix(FileManager.default.temporaryDirectory.path))

        // Re-resolving the same bundle reuses the same staged folder.
        #expect(try GlossTokenizerStaging.resolvedFolder(for: bundle) == staged)

        // A tokenizer folder that already ships config.json is used in place.
        try Data("{}".utf8).write(to: tokenizerDirectory.appendingPathComponent("config.json"))
        #expect(try GlossTokenizerStaging.resolvedFolder(for: bundle) == tokenizerDirectory)
    }

    // MARK: - Fallback

    @Test("a missing bundle resolves to the hashing mock embedder")
    func fallbackWhenMissing() async {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let resolved = await IndexEngineGloss.resolveEmbedder(additionalCandidates: [missing])
        // Only assert the fallback when this machine has no real bundle installed,
        // otherwise the locator would (correctly) find the development bundle.
        if Self.repoBundle == nil {
            #expect(resolved.isModelBacked == false)
            #expect(resolved.embedder.modelID == HashingEmbedder().modelID)
        }
    }

    // MARK: - Model-backed behavior

    @Test(
        "text embeddings are 1024-d and rank a paraphrase above an unrelated sentence",
        .enabled(if: IndexEngineGlossTests.repoBundle != nil)
    )
    func semanticVectors() async throws {
        let bundle = try #require(Self.repoBundle)
        let provider = try await GlossTextEmbeddingProvider.load(bundleURL: bundle)
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
        .enabled(if: IndexEngineGlossTests.repoBundle != nil)
    )
    func endToEndSemanticRetrieval() async throws {
        let bundle = try #require(Self.repoBundle)
        let provider = try await GlossTextEmbeddingProvider.load(bundleURL: bundle)
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
        // channels cannot rank the thermal doc — only the vector channel can.
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
        .enabled(if: IndexEngineGlossTests.repoBundle != nil)
    )
    func omniSharedSpace() async throws {
        let bundle = try #require(Self.repoBundle)
        let provider = try await GlossOmniEmbeddingProvider.load(bundleURL: bundle)
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
        .enabled(if: IndexEngineGlossTests.repoBundle != nil)
    )
    func imageEndToEnd() async throws {
        let bundle = try #require(Self.repoBundle)
        let provider = try await GlossOmniEmbeddingProvider.load(bundleURL: bundle)
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

    @Test(
        "an oversized raw image is rejected before Core ML runs",
        .enabled(if: IndexEngineGlossTests.repoBundle != nil)
    )
    func oversizedRawImageIsRejected() throws {
        let bundleURL = try #require(Self.repoBundle)
        let loaded = try GlossBundleLoader.load(url: bundleURL)
        let embedder = try GlossOmniEmbedder(bundle: loaded.bundle).imageEmbedder()
        // 1024x1024 → a 64x64 patch grid (4096) over the 4032 bucket ceiling.
        let rgb = [UInt8](repeating: 128, count: 1024 * 1024 * 3)
        // Glossematics 0.2.0 reports this as `noOutput` where the retired in-repo embedder
        // had a purpose-built `tooLarge` (see EMB-1). Until that lands upstream, assert the
        // guard fires at all — the input must be refused, not fed to Core ML.
        #expect(throws: (any Error).self) {
            _ = try embedder.embed(rgb: rgb, h: 1024, w: 1024)
        }
    }

    @Test(
        "batched text embeddings match single embeddings",
        .enabled(if: IndexEngineGlossTests.repoBundle != nil)
    )
    func batchMatchesSingle() async throws {
        let bundle = try #require(Self.repoBundle)
        let provider = try await GlossOmniEmbeddingProvider.load(bundleURL: bundle)

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

    // MARK: - Benchmarks (opt-in via environment)

    @Test(
        "benchmark: ANE vs GPU text inference (set JINA_BENCH=1)",
        .enabled(if: IndexEngineGlossTests.repoBundle != nil && ProcessInfo.processInfo.environment["JINA_BENCH"] != nil)
    )
    func computeUnitBenchmark() async throws {
        let loaded = try GlossBundleLoader.load(url: #require(Self.repoBundle))
        let modelURL = loaded.bundle.resolve(loaded.bundle.manifest.text.model)
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

        print("\n=== Text tower — per-text latency (ms), lower is better ===")
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
                    // Single-row functions: a "batch" is a loop of independent calls.
                    func run() throws { for row in inputs { _ = try encoder.encode(tokenIds: row) } }
                    try run()   // warm up
                    var total = 0.0
                    for _ in 0 ..< iterations { total += try milliseconds(run) }
                    let perText = total / Double(iterations) / Double(batch)
                    row += padL(String(format: "%.1f", perText), 11)
                }
                print(row)
            }
        }
        #expect(Bool(true))
    }

    @Test(
        "benchmark: power per compute unit (set JINA_BENCH_POWER=1, sample with powermetrics)",
        .enabled(if: IndexEngineGlossTests.repoBundle != nil && ProcessInfo.processInfo.environment["JINA_BENCH_POWER"] != nil)
    )
    func powerBenchmark() async throws {
        let loaded = try GlossBundleLoader.load(url: #require(Self.repoBundle))
        let modelURL = loaded.bundle.resolve(loaded.bundle.manifest.text.model)
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
                for row in inputs { _ = try encoder.encode(tokenIds: row) }   // warm

                try await Task.sleep(for: .seconds(1.5))   // let power settle before the window
                let label = "\(name)_\(bucket)"
                mark("BEGIN", label)
                let start = Date()
                var count = 0
                while Date().timeIntervalSince(start) < window {
                    for row in inputs { _ = try encoder.encode(tokenIds: row) }
                    count += batch
                }
                let perText = Date().timeIntervalSince(start) / Double(count) * 1000
                mark("END", label, "perText=\(String(format: "%.2f", perText)) n=\(count)")
            }
        }
        #expect(Bool(true))
    }

    // MARK: - Helpers

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
