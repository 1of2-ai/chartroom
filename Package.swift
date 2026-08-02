// swift-tools-version: 6.0
import PackageDescription

// Chartroom — the local-first hybrid retrieval engine.
//
// A distributable package of six layered libraries, two external dependencies:
//
//   IndexEngine      SQLite FTS5 (BM25) + exact vector cosine, fused with Reciprocal Rank
//                    Fusion. Built on the system SQLite, so it has no external dependencies
//                    and is fully buildable and testable offline.
//   ConnectorEngine  Source connectors (local files, vaults) → normalized SourcePayload events.
//   SyncEngine       Connector→engine sync orchestration: event ordering, cursor advancement,
//                    pause/stop checkpoints.
//   IndexEnginePDF   Target-separated PDFKit ContentExtractor, registered by the host.
//   JinaEmbeddings   jina-embeddings-v5-omni-small on the Apple Neural Engine (text/image/
//                    audio/video into one 1024-d space).
//   IndexEngineJina  Adapter binding JinaEmbeddings behind IndexEngine's Embedder protocol.
//
// One platform floor: IndexEngineJina and JinaEmbeddings require macOS 15 (MLComputePlan),
// so the whole package floors there.
let package = Package(
    name: "Chartroom",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "IndexEngine", targets: ["IndexEngine"]),
        .library(name: "ConnectorEngine", targets: ["ConnectorEngine"]),
        .library(name: "SyncEngine", targets: ["SyncEngine"]),
        .library(name: "IndexEnginePDF", targets: ["IndexEnginePDF"]),
        .library(name: "JinaEmbeddings", targets: ["JinaEmbeddings"]),
        .library(name: "IndexEngineJina", targets: ["IndexEngineJina"]),
        .library(name: "ChartroomControl", targets: ["ChartroomControl"]),
    ],
    dependencies: [
        .package(name: "GlossematicsSDK", url: "https://github.com/1of2-ai/GlossematicsSDK", from: "0.2.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.17"),
    ],
    targets: [
        .target(
            name: "ChartroomTestSupport",
            dependencies: ["ConnectorEngine", "IndexEngine", "SyncEngine"],
            path: "Tests/ChartroomTestSupport"
        ),

        .target(
            name: "IndexEngine",
            linkerSettings: [.linkedLibrary("sqlite3")]
        ),
        .testTarget(name: "IndexEngineTests", dependencies: ["ChartroomTestSupport", "IndexEngine"]),

        .target(name: "ConnectorEngine", dependencies: ["IndexEngine"]),
        .testTarget(name: "ConnectorEngineTests", dependencies: ["ConnectorEngine"]),

        .target(name: "SyncEngine", dependencies: ["ConnectorEngine", "IndexEngine"]),
        .testTarget(name: "SyncEngineTests", dependencies: ["ChartroomTestSupport", "SyncEngine"]),

        .target(name: "IndexEnginePDF", dependencies: ["IndexEngine"]),
        .testTarget(name: "IndexEnginePDFTests", dependencies: ["IndexEnginePDF"]),

        .target(
            name: "JinaEmbeddings",
            dependencies: [.product(name: "Transformers", package: "swift-transformers")],
            resources: [
                .copy("Resources/mel_filters.f32"),
                .copy("Resources/mel_window.f32"),
            ]
        ),
        .testTarget(name: "JinaEmbeddingsTests", dependencies: ["JinaEmbeddings"]),

        .target(
            name: "IndexEngineJina",
            dependencies: ["IndexEngine", "JinaEmbeddings"],
            resources: [.copy("Resources/CoreML/JinaV5OmniSmall.bundle")]
        ),
        .testTarget(name: "IndexEngineJinaTests", dependencies: ["IndexEngineJina"]),

        .target(name: "ChartroomControl", dependencies: ["ConnectorEngine", "IndexEngine", "SyncEngine"]),
        .testTarget(name: "ChartroomControlTests", dependencies: ["ChartroomControl", "ChartroomTestSupport", "ConnectorEngine", "IndexEngine"]),

        .testTarget(
            name: "GlossematicsSDKTests",
            dependencies: [.product(name: "Glossematics", package: "GlossematicsSDK")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
