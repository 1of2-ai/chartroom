// swift-tools-version: 6.0
import PackageDescription

// Chartroom — the local-first hybrid retrieval engine.
//
// A distributable package of five layered libraries, one external dependency:
//
//   IndexEngine       SQLite FTS5 (BM25) + exact vector cosine, fused with Reciprocal Rank
//                     Fusion. Built on the system SQLite, so it has no external dependencies
//                     and is fully buildable and testable offline.
//   ConnectorEngine   Source connectors (local files, vaults) → normalized SourcePayload events.
//   SyncEngine        Connector→engine sync orchestration: event ordering, cursor advancement,
//                     pause/stop checkpoints.
//   IndexEnginePDF    Target-separated PDFKit ContentExtractor, registered by the host.
//   IndexEngineGloss  Adapter binding GlossematicsSDK (jina-embeddings v5 on Core ML: text/
//                     image/audio/video into one shared space) behind IndexEngine's Embedder
//                     protocol.
//
// GlossematicsSDK is consumed as a versioned remote dependency; the GlossematicsSDK submodule
// vendors the same pinned revision for source-of-truth browsing and offline reference. Keep
// the two pins moving together.
//
// One platform floor: IndexEngineGloss and Glossematics require macOS 15 (MLComputePlan),
// so the whole package floors there.
let package = Package(
    name: "Chartroom",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "IndexEngine", targets: ["IndexEngine"]),
        .library(name: "ConnectorEngine", targets: ["ConnectorEngine"]),
        .library(name: "SyncEngine", targets: ["SyncEngine"]),
        .library(name: "IndexEnginePDF", targets: ["IndexEnginePDF"]),
        .library(name: "IndexEngineGloss", targets: ["IndexEngineGloss"]),
        .library(name: "ChartroomControl", targets: ["ChartroomControl"]),
    ],
    dependencies: [
        .package(name: "GlossematicsSDK", url: "https://github.com/1of2-ai/GlossematicsSDK", from: "0.2.0"),
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
            name: "IndexEngineGloss",
            dependencies: [
                "IndexEngine",
                .product(name: "Glossematics", package: "GlossematicsSDK"),
            ],
            resources: [.copy("Resources/CoreML/JinaV5OmniSmall.bundle")]
        ),
        .testTarget(name: "IndexEngineGlossTests", dependencies: ["IndexEngineGloss"]),

        .target(name: "ChartroomControl", dependencies: ["ConnectorEngine", "IndexEngine", "SyncEngine"]),
        .testTarget(name: "ChartroomControlTests", dependencies: ["ChartroomControl", "ChartroomTestSupport", "ConnectorEngine", "IndexEngine"]),

        .testTarget(
            name: "GlossematicsSDKTests",
            dependencies: [.product(name: "Glossematics", package: "GlossematicsSDK")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
