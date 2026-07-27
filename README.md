# Chartroom

A local-first hybrid retrieval engine for Apple platforms: SQLite FTS5 (BM25) keyword
search and exact vector cosine, fused with Reciprocal Rank Fusion, with optional
on-device embeddings.

Six layered Swift libraries over one external dependency:

| Library | Role |
| --- | --- |
| `IndexEngine` | SQLite FTS5 + exact vector search, fused with Reciprocal Rank Fusion. No external dependencies; builds and tests offline. |
| `ConnectorEngine` | Source connectors (local files, vaults) into normalized events. |
| `SyncEngine` | Connector-to-engine sync orchestration: event ordering, cursor advancement, pause/stop checkpoints. |
| `IndexEnginePDF` | Target-separated PDFKit content extractor. |
| `JinaEmbeddings` | `jina-embeddings-v5-omni-small` on the Apple Neural Engine (text, image, audio, video into one 1024-d space). |
| `IndexEngineJina` | Binds `JinaEmbeddings` behind `IndexEngine`'s `Embedder` protocol. |

## Requirements

macOS 15+.

## Installation

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/1of2-ai/Chartroom", from: "0.6.0")
```

Then depend on the products you need, e.g. `IndexEngine` and `IndexEngineJina`.

## License

MIT — see [LICENSE](LICENSE).
