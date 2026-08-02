import ConnectorEngine
import Foundation
import IndexEngine
import UniformTypeIdentifiers

/// Rebuilds retryable local-file payloads from engine failure records. Only failures whose
/// source file still exists on disk — and still passes the connector's file policy — can be
/// retried; everything else is skipped. The file may have changed arbitrarily between the
/// failed sync and the retry, so the policy is re-applied, not assumed.
///
/// Root containment is the one connector check that cannot be re-applied here: the failure
/// record carries the file's URI but not the root it was synced under.
public enum LocalFileRetry {
    /// The source ID `LocalFileConnector`-backed ingests record on their payloads.
    public static let localFilesSourceID: SourceID = "local-files"

    public static func payloads(
        for failures: [FailureSnapshot],
        sourceID: SourceID = LocalFileRetry.localFilesSourceID,
        options: LocalFileConnectorOptions = .init()
    ) -> [SourcePayload] {
        failures.compactMap { failure in
            guard failure.sourceID == sourceID, let documentID = failure.documentID else { return nil }
            return payload(
                forDocumentID: documentID,
                sourceURI: failure.sourceURI,
                sourceID: sourceID,
                options: options
            )
        }
    }

    /// Rebuild one payload from the failure's recorded source location.
    static func payload(
        forDocumentID documentID: DocumentID,
        sourceURI: URL?,
        sourceID: SourceID,
        options: LocalFileConnectorOptions = .init()
    ) -> SourcePayload? {
        guard let fileURL = sourceURI?.standardizedFileURL, fileURL.isFileURL else { return nil }
        guard let values = try? fileURL.resourceValues(forKeys: [
            .isRegularFileKey, .isSymbolicLinkKey, .isHiddenKey, .fileSizeKey, .contentTypeKey,
        ]) else { return nil }
        guard values.isSymbolicLink != true, values.isRegularFile == true else { return nil }
        if !options.includeHiddenFiles, values.isHidden == true { return nil }
        if let allowed = options.allowedPathExtensions,
           !allowed.contains(fileURL.pathExtension.lowercased()) {
            return nil
        }
        if let size = values.fileSize, Int64(size) > options.maxFileSizeBytes { return nil }

        return SourcePayload(
            documentID: documentID,
            sourceID: sourceID,
            sourceURI: fileURL,
            displayName: fileURL.lastPathComponent,
            contentType: values.contentType?.identifier ?? UTType.data.identifier,
            body: .binaryReference(fileURL)
        )
    }
}
