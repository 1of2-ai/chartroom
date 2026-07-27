import Foundation
import Testing
@testable import IndexEngine

@Suite("Vector scan")
struct VectorScanTests {
    /// The scan keeps `k` of `n` rows. Ascending input is the case a naive "insert if better than
    /// the worst kept" would degrade on, and the case where an unbounded collect-then-sort was
    /// previously holding every row.
    @Test("bounded top-k keeps the best k regardless of arrival order")
    func boundedTopKSelection() {
        for ordering in ["ascending", "descending", "interleaved"] {
            var similarities = (1...100).map { Float($0) / 100 }
            switch ordering {
            case "descending": similarities.reverse()
            case "interleaved": similarities = stride(from: 0, to: 50, by: 1).flatMap { [similarities[$0], similarities[99 - $0]] }
            default: break
            }

            var heap = BoundedTopKHits(capacity: 5)
            for similarity in similarities {
                heap.insert(similarity: similarity, id: "id-\(Int(similarity * 100))")
            }

            let result = heap.sortedDescending()
            #expect(result.count == 5, "\(ordering)")
            #expect(result.map(\.id) == ["id-100", "id-99", "id-98", "id-97", "id-96"], "\(ordering)")
        }
    }

    @Test("a capacity larger than the input keeps everything, best first")
    func boundedTopKUnderfilled() {
        var heap = BoundedTopKHits(capacity: 10)
        for (index, similarity) in [Float(0.2), 0.9, 0.5].enumerated() {
            heap.insert(similarity: similarity, id: "id-\(index)")
        }
        #expect(heap.sortedDescending().map(\.id) == ["id-1", "id-2", "id-0"])
    }

    @Test("equal similarities break ties by id so repeated queries are stable")
    func boundedTopKTieBreak() {
        var heap = BoundedTopKHits(capacity: 3)
        for id in ["charlie", "alpha", "bravo"] {
            heap.insert(similarity: 0.5, id: id)
        }
        #expect(heap.sortedDescending().map(\.id) == ["alpha", "bravo", "charlie"])
    }

    @Test("zero capacity keeps nothing rather than trapping")
    func boundedTopKZeroCapacity() {
        var heap = BoundedTopKHits(capacity: 0)
        heap.insert(similarity: 1, id: "id")
        #expect(heap.sortedDescending().isEmpty)
    }

    /// A rejected row must not pay for its identifier. This is the allocation the scan avoids on
    /// every corpus chunk it does not return.
    @Test("a rejected row never materializes its id")
    func rejectedRowsSkipIdentifierWork() {
        var heap = BoundedTopKHits(capacity: 1)
        heap.insert(similarity: 1.0, id: "kept")

        var evaluations = 0
        for _ in 0..<100 {
            heap.insert(similarity: 0.1, id: { evaluations += 1; return "rejected" }())
        }

        #expect(evaluations == 0)
        #expect(heap.sortedDescending().map(\.id) == ["kept"])
    }

    /// Cosine reads the stored vector straight out of SQLite's buffer, which carries no alignment
    /// guarantee. Both paths must agree, and a wrong-sized blob must report rather than score.
    @Test("cosine over raw bytes matches the array path at any alignment")
    func cosineOverRawBytes() {
        let query: [Float] = (0..<64).map { Float($0).truncatingRemainder(dividingBy: 7) - 3 }
        let stored: [Float] = (0..<64).map { Float($0).truncatingRemainder(dividingBy: 5) - 2 }
        let expected = Vector.cosine(query, stored)

        let bytes = Vector.toBytes(stored)
        // A one-byte lead offset guarantees the float payload starts misaligned.
        var offsetBuffer = [UInt8](repeating: 0, count: 1)
        offsetBuffer.append(contentsOf: bytes)

        bytes.withUnsafeBytes { aligned in
            #expect(Vector.cosine(query: query, storedBytes: aligned) == expected)
        }
        offsetBuffer.withUnsafeBytes { raw in
            let misaligned = UnsafeRawBufferPointer(rebasing: raw[1...])
            #expect(Vector.cosine(query: query, storedBytes: misaligned) == expected)
        }
    }

    @Test("a blob of the wrong dimension scores nil instead of a wrong number")
    func cosineRejectsWrongDimension() {
        let query = [Float](repeating: 1, count: 8)
        let stored = [Float](repeating: 1, count: 4)
        Vector.toBytes(stored).withUnsafeBytes { bytes in
            #expect(Vector.cosine(query: query, storedBytes: bytes) == nil)
        }
    }
}
