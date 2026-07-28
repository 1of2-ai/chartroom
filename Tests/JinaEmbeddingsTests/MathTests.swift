import Testing
@testable import JinaEmbeddings

@Test func cosineIdentity() {
    let v: [Float] = [1, 2, 3, 4]
    #expect(abs(cosine(v, v) - 1.0) < 1e-9)
}

@Test func matryoshkaIsUnitNorm() throws {
    let v: [Float] = (0..<1024).map { Float($0 % 7) - 3 }
    let m = try matryoshka(v, dim: 256)
    #expect(m.count == 256)
    var n: Float = 0; for x in m { n += x * x }
    #expect(abs(n.squareRoot() - 1.0) < 1e-5)
}

@Test func matryoshkaEnforcesFullDimensionBoundary() throws {
    let embedding: [Float] = [1, 2, 3, 4]

    #expect(throws: MatryoshkaError.self) {
        _ = try matryoshka(embedding, dim: -1)
    }
    #expect(throws: MatryoshkaError.self) {
        _ = try matryoshka(embedding, dim: 0)
    }
    let exact = try matryoshka(embedding, dim: embedding.count)
    #expect(exact.count == embedding.count)
    #expect(throws: MatryoshkaError.self) {
        _ = try matryoshka(embedding, dim: embedding.count + 1)
    }
}

@Test func matryoshkaBatchMappingPropagatesInvalidDimensions() {
    let embeddings: [[Float]] = [
        [1, 2, 3, 4],
        [1, 2],
    ]

    #expect(throws: MatryoshkaError.self) {
        _ = try embeddings.map { try matryoshka($0, dim: 3) }
    }
}

@Test func textBatchPreflightRejectsInvalidDimensionsWithoutInputs() throws {
    for invalidDimension in [-1, 0, 1025] {
        #expect(throws: MatryoshkaError.self) {
            try JinaTextEmbedder.validateRequestedDimension(invalidDimension)
        }
    }
    try JinaTextEmbedder.validateRequestedDimension(1024)
}
