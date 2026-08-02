import Foundation

enum FloatBufferCopyError: Error, Equatable {
    case countExceedsSource(requested: Int, available: Int)
    case misalignedFloatData(path: String, byteCount: Int)
}

/// Raw little-endian float32 file -> [Float]; shared by every resource loader.
/// A byte count that is not a whole number of floats is a corrupt resource —
/// `bindMemory` would floor-divide it into fewer floats and the damage would
/// surface later as an unrelated shape mismatch, far from the actual cause.
func loadF32(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    guard data.count.isMultiple(of: MemoryLayout<Float>.size) else {
        throw FloatBufferCopyError.misalignedFloatData(path: url.path, byteCount: data.count)
    }
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

func copyFloats(_ source: [Float], to destination: UnsafeMutablePointer<Float>, count: Int) throws {
    guard count <= source.count else {
        throw FloatBufferCopyError.countExceedsSource(requested: count, available: source.count)
    }
    guard count > 0 else { return }

    try source.withUnsafeBufferPointer { buffer in
        guard let baseAddress = buffer.baseAddress else {
            throw FloatBufferCopyError.countExceedsSource(requested: count, available: 0)
        }
        destination.update(from: baseAddress, count: count)
    }
}
