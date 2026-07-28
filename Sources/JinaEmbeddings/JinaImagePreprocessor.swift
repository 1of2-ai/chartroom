import CoreGraphics
import Foundation
import ImageIO

/// Replicates the Qwen3VL image processor for the fixed-resolution ViT: resize to a canonical
/// `size`x`size` (square; MVP — full smart-resize/aspect ratio is future multi-resolution work),
/// normalize `(v/255 - 0.5)/0.5`, and patchify to `pixel_values` of shape
/// `(gh*gw*merge*merge, channel*temporal*patch*patch)` = `(1024, 1536)` for size 512.
///
/// Patchify index map (verified against the processor):
///   pixel_values[((gh*GW+gw)*merge+mh)*merge+mw][((c*temporal+t)*patch+ph)*patch+pw]
///     = norm(image[c][gh*merge*patch + mh*patch + ph][gw*merge*patch + mw*patch + pw])
/// with temporal frames identical (single image repeated).
public struct JinaImagePreprocessor {
    public let size: Int
    public let patch = 16, merge = 2, temporal = 2

    public init(size: Int = 512) { self.size = size }

    public var numPatches: Int { let g = size / patch; return g * g }              // 1024
    public var featuresPerPatch: Int { 3 * temporal * patch * patch }             // 1536

    // Smart-resize bounds for this model's processor (Qwen3VL): factor = patch*merge = 32.
    public let minPixels = 65536, maxPixels = 16777216
    public var factor: Int { patch * merge }

    /// Qwen smart_resize: round H,W to multiples of `factor`, keep aspect ratio, clamp the pixel
    /// budget to [minPixels, maxPixels]. Returns (Hbar, Wbar) — the ViT then sees grid (Hbar/16, Wbar/16).
    /// `maxPixelsOverride` caps the budget below the model's native max (used to fit a patch-bucket
    /// ceiling: images larger than the largest converted bucket are downscaled instead of unsupported).
    public func smartResize(h: Int, w: Int, maxPixelsOverride: Int? = nil) -> (Int, Int) {
        let f = Double(factor)
        let maxPixels = min(self.maxPixels, maxPixelsOverride ?? self.maxPixels)
        func roundF(_ x: Double) -> Int { Int((x / f).rounded()) * factor }
        var hbar = max(factor, roundF(Double(h)))
        var wbar = max(factor, roundF(Double(w)))
        if hbar * wbar > maxPixels {
            let beta = (Double(h * w) / Double(maxPixels)).squareRoot()
            hbar = max(factor, Int((Double(h) / beta / f).rounded(.down)) * factor)
            wbar = max(factor, Int((Double(w) / beta / f).rounded(.down)) * factor)
        } else if hbar * wbar < minPixels {
            let beta = (Double(minPixels) / Double(h * w)).squareRoot()
            hbar = Int((Double(h) * beta / f).rounded(.up)) * factor
            wbar = Int((Double(w) * beta / f).rounded(.up)) * factor
        }
        return (hbar, wbar)
    }

    /// General variable-resolution patchify from a row-major RGB buffer already sized to (h,w) with
    /// h,w factor-aligned (e.g. a smart-resized image). Returns pixel_values (gh*gw, featuresPerPatch)
    /// in the merger's 2×2-block order plus the patch grid (gh=h/16, gw=w/16).
    public func pixelValues(rgb: [UInt8], h: Int, w: Int) throws -> (pixels: [Float], gh: Int, gw: Int) {
        let layout = try validatedRawTensorLayout(t: 1, h: h, w: w)
        guard rgb.count == layout.expectedByteCount else {
            throw ImageError.invalidBufferLength(expected: layout.expectedByteCount, actual: rgb.count)
        }
        let GH = layout.gh / merge, GW = layout.gw / merge       // merge-block grid
        let gh = layout.gh, gw = layout.gw                        // patch grid (for positions)
        let FPP = featuresPerPatch, bpr = w * 3
        var out = [Float](repeating: 0, count: layout.elementCount)
        for bh in 0..<GH {
            for bw in 0..<GW {
                for mh in 0..<merge {
                    for mw in 0..<merge {
                        let patchIdx = ((bh * GW + bw) * merge + mh) * merge + mw
                        let base = patchIdx * FPP
                        for c in 0..<3 {
                            for ph in 0..<patch {
                                let H = bh * (merge * patch) + mh * patch + ph
                                let row = H * bpr
                                for pw in 0..<patch {
                                    let W = bw * (merge * patch) + mw * patch + pw
                                    let v = Float(rgb[row + W * 3 + c]) / 127.5 - 1.0
                                    for t in 0..<temporal {
                                        out[base + ((c * temporal + t) * patch + ph) * patch + pw] = v
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return (out, gh, gw)
    }

    public enum ImageError: Error {
        case load(String)
        case context
        case badVideoFrameCount(Int)
        case invalidDimensions(h: Int, w: Int, factor: Int)
        case bufferSizeOverflow(h: Int, w: Int)
        case invalidBufferLength(expected: Int, actual: Int)
        case invalidTensorGrid(t: Int, gh: Int, gw: Int)
        case unmergeableTensorGrid(gh: Int, gw: Int, merge: Int)
        case tensorSizeOverflow(t: Int, gh: Int, gw: Int)
        case invalidPixelValuesLength(expected: Int, actual: Int)
    }

    public static func loadCGImage(_ url: URL) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw ImageError.load(url.path)
        }
        return img
    }

    /// VIDEO frame-patchify: `frames` = `2·t` RGB buffers (h*w*3 each, h/w factor-aligned). Pairs
    /// consecutive frames into the temporal_patch_size=2 slots (patch's 1536 = c×temporal×16×16, slot
    /// 0 = frame 2g, slot 1 = frame 2g+1), in the merger's 2×2-block order. Returns pixel_values_videos
    /// (t·gh·gw, featuresPerPatch) + grid (t, gh, gw). (fps frame *sampling* is the caller's job.)
    public func videoPixelValues(frames: [[UInt8]], h: Int, w: Int) throws -> (pixels: [Float], t: Int, gh: Int, gw: Int) {
        guard frames.count > 0, frames.count % temporal == 0 else { throw ImageError.badVideoFrameCount(frames.count) }
        let t = frames.count / temporal
        let layout = try validatedRawTensorLayout(t: t, h: h, w: w)
        for frame in frames where frame.count != layout.expectedByteCount {
            throw ImageError.invalidBufferLength(expected: layout.expectedByteCount, actual: frame.count)
        }
        let GH = layout.gh / merge, GW = layout.gw / merge
        let gh = layout.gh, gw = layout.gw
        let FPP = featuresPerPatch, bpr = w * 3, fpatch = layout.framePatchCount
        var out = [Float](repeating: 0, count: layout.elementCount)
        for g in 0..<t {
            for bh in 0..<GH {
                for bw in 0..<GW {
                    for mh in 0..<merge {
                        for mw in 0..<merge {
                            let patchIdx = g * fpatch + ((bh * GW + bw) * merge + mh) * merge + mw
                            let base = patchIdx * FPP
                            for c in 0..<3 {
                                for ph in 0..<patch {
                                    let H = bh * (merge * patch) + mh * patch + ph
                                    let row = H * bpr
                                    for pw in 0..<patch {
                                        let W = bw * (merge * patch) + mw * patch + pw
                                        for tt in 0..<temporal {
                                            let v = Float(frames[g * temporal + tt][row + W * 3 + c]) / 127.5 - 1.0
                                            out[base + ((c * temporal + tt) * patch + ph) * patch + pw] = v
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return (out, t, gh, gw)
    }

    private func expectedRGBByteCount(h: Int, w: Int) throws -> Int {
        guard h > 0, w > 0, h.isMultiple(of: factor), w.isMultiple(of: factor) else {
            throw ImageError.invalidDimensions(h: h, w: w, factor: factor)
        }
        let (pixelCount, pixelCountOverflow) = h.multipliedReportingOverflow(by: w)
        let (byteCount, byteCountOverflow) = pixelCount.multipliedReportingOverflow(by: 3)
        guard !pixelCountOverflow, !byteCountOverflow else {
            throw ImageError.bufferSizeOverflow(h: h, w: w)
        }
        return byteCount
    }

    private func validatedRawTensorLayout(
        t: Int,
        h: Int,
        w: Int
    ) throws -> (
        expectedByteCount: Int,
        gh: Int,
        gw: Int,
        framePatchCount: Int,
        elementCount: Int
    ) {
        let expectedByteCount = try expectedRGBByteCount(h: h, w: w)
        let gh = h / patch
        let gw = w / patch
        let layout = try validatedTensorLayout(t: t, gh: gh, gw: gw)
        return (
            expectedByteCount,
            gh,
            gw,
            layout.framePatchCount,
            layout.elementCount
        )
    }

    private func validatedTensorLayout(
        t: Int,
        gh: Int,
        gw: Int
    ) throws -> (
        framePatchCount: Int,
        patchCount: Int,
        elementCount: Int
    ) {
        guard t > 0, gh > 0, gw > 0 else {
            throw ImageError.invalidTensorGrid(t: t, gh: gh, gw: gw)
        }
        guard gh.isMultiple(of: merge), gw.isMultiple(of: merge) else {
            throw ImageError.unmergeableTensorGrid(gh: gh, gw: gw, merge: merge)
        }
        let (framePatchCount, framePatchOverflow) = gh.multipliedReportingOverflow(by: gw)
        let (patchCount, patchCountOverflow) = t.multipliedReportingOverflow(by: framePatchCount)
        let (elementCount, elementCountOverflow) = patchCount.multipliedReportingOverflow(by: featuresPerPatch)
        guard !framePatchOverflow, !patchCountOverflow, !elementCountOverflow else {
            throw ImageError.tensorSizeOverflow(t: t, gh: gh, gw: gw)
        }
        return (framePatchCount, patchCount, elementCount)
    }

    func validatedTensorShape(
        pixelValuesCount: Int,
        t: Int,
        gh: Int,
        gw: Int
    ) throws -> (framePatchCount: Int, patchCount: Int, pixelDimension: Int) {
        let layout = try validatedTensorLayout(t: t, gh: gh, gw: gw)
        guard pixelValuesCount == layout.elementCount else {
            throw ImageError.invalidPixelValuesLength(expected: layout.elementCount, actual: pixelValuesCount)
        }
        return (layout.framePatchCount, layout.patchCount, featuresPerPatch)
    }

    /// Draw a CGImage resized to (w,h) into an RGBA8 buffer and return packed RGB (h*w*3) row-major.
    /// NOTE: CoreGraphics resampling is not bit-identical to PIL bicubic, so for images whose native
    /// size differs from (w,h) the pixel_values (hence embedding) carry a small resample-only error.
    public func resizedRGB(_ cgImage: CGImage, w: Int, h: Int) throws -> [UInt8] {
        let bpr = w * 4
        let space = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: bpr,
                                  space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw ImageError.context
        }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let dp = ctx.data, ctx.bytesPerRow == bpr else { throw ImageError.context }
        let buf = dp.bindMemory(to: UInt8.self, capacity: h * bpr)
        var rgb = [UInt8](repeating: 0, count: h * w * 3)
        for y in 0..<h {
            for x in 0..<w {
                let s = y * bpr + x * 4, d = (y * w + x) * 3
                rgb[d] = buf[s]; rgb[d + 1] = buf[s + 1]; rgb[d + 2] = buf[s + 2]
            }
        }
        return rgb
    }

    /// Normalized image as row-major `(H, W, 3)` — for parity debugging.
    public func normalizedHWC(_ cgImage: CGImage) throws -> [Float] {
        let S = size, bpr = S * 4
        let rgba = try drawRGBA(cgImage)
        var out = [Float](repeating: 0, count: S * S * 3)
        for h in 0..<S { for w in 0..<S { for c in 0..<3 {
            out[(h * S + w) * 3 + c] = Float(rgba[h * bpr + w * 4 + c]) / 127.5 - 1.0
        }}}
        return out
    }

    /// Draw into an SxS RGBA8 buffer and return a COPY (the CGContext owns the backing store,
    /// which is freed when ctx deinits — never return a pointer into it).
    private func drawRGBA(_ cgImage: CGImage) throws -> [UInt8] {
        let S = size, bytesPerRow = S * 4
        let space = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: S, height: S, bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow, space: space,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw ImageError.context
        }
        ctx.interpolationQuality = .high
        // No CTM flip: CGContext into this RGBA buffer already yields top-down rows matching
        // PIL's row order (verified by corner-pixel parity).
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: S, height: S))
        guard let dataPtr = ctx.data, ctx.bytesPerRow == bytesPerRow else { throw ImageError.context }
        let buf = dataPtr.bindMemory(to: UInt8.self, capacity: S * bytesPerRow)
        return Array(UnsafeBufferPointer(start: buf, count: S * bytesPerRow))
    }

    /// Flattened `pixel_values` (row-major `(numPatches, featuresPerPatch)`).
    public func pixelValues(_ cgImage: CGImage) throws -> [Float] {
        let S = size
        let bytesPerRow = S * 4
        let rgba = try drawRGBA(cgImage)

        let GH = S / (patch * merge), GW = S / (patch * merge)
        let FPP = featuresPerPatch
        var out = [Float](repeating: 0, count: numPatches * FPP)
        for gh in 0..<GH {
            for gw in 0..<GW {
                for mh in 0..<merge {
                    for mw in 0..<merge {
                        let patchIdx = ((gh * GW + gw) * merge + mh) * merge + mw
                        let base = patchIdx * FPP
                        for c in 0..<3 {
                            for ph in 0..<patch {
                                let H = gh * (merge * patch) + mh * patch + ph
                                let row = H * bytesPerRow
                                for pw in 0..<patch {
                                    let W = gw * (merge * patch) + mw * patch + pw
                                    let v = Float(rgba[row + W * 4 + c]) / 127.5 - 1.0
                                    for t in 0..<temporal {
                                        out[base + ((c * temporal + t) * patch + ph) * patch + pw] = v
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return out
    }
}
