import CoreGraphics
import Foundation
import ImageIO
import MLX

#if canImport(UniformTypeIdentifiers)
import UniformTypeIdentifiers
#endif

// Image preprocessing for MegaLoc: CGImage -> ImageNet-normalised NHWC MLXArray.
//
// MegaLoc does not normalise internally (see megaloc_model.py), so this is where
// the DINOv2 ImageNet mean/std is applied. Images are resized (squashed) to a
// square whose side is a multiple of the patch size (14).

public enum MegaLocPreprocess {

    /// Default square input side used by the pipeline (multiple of 14).
    public static let defaultImageSize = 322

    /// Resize `image` to `size`×`size`, convert to RGB float in `[0,1]`,
    /// apply ImageNet normalisation, and return an NHWC `[1, size, size, 3]` array.
    public static func imageToNHWC(_ image: CGImage, size: Int = defaultImageSize) -> MLXArray {
        precondition(size % 14 == 0, "size must be a multiple of the patch size (14)")

        let width = size, height = size
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var buffer = [UInt8](repeating: 0, count: bytesPerRow * height)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        buffer.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(
                data: raw.baseAddress,
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return }
            ctx.interpolationQuality = .high
            // CoreGraphics origin is bottom-left; drawing straight into the rect
            // matches how CGImage pixels map, and both the query and database go
            // through the same path so retrieval is consistent.
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }

        let mean = ImageNetNorm.mean
        let std = ImageNetNorm.std
        var floats = [Float](repeating: 0, count: width * height * 3)
        var di = 0
        var si = 0
        for _ in 0 ..< (width * height) {
            let r = Float(buffer[si]) / 255.0
            let g = Float(buffer[si + 1]) / 255.0
            let b = Float(buffer[si + 2]) / 255.0
            floats[di] = (r - mean[0]) / std[0]
            floats[di + 1] = (g - mean[1]) / std[1]
            floats[di + 2] = (b - mean[2]) / std[2]
            di += 3
            si += 4
        }
        return MLXArray(floats, [1, height, width, 3])
    }

    /// Load a `CGImage` from a file URL (PNG/JPEG/HEIC/… via ImageIO).
    public static func loadCGImage(_ url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { return nil }
        return img
    }
}
