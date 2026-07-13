import CoreGraphics
import Foundation
import MLX

// The shared library-side driver both the CLI and the example GUI consume.
//
// Per the swift-cli-gui-shared-driver pattern: ALL non-presentation work
// (model load, preprocessing, forward pass, descriptor extraction) lives here.
// The expensive network forward (`embed`) is deliberately separated from the
// cheap post-processing (`similarity` / `rank`), so a GUI can re-rank cached
// descriptors instantly without re-running the model.

/// An L2-normalised global descriptor. A plain `Sendable` value type — it holds
/// no `MLXArray`, so it crosses actor / task boundaries freely.
public struct MegaLocDescriptor: Sendable, Hashable, Codable {
    public let values: [Float]
    public init(_ values: [Float]) { self.values = values }
    public var dimension: Int { values.count }
}

/// A ranked retrieval result: index into the database + cosine similarity.
public struct MegaLocMatch: Sendable, Hashable {
    public let index: Int
    public let similarity: Float
}

/// Loads MegaLoc once and runs inference. Marked `@unchecked Sendable` under a
/// documented single-writer invariant: inference is serialised by the caller
/// (the CLI runs sequentially; the GUI funnels every `embed` through one
/// detached task at a time). The wrapped `MLXArray` graph is never mutated
/// concurrently.
public final class MegaLocSession: @unchecked Sendable {
    public let config: MegaLocConfiguration
    /// Square input side (multiple of 14) used for every image.
    public let imageSize: Int
    private let model: MegaLoc

    private init(model: MegaLoc, config: MegaLocConfiguration, imageSize: Int) {
        self.model = model
        self.config = config
        self.imageSize = imageSize
    }

    /// Load a MegaLoc checkpoint (`model.safetensors`).
    public static func load(
        weights url: URL,
        config: MegaLocConfiguration = .megaLoc,
        imageSize: Int = MegaLocPreprocess.defaultImageSize,
        dtype: DType = .float32
    ) throws -> MegaLocSession {
        let model = try MegaLoc.load(weights: url, config: config, dtype: dtype)
        return MegaLocSession(model: model, config: config, imageSize: imageSize)
    }

    // MARK: - Expensive: the network forward

    /// Embed a preprocessed NHWC batch `[B, H, W, 3]` and return `B` descriptors.
    public func embed(batch: MLXArray) -> [MegaLocDescriptor] {
        let out = model(batch)                      // [B, feat_dim], L2-normalised
        eval(out)
        let b = out.dim(0)
        let d = out.dim(1)
        let flat = out.asType(.float32).asArray(Float.self)
        return (0 ..< b).map { i in
            MegaLocDescriptor(Array(flat[(i * d) ..< ((i + 1) * d)]))
        }
    }

    /// Embed a single image.
    public func embed(image: CGImage) -> MegaLocDescriptor {
        let x = MegaLocPreprocess.imageToNHWC(image, size: imageSize)
        return embed(batch: x)[0]
    }

    /// Embed several images. Batched into one forward when the images share the
    /// preprocessed size (they always do here — fixed square input).
    public func embed(images: [CGImage]) -> [MegaLocDescriptor] {
        guard !images.isEmpty else { return [] }
        let arrays = images.map { MegaLocPreprocess.imageToNHWC($0, size: imageSize) }
        let batch = arrays.count == 1 ? arrays[0] : concatenated(arrays, axis: 0)
        return embed(batch: batch)
    }

    /// Embed image files by URL (skips any that fail to decode; returns the
    /// descriptors paired with the URLs that succeeded).
    public func embed(urls: [URL]) -> [(url: URL, descriptor: MegaLocDescriptor)] {
        var loaded: [(URL, CGImage)] = []
        for url in urls {
            if let img = MegaLocPreprocess.loadCGImage(url) { loaded.append((url, img)) }
        }
        let descriptors = embed(images: loaded.map { $0.1 })
        return zip(loaded.map { $0.0 }, descriptors).map { ($0, $1) }
    }

    // MARK: - Cheap: post-processing (no model involved)

    /// Cosine similarity. Descriptors are L2-normalised, so this is a dot product.
    public static func similarity(_ a: MegaLocDescriptor, _ b: MegaLocDescriptor) -> Float {
        precondition(a.values.count == b.values.count, "descriptor dimension mismatch")
        var dot: Float = 0
        for i in 0 ..< a.values.count { dot += a.values[i] * b.values[i] }
        return dot
    }

    /// Rank a database of descriptors against a query, best first.
    public static func rank(
        query: MegaLocDescriptor, database: [MegaLocDescriptor], topK: Int? = nil
    ) -> [MegaLocMatch] {
        var matches = database.enumerated().map {
            MegaLocMatch(index: $0.offset, similarity: similarity(query, $0.element))
        }
        matches.sort { $0.similarity > $1.similarity }
        if let k = topK { return Array(matches.prefix(k)) }
        return matches
    }
}
