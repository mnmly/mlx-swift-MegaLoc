import Foundation

// Port of the config used by gmberton/MegaLoc (megaloc_model.py + HF config.json).
//
// MegaLoc = DINOv2 ViT-B/14 backbone + SALAD optimal-transport aggregator + a
// final linear projection, producing an L2-normalised global descriptor.

/// Backbone (DINOv2 ViT-B/14) hyper-parameters.
public struct DINOv2Configuration: Codable, Sendable {
    public var imageSize: Int
    public var patchSize: Int
    public var inChannels: Int
    public var embedDim: Int
    public var depth: Int
    public var numHeads: Int
    public var mlpRatio: Float
    /// LayerNorm epsilon (DINOv2 uses 1e-6 for block + final norms).
    public var lnEps: Float
    /// DINOv2 positional-encoding interpolation offset (0.1).
    public var interpolateOffset: Float

    public init(
        imageSize: Int = 518,
        patchSize: Int = 14,
        inChannels: Int = 3,
        embedDim: Int = 768,
        depth: Int = 12,
        numHeads: Int = 12,
        mlpRatio: Float = 4.0,
        lnEps: Float = 1e-6,
        interpolateOffset: Float = 0.1
    ) {
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.embedDim = embedDim
        self.depth = depth
        self.numHeads = numHeads
        self.mlpRatio = mlpRatio
        self.lnEps = lnEps
        self.interpolateOffset = interpolateOffset
    }

    public static let vitB14 = DINOv2Configuration()
}

/// Top-level MegaLoc hyper-parameters.
public struct MegaLocConfiguration: Codable, Sendable {
    public var backbone: DINOv2Configuration
    public var featDim: Int
    public var numClusters: Int
    public var clusterDim: Int
    public var tokenDim: Int
    public var mlpDim: Int

    /// SALAD aggregator output dim (`num_clusters * cluster_dim + token_dim`),
    /// which is the input dim of the final linear head.
    public var saladOutDim: Int { numClusters * clusterDim + tokenDim }

    public init(
        backbone: DINOv2Configuration = .vitB14,
        featDim: Int = 8448,
        numClusters: Int = 64,
        clusterDim: Int = 256,
        tokenDim: Int = 256,
        mlpDim: Int = 512
    ) {
        self.backbone = backbone
        self.featDim = featDim
        self.numClusters = numClusters
        self.clusterDim = clusterDim
        self.tokenDim = tokenDim
        self.mlpDim = mlpDim
    }

    /// The pretrained gmberton/MegaLoc configuration.
    public static let megaLoc = MegaLocConfiguration()
}

/// ImageNet normalisation constants (DINOv2 preprocessing).
public enum ImageNetNorm {
    public static let mean: [Float] = [0.485, 0.456, 0.406]
    public static let std: [Float] = [0.229, 0.224, 0.225]
}
