import Foundation
import MLX
import MLXNN

// Port of gmberton/MegaLoc — "One Retrieval to Place Them All" (arXiv:2502.17237).
//
// MegaLoc(images [B, H, W, 3] NHWC) -> L2-normalised descriptors [B, feat_dim].
//
//   backbone  : DINOv2 ViT-B/14  -> (patches [B,Hp,Wp,768], cls [B,768])
//   aggregator: SALAD agg + Linear(16640 -> 8448)
//   l2norm    : final F.normalize over the descriptor dim

public final class MegaLoc: Module {
    public let config: MegaLocConfiguration

    @ModuleInfo(key: "backbone") var backbone: DINOv2Backbone
    @ModuleInfo(key: "aggregator") var aggregator: Aggregator

    public init(_ config: MegaLocConfiguration = .megaLoc) {
        self.config = config
        self._backbone.wrappedValue = DINOv2Backbone(config.backbone)
        self._aggregator.wrappedValue = Aggregator(config: config)
        super.init()
    }

    /// `images`: `[B, H, W, 3]` NHWC, ImageNet-normalised, `H`/`W` multiples of 14.
    /// Returns L2-normalised global descriptors `[B, feat_dim]`.
    public func callAsFunction(_ images: MLXArray) -> MLXArray {
        let (patches, cls) = backbone(images)
        let features = aggregator(patches: patches, cls: cls)
        return l2Normalize(features, axis: 1)
    }
}
