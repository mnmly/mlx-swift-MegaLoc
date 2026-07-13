import Foundation
import MLX
import MLXNN

// Weight loading for MegaLoc.
//
// The Swift module tree is structured to mirror the checkpoint's key layout
// exactly (see DINOv2.swift / FeatureAggregator.swift), so loading needs no key
// renaming. The only transforms are:
//   1. Cast every tensor to the target `dtype`.
//   2. Transpose 4-D conv weights from PyTorch NCHW (O,I,kH,kW) to MLX NHWC
//      (O,kH,kW,I). These are exactly the 1x1 aggregator convs and the
//      patch-embed conv — all other `.weight` tensors are 2-D (Linear).
//
// `verify: [.noUnusedKeys]` catches any structural mismatch loudly.

extension MegaLoc {

    /// Load a MegaLoc safetensors checkpoint (`gberton/MegaLoc/model.safetensors`).
    /// - Parameters:
    ///   - url: local path to `model.safetensors`.
    ///   - config: model configuration (defaults to the pretrained one).
    ///   - dtype: compute dtype for the weights (default `.float32` for parity).
    public static func load(
        weights url: URL,
        config: MegaLocConfiguration = .megaLoc,
        dtype: DType = .float32
    ) throws -> MegaLoc {
        let model = MegaLoc(config)
        let raw = try loadArrays(url: url)

        var params: [(String, MLXArray)] = []
        params.reserveCapacity(raw.count)
        for (key, value) in raw {
            var v = value.asType(dtype)
            // PyTorch conv weight (4-D) -> MLX NHWC layout.
            if key.hasSuffix(".weight"), v.ndim == 4 {
                v = v.transposed(axes: [0, 2, 3, 1])
            }
            params.append((key, v))
        }

        try model.update(
            parameters: ModuleParameters.unflattened(params),
            verify: [.noUnusedKeys]
        )
        eval(model)
        return model
    }
}
