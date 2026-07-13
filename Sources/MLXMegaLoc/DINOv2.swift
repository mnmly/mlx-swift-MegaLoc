import Foundation
import MLX
import MLXNN

// Port of the DINOv2 ViT-B/14 backbone in gmberton/MegaLoc (megaloc_model.py).
//
// The Swift module tree is structured to mirror the checkpoint keys exactly, so
// weight loading needs no key renaming — only a dtype cast and an NCHW→NHWC
// transpose of the 4-D patch-embed conv weight:
//
//   backbone.patch_embed.proj.{weight,bias}
//   backbone.cls_token
//   backbone.pos_embed
//   backbone.blocks.<i>.{norm1,attn.qkv,attn.proj,ls1,norm2,mlp.fc1,mlp.fc2,ls2}
//   backbone.norm.{weight,bias}
//
// The bicubic pos-embed resampler is copied from the (PyTorch-parity-tested)
// implementation in mlx-swift-da3 (Sources/MLXDA3/Embeddings.swift).

// MARK: - Patch embedding (NHWC Conv2d → flatten spatial → tokens)

final class DinoPatchEmbedding: Module {
    private let patchSize: Int

    @ModuleInfo(key: "proj") private var proj: Conv2d

    init(patchSize: Int, inChannels: Int, embedDim: Int) {
        self.patchSize = patchSize
        self._proj.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: .init(patchSize),
            stride: .init(patchSize),
            bias: true
        )
    }

    /// `x`: `[B, H, W, 3]` NHWC → `[B, N, D]` where `N = (H/p)*(W/p)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let projected = proj(x)                       // [B, H/p, W/p, D]
        return projected.reshaped([b, -1, projected.dim(-1)])
    }
}

// MARK: - LayerScale (per-channel learnable gamma)

final class DinoLayerScale: Module {
    @ParameterInfo(key: "gamma") private var gamma: MLXArray

    init(_ dim: Int, initValue: Float = 1e-5) {
        self._gamma = ParameterInfo(
            wrappedValue: MLXArray.full([dim], values: MLXArray(initValue)),
            key: "gamma"
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

// MARK: - Multi-head self-attention (no RoPE / QK-norm)

final class DinoAttention: Module {
    private let numHeads: Int
    private let headDim: Int
    private let scale: Float

    @ModuleInfo(key: "qkv") private var qkv: Linear
    @ModuleInfo(key: "proj") private var proj: Linear

    init(dim: Int, numHeads: Int, qkvBias: Bool = true) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / Float(headDim).squareRoot()
        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: qkvBias)
        self._proj.wrappedValue = Linear(dim, dim, bias: true)
    }

    /// `x`: `[B, N, D]` → `[B, N, D]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)
        let d = x.dim(2)

        let qkvOut = qkv(x)
            .reshaped([b, n, 3, numHeads, headDim])
            .transposed(axes: [2, 0, 3, 1, 4])        // [3, B, H, N, head_dim]
        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]

        // Fused SDPA. Numerically equivalent (verified) to the reference's
        // explicit `softmax((q @ k^T) * scale) @ v`; on GPU both differ from a
        // CPU/PyTorch run only by Metal fp32 accumulation order (~1e-4/block).
        let out = scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil
        )                                              // [B, H, N, head_dim]
        return proj(out.transposed(axes: [0, 2, 1, 3]).reshaped([b, n, d]))
    }
}

// MARK: - MLP (fc1 → GELU → fc2)

final class DinoMLP: Module {
    @ModuleInfo(key: "fc1") private var fc1: Linear
    @ModuleInfo(key: "fc2") private var fc2: Linear
    private let act: GELU

    init(inFeatures: Int, hiddenFeatures: Int) {
        self._fc1.wrappedValue = Linear(inFeatures, hiddenFeatures, bias: true)
        self.act = GELU()                              // exact (erf) GELU, matches nn.GELU()
        self._fc2.wrappedValue = Linear(hiddenFeatures, inFeatures, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(act(fc1(x)))
    }
}

// MARK: - Transformer block (pre-norm, LayerScale)

final class DinoBlock: Module {
    @ModuleInfo(key: "norm1") private var norm1: LayerNorm
    @ModuleInfo(key: "attn") private var attn: DinoAttention
    @ModuleInfo(key: "ls1") private var ls1: DinoLayerScale
    @ModuleInfo(key: "norm2") private var norm2: LayerNorm
    @ModuleInfo(key: "mlp") private var mlp: DinoMLP
    @ModuleInfo(key: "ls2") private var ls2: DinoLayerScale

    init(dim: Int, numHeads: Int, mlpRatio: Float, lnEps: Float) {
        let hidden = Int(Float(dim) * mlpRatio)
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: lnEps)
        self._attn.wrappedValue = DinoAttention(dim: dim, numHeads: numHeads)
        self._ls1.wrappedValue = DinoLayerScale(dim)
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: lnEps)
        self._mlp.wrappedValue = DinoMLP(inFeatures: dim, hiddenFeatures: hidden)
        self._ls2.wrappedValue = DinoLayerScale(dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + ls1(attn(norm1(x)))
        h = h + ls2(mlp(norm2(h)))
        return h
    }
}

// MARK: - DINOv2 backbone

public final class DINOv2Backbone: Module {
    private let config: DINOv2Configuration

    @ModuleInfo(key: "patch_embed") private var patchEmbed: DinoPatchEmbedding
    @ParameterInfo(key: "cls_token") private var clsToken: MLXArray
    @ParameterInfo(key: "pos_embed") private var posEmbed: MLXArray
    @ModuleInfo(key: "blocks") private var blocks: [DinoBlock]
    @ModuleInfo(key: "norm") private var norm: LayerNorm

    public init(_ config: DINOv2Configuration = .vitB14) {
        self.config = config
        let numPatches = (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)

        self._patchEmbed.wrappedValue = DinoPatchEmbedding(
            patchSize: config.patchSize, inChannels: config.inChannels, embedDim: config.embedDim
        )
        self._clsToken = ParameterInfo(
            wrappedValue: MLXArray.zeros([1, 1, config.embedDim]), key: "cls_token"
        )
        self._posEmbed = ParameterInfo(
            wrappedValue: MLXArray.zeros([1, numPatches + 1, config.embedDim]), key: "pos_embed"
        )
        self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
            DinoBlock(
                dim: config.embedDim, numHeads: config.numHeads,
                mlpRatio: config.mlpRatio, lnEps: config.lnEps
            )
        }
        self._norm.wrappedValue = LayerNorm(dimensions: config.embedDim, eps: config.lnEps)
        super.init()
    }

    /// Interpolate positional encoding for input patch grids that differ from
    /// the pretrained `imageSize`. Mirrors `DINOv2.interpolate_pos_encoding`
    /// in megaloc_model.py, including its `(w0, h0)` output ordering.
    private func interpolatePosEncoding(numTokens: Int, height: Int, width: Int) -> MLXArray {
        let n = posEmbed.dim(1) - 1
        if numTokens - 1 == n && width == height {
            return posEmbed
        }
        let dim = posEmbed.dim(-1)
        let classPos = posEmbed[0..., ..<1]            // [1, 1, dim]
        let patchPos = posEmbed[0..., 1...]            // [1, N, dim]

        let w0 = width / config.patchSize
        let h0 = height / config.patchSize
        let m = Int(Float(n).squareRoot())

        var grid = patchPos.reshaped([1, m, m, dim])   // NHWC, axis1=H rows, axis2=W cols
        // Python: F.interpolate(scale_factor=(sx=(w0+off)/M, sy=(h0+off)/M)) on the
        // [1, dim, M, M] tensor, so the H(rows) axis maps to w0 and W(cols) to h0.
        grid = pytorchBicubicResample(grid, outH: w0, outW: h0, offset: config.interpolateOffset)
        let flat = grid.reshaped([1, w0 * h0, dim])
        return concatenated([classPos, flat], axis: 1)
    }

    /// Debug hook: returns intermediate token tensors (`[B, 1+N, D]`) for
    /// per-block parity bisection — `emb`, `block_0`…`block_{depth-1}`, `post_norm`.
    func debugStages(_ x: MLXArray) -> [String: MLXArray] {
        let b = x.dim(0)
        let h = x.dim(1)
        let w = x.dim(2)
        let patchTokens = patchEmbed(x)
        let cls = broadcast(clsToken, to: [b, 1, config.embedDim])
        var tokens = concatenated([cls, patchTokens], axis: 1)
        tokens = tokens + interpolatePosEncoding(numTokens: tokens.dim(1), height: h, width: w)
        var out: [String: MLXArray] = ["emb": tokens]
        for (i, block) in blocks.enumerated() {
            tokens = block(tokens)
            out["block_\(i)"] = tokens
        }
        out["post_norm"] = norm(tokens)
        return out
    }

    /// `x`: `[B, H, W, 3]` NHWC (ImageNet-normalised).
    /// Returns (`patches` `[B, H/p, W/p, D]`, `cls` `[B, D]`).
    public func callAsFunction(_ x: MLXArray) -> (patches: MLXArray, cls: MLXArray) {
        let b = x.dim(0)
        let h = x.dim(1)
        let w = x.dim(2)

        let patchTokens = patchEmbed(x)                              // [B, N, D]
        let cls = broadcast(clsToken, to: [b, 1, config.embedDim])   // [B, 1, D]
        var tokens = concatenated([cls, patchTokens], axis: 1)       // [B, 1+N, D]
        tokens = tokens + interpolatePosEncoding(numTokens: tokens.dim(1), height: h, width: w)

        for block in blocks { tokens = block(tokens) }
        tokens = norm(tokens)

        let clsOut = tokens[0..., 0]                                 // [B, D]
        let patchOut = tokens[0..., 1...]                            // [B, N, D]
        let pH = h / config.patchSize
        let pW = w / config.patchSize
        let patchesNHWC = patchOut.reshaped([b, pH, pW, config.embedDim])
        return (patchesNHWC, clsOut)
    }
}

// MARK: - PyTorch-parity bicubic resample (NHWC)
//
// Copied from mlx-swift-da3 (Sources/MLXDA3/Embeddings.swift), which verified it
// against `F.interpolate(mode="bicubic", align_corners=False, antialias=False)`.
// `a = -0.75` cubic kernel, half-pixel sample mapping, border-replicate padding.

func pytorchBicubicResample(_ x: MLXArray, outH: Int, outW: Int, offset: Float = 0.1) -> MLXArray {
    let inH = x.dim(1), inW = x.dim(2)
    let sy = (Float(outH) + offset) / Float(inH)
    let sx = (Float(outW) + offset) / Float(inW)

    var ys = [Float](repeating: 0, count: outH)
    for i in 0 ..< outH { ys[i] = (Float(i) + 0.5) / sy - 0.5 }
    var xs = [Float](repeating: 0, count: outW)
    for i in 0 ..< outW { xs[i] = (Float(i) + 0.5) / sx - 0.5 }

    func cubicWeights(_ t: Float) -> [Float] {
        let a: Float = -0.75
        let d_1 = 1 + t
        let d0 = t
        let d1 = 1 - t
        let d2 = 2 - t
        let w_1 = a * d_1 * d_1 * d_1 - 5 * a * d_1 * d_1 + 8 * a * d_1 - 4 * a
        let w0 = (a + 2) * d0 * d0 * d0 - (a + 3) * d0 * d0 + 1
        let w1 = (a + 2) * d1 * d1 * d1 - (a + 3) * d1 * d1 + 1
        let w2 = a * d2 * d2 * d2 - 5 * a * d2 * d2 + 8 * a * d2 - 4 * a
        return [w_1, w0, w1, w2]
    }

    func build(_ coords: [Float], inSize: Int) -> (idx: [[Int]], wts: [[Float]]) {
        var idx = [[Int]]()
        var wts = [[Float]]()
        for c in coords {
            let f = floor(c)
            let t = c - f
            let base = Int(f)
            let ii = [base - 1, base, base + 1, base + 2].map { max(0, min(inSize - 1, $0)) }
            idx.append(ii)
            wts.append(cubicWeights(t))
        }
        return (idx, wts)
    }

    let (yIdx, yWts) = build(ys, inSize: inH)
    let (xIdx, xWts) = build(xs, inSize: inW)

    var hRows = [MLXArray]()
    for j in 0 ..< outH {
        let i0 = yIdx[j][0], i1 = yIdx[j][1], i2 = yIdx[j][2], i3 = yIdx[j][3]
        let w0 = yWts[j][0], w1 = yWts[j][1], w2 = yWts[j][2], w3 = yWts[j][3]
        let row = x[0..., i0 ..< (i0 + 1), 0..., 0...] * MLXArray(w0)
            + x[0..., i1 ..< (i1 + 1), 0..., 0...] * MLXArray(w1)
            + x[0..., i2 ..< (i2 + 1), 0..., 0...] * MLXArray(w2)
            + x[0..., i3 ..< (i3 + 1), 0..., 0...] * MLXArray(w3)
        hRows.append(row)
    }
    let hStack = concatenated(hRows, axis: 1)          // [B, outH, inW, C]

    var wCols = [MLXArray]()
    for j in 0 ..< outW {
        let i0 = xIdx[j][0], i1 = xIdx[j][1], i2 = xIdx[j][2], i3 = xIdx[j][3]
        let w0 = xWts[j][0], w1 = xWts[j][1], w2 = xWts[j][2], w3 = xWts[j][3]
        let col = hStack[0..., 0..., i0 ..< (i0 + 1), 0...] * MLXArray(w0)
            + hStack[0..., 0..., i1 ..< (i1 + 1), 0...] * MLXArray(w1)
            + hStack[0..., 0..., i2 ..< (i2 + 1), 0...] * MLXArray(w2)
            + hStack[0..., 0..., i3 ..< (i3 + 1), 0...] * MLXArray(w3)
        wCols.append(col)
    }
    return concatenated(wCols, axis: 2)                // [B, outH, outW, C]
}
