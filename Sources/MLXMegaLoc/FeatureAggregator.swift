import Foundation
import MLX
import MLXNN

// Port of the SALAD optimal-transport aggregator in gmberton/MegaLoc
// (megaloc_model.py: FeatureAggregator + Aggregator).
//
// The three MLPs are stored as `[UnaryLayer]` arrays whose indices match the
// PyTorch `nn.Sequential` indices, so the checkpoint keys map directly:
//
//   aggregator.agg.token_features.{0,2}.{weight,bias}   (Linear, ReLU at idx 1)
//   aggregator.agg.cluster_features.{0,3}.{weight,bias}  (Conv2d, Identity+ReLU at 1,2)
//   aggregator.agg.score.{0,3}.{weight,bias}             (Conv2d, Identity+ReLU at 1,2)
//   aggregator.agg.dust_bin
//   aggregator.linear.{weight,bias}
//
// The Sinkhorn solver mirrors log_otp_solver / get_matching_probs, adapted from
// the parity-tested implementation in mlx-swift-da3.

// MARK: - SALAD feature aggregator

public final class FeatureAggregator: Module {
    public let numChannels: Int
    public let numClusters: Int
    public let clusterDim: Int
    public let tokenDim: Int
    public let mlpDim: Int

    @ModuleInfo(key: "token_features") private var tokenFeatures: [UnaryLayer]
    @ModuleInfo(key: "cluster_features") private var clusterFeatures: [UnaryLayer]
    @ModuleInfo(key: "score") private var score: [UnaryLayer]
    @ParameterInfo(key: "dust_bin") private var dustBin: MLXArray

    public init(
        numChannels: Int = 768,
        numClusters: Int = 64,
        clusterDim: Int = 256,
        tokenDim: Int = 256,
        mlpDim: Int = 512
    ) {
        self.numChannels = numChannels
        self.numClusters = numClusters
        self.clusterDim = clusterDim
        self.tokenDim = tokenDim
        self.mlpDim = mlpDim

        // token_features: Linear -> ReLU -> Linear  (Sequential idx 0, 1, 2)
        self._tokenFeatures.wrappedValue = [
            Linear(numChannels, mlpDim, bias: true),
            ReLU(),
            Linear(mlpDim, tokenDim, bias: true),
        ]
        // cluster_features: Conv2d -> Dropout(Identity) -> ReLU -> Conv2d  (idx 0..3)
        self._clusterFeatures.wrappedValue = [
            Self.conv1x1(numChannels, mlpDim),
            Identity(),
            ReLU(),
            Self.conv1x1(mlpDim, clusterDim),
        ]
        // score: Conv2d -> Dropout(Identity) -> ReLU -> Conv2d  (idx 0..3)
        self._score.wrappedValue = [
            Self.conv1x1(numChannels, mlpDim),
            Identity(),
            ReLU(),
            Self.conv1x1(mlpDim, numClusters),
        ]
        self._dustBin = ParameterInfo(wrappedValue: MLXArray(Float(1.0)), key: "dust_bin")
        super.init()
    }

    private static func conv1x1(_ inCh: Int, _ outCh: Int) -> Conv2d {
        Conv2d(
            inputChannels: inCh, outputChannels: outCh,
            kernelSize: .init(1), stride: .init(1), padding: .init(0), bias: true
        )
    }

    /// Sinkhorn matrix scaling (`log_otp_solver` + `get_matching_probs`).
    /// `s`: `[B, m, n]` score matrix. Returns log assignment `[B, m+1, n]`.
    private static func getMatchingProbs(_ s: MLXArray, dustbin: MLXArray, numIters: Int) -> MLXArray {
        let b = s.dim(0), m = s.dim(1), n = s.dim(2)

        // Augment with a dustbin row filled with the (scalar) dustbin score.
        let dustRow = broadcast(dustbin.reshaped([1, 1, 1]), to: [b, 1, n])
        let mAug = concatenated([s, dustRow], axis: 1)      // [B, m+1, n]

        // Normalised source/target log-weights (constants for a given n, m).
        let norm = -log(Float(n + m))
        var logAValues = [Float](repeating: norm, count: m + 1)
        logAValues[m] += log(Float(n - m))
        let logA = broadcast(MLXArray(logAValues).reshaped([1, m + 1]), to: [b, m + 1])
        let logB = broadcast(
            MLXArray([Float](repeating: norm, count: n)).reshaped([1, n]), to: [b, n]
        )

        var u = MLXArray.zeros([b, m + 1])
        var v = MLXArray.zeros([b, n])
        for _ in 0 ..< numIters {
            u = logA - logSumExp(mAug + v.expandedDimensions(axis: 1), axis: 2)
            v = logB - logSumExp(mAug + u.expandedDimensions(axis: 2), axis: 1)
        }
        let logP = mAug + u.expandedDimensions(axis: 2) + v.expandedDimensions(axis: 1)
        return logP - norm
    }

    private func runSequential(_ layers: [UnaryLayer], _ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h) }
        return h
    }

    /// `patches`: `[B, Hp, Wp, C]` NHWC, `cls`: `[B, C]`.
    /// Returns the L2-normalised SALAD descriptor `[B, num_clusters*cluster_dim + token_dim]`.
    public func callAsFunction(patches: MLXArray, cls: MLXArray) -> MLXArray {
        let b = patches.dim(0)
        let hw = patches.dim(1) * patches.dim(2)

        // Local cluster features: [B, Hp, Wp, l] -> [B, l, n]  (matches python flatten(2)).
        let fNHWC = runSequential(clusterFeatures, patches)
        let l = fNHWC.dim(-1)
        let f = fNHWC.reshaped([b, hw, l]).transposed(axes: [0, 2, 1])   // [B, l, n]

        // Score / assignment logits: [B, Hp, Wp, m] -> [B, m, n].
        let pNHWC = runSequential(score, patches)
        let m = pNHWC.dim(-1)
        let p = pNHWC.reshaped([b, hw, m]).transposed(axes: [0, 2, 1])   // [B, m, n]

        // Global scene token: [B, C] -> [B, token_dim].
        let t = runSequential(tokenFeatures, cls)

        // Sinkhorn assignment, drop the dustbin row.
        var assignment = FeatureAggregator.getMatchingProbs(p, dustbin: dustBin, numIters: 3)
        assignment = exp(assignment)
        assignment = assignment[0..., ..<m, 0...]                        // [B, m, n]

        // Weighted sum over spatial positions: sum_n f[b,l,n] * p[b,m,n] -> [B, l, m].
        //   python: (f.unsqueeze(2) * p.unsqueeze(1)).sum(dim=-1)  ==  f @ p^T
        let weighted = matmul(f, assignment.transposed(axes: [0, 2, 1]))  // [B, l, m]
        let weightedNorm = l2Normalize(weighted, axis: 1)                 // normalize over l
        let clusterVec = weightedNorm.reshaped([b, l * m])                // [B, l*m]

        let tokenVec = l2Normalize(t, axis: -1)                           // [B, token_dim]

        let combined = concatenated([tokenVec, clusterVec], axis: -1)     // [B, token_dim + l*m]
        return l2Normalize(combined, axis: -1)
    }
}

// MARK: - Aggregator (SALAD + final linear projection)

public final class Aggregator: Module {
    @ModuleInfo(key: "agg") var agg: FeatureAggregator
    @ModuleInfo(key: "linear") var linear: Linear

    public init(config: MegaLocConfiguration) {
        self._agg.wrappedValue = FeatureAggregator(
            numChannels: config.backbone.embedDim,
            numClusters: config.numClusters,
            clusterDim: config.clusterDim,
            tokenDim: config.tokenDim,
            mlpDim: config.mlpDim
        )
        self._linear.wrappedValue = Linear(config.saladOutDim, config.featDim, bias: true)
        super.init()
    }

    /// `patches`: `[B, Hp, Wp, C]`, `cls`: `[B, C]` → `[B, feat_dim]` (un-normalised).
    public func callAsFunction(patches: MLXArray, cls: MLXArray) -> MLXArray {
        linear(agg(patches: patches, cls: cls))
    }
}

// MARK: - L2 normalisation (F.normalize, eps 1e-12)

func l2Normalize(_ x: MLXArray, axis: Int) -> MLXArray {
    let norm = sqrt((x * x).sum(axis: axis, keepDims: true))
    return x / maximum(norm, MLXArray(Float(1e-12)))
}
