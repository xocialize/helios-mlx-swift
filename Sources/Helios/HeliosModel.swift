// HeliosModel — the Helios-Distilled DiT. 1:1 with PR#21 `transformer.py`.
// Backbone is Wan2.2-A14B-shaped; RoPE tables/compute REUSE wan-core's public
// `ropeParams`/`ropePrecomputeCosSin` (Helios per-axis freqs == wan-core's, verified).
// This file implements the S1 NO-HISTORY forward (patchify → embeds → blocks →
// output); the AR history path (multi-scale patchify, frame-offset/downsampled
// RoPE, restricted attn) lands in S2/S3.

import Foundation
import MLX
import MLXNN
import WanCore

/// Non-Module buffers (RoPE freqs + sinusoidal inv_freq). Kept OUT of the Module
/// so reflection doesn't register them as bogus parameters (the wan-core rule).
final class HeliosBuffers: @unchecked Sendable {
    let ropeFreqs: MLXArray   // [1024, 64, 2] = concat[ropeParams(44),(42),(42)]
    let invFreq: MLXArray     // [freq_dim/2]

    init(_ config: HeliosConfig) {
        let (dt, dh, dw) = (config.ropeDim[0], config.ropeDim[1], config.ropeDim[2])
        let theta = Double(config.ropeTheta)
        ropeFreqs = concatenated(
            [ropeParams(1024, dt, theta: theta),
             ropeParams(1024, dh, theta: theta),
             ropeParams(1024, dw, theta: theta)], axis: 1)
        let half = config.freqDim / 2
        let exps = (0..<half).map { -Double($0) / Double(half) }
        invFreq = MLXArray(exps.map { Float(pow(10000.0, $0)) })
    }
}

/// Output head: parameterless norm + AdaLN table (`head.modulation`) + projection (`head.head`).
final class HeliosHead: Module, @unchecked Sendable {
    let outputNorm: HeliosLayerNorm
    @ModuleInfo(key: "head") var head: Linear
    @ParameterInfo(key: "modulation") var modulation: MLXArray  // [1, 2, dim]

    init(dim: Int, outDim: Int, patchSize: [Int], eps: Float) {
        self.outputNorm = HeliosLayerNorm(dim, eps)
        let projDim = patchSize.reduce(1, *) * outDim
        self._head.wrappedValue = Linear(dim, projDim)
        self._modulation.wrappedValue = MLXRandom.normal([1, 2, dim]) * pow(Float(dim), -0.5)
    }

    /// `temb`: [B, dim] base time embedding. x: [B, L, dim] → [B, L, projDim].
    func callAsFunction(_ x: MLXArray, _ temb: MLXArray) -> MLXArray {
        let l = x.dim(1)
        let wDtype = linearDtype(head)
        let tembExp = broadcast(temb[0..., .newAxis, 0...], to: [x.dim(0), l, x.dim(2)])
        let modOut = (modulation.expandedDimensions(axis: 0) + tembExp.expandedDimensions(axis: 2)).asType(wDtype)
        let shift = modOut[0..., 0..., 0]
        let scale = modOut[0..., 0..., 1]
        let out = (outputNorm(x) * (1 + scale) + shift).asType(wDtype)
        return head(out)
    }
}

public final class HeliosModel: Module, @unchecked Sendable {
    public let config: HeliosConfig
    let dim: Int
    let patchSize: [Int]

    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "patch_short") var patchShort: Linear
    @ModuleInfo(key: "patch_mid") var patchMid: Linear
    @ModuleInfo(key: "patch_long") var patchLong: Linear
    @ModuleInfo(key: "text_embedding_0") var textEmbedding0: Linear
    @ModuleInfo(key: "text_embedding_1") var textEmbedding1: Linear
    @ModuleInfo(key: "time_embedding_0") var timeEmbedding0: Linear
    @ModuleInfo(key: "time_embedding_1") var timeEmbedding1: Linear
    @ModuleInfo(key: "time_projection") var timeProjection: Linear
    @ModuleInfo(key: "blocks") var blocks: [HeliosTransformerBlock]
    @ModuleInfo(key: "head") var head: HeliosHead

    let buffers: HeliosBuffers

    public init(_ config: HeliosConfig) {
        self.config = config
        self.dim = config.dim
        self.patchSize = config.patchSize
        let dim = config.dim
        let patchDim = config.inDim * config.patchSize.reduce(1, *)
        self._patchEmbedding.wrappedValue = Linear(patchDim, dim)
        self._patchShort.wrappedValue = Linear(config.inDim * 1 * 2 * 2, dim)
        self._patchMid.wrappedValue = Linear(config.inDim * 2 * 4 * 4, dim)
        self._patchLong.wrappedValue = Linear(config.inDim * 4 * 8 * 8, dim)
        self._textEmbedding0.wrappedValue = Linear(config.textDim, dim)
        self._textEmbedding1.wrappedValue = Linear(dim, dim)
        self._timeEmbedding0.wrappedValue = Linear(config.freqDim, dim)
        self._timeEmbedding1.wrappedValue = Linear(dim, dim)
        self._timeProjection.wrappedValue = Linear(dim, dim * 6)
        self._blocks.wrappedValue = (0..<config.numLayers).map { _ in
            HeliosTransformerBlock(
                dim: dim, ffnDim: config.ffnDim, numHeads: config.numHeads,
                qkNorm: config.qkNorm, crossAttnNorm: config.crossAttnNorm,
                eps: config.eps, restrictSelfAttn: false)
        }
        self._head.wrappedValue = HeliosHead(
            dim: dim, outDim: config.outDim, patchSize: config.patchSize, eps: config.eps)
        self.buffers = HeliosBuffers(config)
        super.init()
    }

    /// Patchify current latents [C,F,H,W] → ([1, L, dim], grid (f',h',w')).
    func patchify(_ x: MLXArray) -> (MLXArray, (Int, Int, Int)) {
        let c = x.dim(0)
        let (pt, ph, pw) = (patchSize[0], patchSize[1], patchSize[2])
        let f = (x.dim(1) / pt) * pt
        let h = (x.dim(2) / ph) * ph
        let w = (x.dim(3) / pw) * pw
        var xc = x[0..., ..<f, ..<h, ..<w]
        let (fo, ho, wo) = (f / pt, h / ph, w / pw)
        xc = xc.reshaped(c, fo, pt, ho, ph, wo, pw)
            .transposed(1, 3, 5, 0, 2, 4, 6)
            .reshaped(fo * ho * wo, -1)
        let patches = patchEmbedding(xc).asType(linearDtype(patchEmbedding))
        return (patches.expandedDimensions(axis: 0), (fo, ho, wo))
    }

    /// Reconstruct [C,F,H,W] from patch output [B, L, prod(patch)*outDim].
    func unpatchify(_ x: MLXArray, _ grid: (Int, Int, Int)) -> MLXArray {
        let cOut = config.outDim
        let (pt, ph, pw) = (patchSize[0], patchSize[1], patchSize[2])
        let (f, h, w) = grid
        let seqLen = f * h * w
        return x[0, ..<seqLen]
            .reshaped(f, h, w, pt, ph, pw, cOut)
            .transposed(6, 0, 3, 1, 4, 2, 5)
            .reshaped(cOut, f * pt, h * ph, w * pw)
    }

    /// Precompute text embeddings: pad to text_len, project (Linear→GELU→Linear).
    public func embedText(_ context: [MLXArray]) -> MLXArray {
        let modelDtype = linearDtype(patchEmbedding)
        let padded = context.map { ctx -> MLXArray in
            let padLen = config.textLen - ctx.dim(0)
            guard padLen > 0 else { return ctx }
            return concatenated([ctx, MLXArray.zeros([padLen, ctx.dim(1)], dtype: ctx.dtype)], axis: 0)
        }
        let batch = stacked(padded)
        return textEmbedding1(geluApproximate(textEmbedding0(batch))).asType(modelDtype)
    }

    func crossKVCaches(_ context: MLXArray) -> [(MLXArray, MLXArray)] {
        blocks.map { $0.crossAttn.prepareKV(context) }
    }

    /// S1 no-history forward. latents [C,F,H,W], timestep [1], encoderHiddenStates [B,text_len,dim].
    public func callAsFunction(
        _ latents: MLXArray, timestep: MLXArray, encoderHiddenStates: MLXArray,
        crossKVCaches: [(MLXArray, MLXArray)]? = nil
    ) -> MLXArray {
        var (hidden, grid) = patchify(latents)
        let L = hidden.dim(1)

        // Current-chunk RoPE — wan-core public reuse (frame_indices = arange(f)).
        let ropeCosSin = ropePrecomputeCosSin(
            gridSizes: [grid], freqs: buffers.ropeFreqs, dtype: hidden.dtype)

        // Time embedding + 6-vec modulation, broadcast to per-token [1,L,6,dim].
        var tEmb = timestep.asType(.float32) * buffers.invFreq
        tEmb = concatenated([cos(tEmb), sin(tEmb)], axis: -1)
        if tEmb.ndim == 1 { tEmb = tEmb.expandedDimensions(axis: 0) }
        let temb = timeEmbedding1(silu(timeEmbedding0(tEmb)))
        let tproj = timeProjection(silu(temb)).reshaped(1, 6, -1)
        let tprojExp = broadcast(
            tproj[0..., 0..., .newAxis, 0...], to: [1, 6, L, dim]
        ).transposed(0, 2, 1, 3)  // [1, L, 6, dim]

        for (i, block) in blocks.enumerated() {
            hidden = block(
                hidden, context: encoderHiddenStates, tProj: tprojExp,
                ropeCosSin: ropeCosSin, originalContextLength: L,
                crossKVCache: crossKVCaches?[i])
        }

        let hiddenOut = head(hidden[0..., (hidden.dim(1) - L)...], temb)
        return unpatchify(hiddenOut, grid)
    }
}
