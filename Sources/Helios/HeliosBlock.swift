// HeliosTransformerBlock — 1:1 with PR#21 `transformer.py`. 6-vector modulation
// (scale_shift_table → canonical `modulation`), fp32 residual accumulation, and
// the affine cross-attn norm (oracle `norm2` → canonical `norm3`). The self-attn
// and ffn norms are parameterless (no checkpoint keys). guidance_cross_attn is
// always on: with history, only CURRENT tokens get cross-attn.

import Foundation
import MLX
import MLXNN
import WanCore

final class HeliosTransformerBlock: Module, @unchecked Sendable {
    let dim: Int
    // parameterless norms (no keys); the affine cross-attn norm carries keys (norm3).
    let norm1: HeliosLayerNorm
    let normFfn: HeliosLayerNorm
    @ModuleInfo(key: "norm3") var norm3: HeliosLayerNorm?  // affine cross-attn norm (oracle norm2)
    @ModuleInfo(key: "self_attn") var selfAttn: HeliosSelfAttention
    @ModuleInfo(key: "cross_attn") var crossAttn: HeliosCrossAttention
    @ModuleInfo(key: "ffn") var ffn: HeliosFFN
    @ParameterInfo(key: "modulation") var modulation: MLXArray  // [1, 6, dim]

    init(dim: Int, ffnDim: Int, numHeads: Int, qkNorm: Bool = true,
         crossAttnNorm: Bool = true, eps: Float = 1e-6, restrictSelfAttn: Bool = false) {
        self.dim = dim
        self.norm1 = HeliosLayerNorm(dim, eps)
        self.normFfn = HeliosLayerNorm(dim, eps)
        self._norm3.wrappedValue = crossAttnNorm ? HeliosLayerNorm(dim, eps, elementwiseAffine: true) : nil
        self._selfAttn.wrappedValue = HeliosSelfAttention(
            dim, numHeads, qkNorm: qkNorm, eps: eps, restrictSelfAttn: restrictSelfAttn)
        self._crossAttn.wrappedValue = HeliosCrossAttention(dim, numHeads, qkNorm: qkNorm, eps: eps)
        self._ffn.wrappedValue = HeliosFFN(dim, ffnDim)
        self._modulation.wrappedValue = MLXRandom.normal([1, 6, dim]) * pow(Float(dim), -0.5)
    }

    /// `tProj`: per-token modulation [B, L, 6, dim]. `ropeCosSin`: full-sequence
    /// (history+current) cos/sin. `originalContextLength`: current-token count.
    func callAsFunction(
        _ x: MLXArray, context: MLXArray, tProj: MLXArray,
        ropeCosSin: (MLXArray, MLXArray)?, originalContextLength: Int,
        crossKVCache: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        var x = x
        let s = x.dim(1)
        let historyLen = s - originalContextLength
        let wDtype = linearDtype(ffn.fc1)

        // 6-vector modulation: scale_shift_table[None] + tProj  → [B,L,6,dim]
        let mod = (modulation.expandedDimensions(axis: 1) + tProj.asType(.float32)).asType(wDtype)
        let shiftMsa = mod[0..., 0..., 0]
        let scaleMsa = mod[0..., 0..., 1]
        let gateMsa = mod[0..., 0..., 2]
        let cShift = mod[0..., 0..., 3]
        let cScale = mod[0..., 0..., 4]
        let cGate = mod[0..., 0..., 5]

        // 1. Self-attention
        let normX = (norm1(x) * (1 + scaleMsa) + shiftMsa).asType(wDtype)
        let attnOut = selfAttn(normX, ropeCosSin: ropeCosSin, originalContextLength: originalContextLength)
        x = (x.asType(.float32) + attnOut * gateMsa).asType(wDtype)
        // Intra-block eval bounds the Metal command-buffer at video seqLen — a
        // single block's whole lazy graph (90 MB fp32 modulation + self/cross-attn
        // + FFN) overruns the ~10s GPU watchdog at L≈736; per-BLOCK eval alone is
        // too coarse (per the trace bisection). The wan-core watchdog discipline,
        // one notch finer for Helios's larger per-block graph.
        eval(x)
        heliosTrace("  blk:selfattn", x)

        // 2. Cross-attention (history tokens skip it)
        if historyLen > 0 {
            let histX = x[0..., ..<historyLen]
            var currX = x[0..., historyLen...]
            let normCurr = norm3 != nil ? norm3!(currX) : currX
            let crossOut = crossAttn(normCurr, context: context, kvCache: crossKVCache)
            currX = (currX.asType(.float32) + crossOut).asType(wDtype)
            x = concatenated([histX, currX], axis: 1)
        } else {
            let normFull = norm3 != nil ? norm3!(x) : x
            let crossOut = crossAttn(normFull, context: context, kvCache: crossKVCache)
            x = (x.asType(.float32) + crossOut).asType(wDtype)
        }
        eval(x)
        heliosTrace("  blk:crossattn", x)

        // 3. Feed-forward
        let normF = (normFfn(x) * (1 + cScale) + cShift).asType(wDtype)
        let ffOut = ffn(normF)
        x = (x.asType(.float32) + ffOut.asType(.float32) * cGate).asType(wDtype)
        return x
    }
}
