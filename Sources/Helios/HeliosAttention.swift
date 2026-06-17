// Helios attention — 1:1 with PR#21 `attention.py`. The AR superset (restricted
// self-attn: history attends to history only; current attends to history+current)
// is why this is NOT a wan-core reuse. RoPE is applied inline (interleaved pairs,
// fp32) from a precomputed (cos,sin) — for the current chunk that (cos,sin) is
// wan-core's public `ropePrecomputeCosSin`; with history it's the concatenated
// [long,mid,short,current] tensor built in HeliosModel.

import Foundation
import MLX
import MLXFast
import MLXNN
import WanCore

/// Apply interleaved-pair RoPE to [B,L,N,D] given (cos,sin) each [L,1,D/2]. fp32.
private func applyRope(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
    let (b, s, n, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let xf = x.asType(.float32).reshaped(b, s, n, d / 2, 2)
    let xr = xf[.ellipsis, 0]
    let xi = xf[.ellipsis, 1]
    let outR = xr * cos - xi * sin
    let outI = xr * sin + xi * cos
    return stacked([outR, outI], axis: -1).reshaped(b, s, n, d)
}

final class HeliosSelfAttention: Module, @unchecked Sendable {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let restrictSelfAttn: Bool

    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "norm_q") var normQ: HeliosRMSNorm?
    @ModuleInfo(key: "norm_k") var normK: HeliosRMSNorm?

    init(_ dim: Int, _ numHeads: Int, qkNorm: Bool = true, eps: Float = 1e-6,
         restrictSelfAttn: Bool = false) {
        precondition(dim % numHeads == 0)
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self.restrictSelfAttn = restrictSelfAttn
        self._q.wrappedValue = Linear(dim, dim)
        self._k.wrappedValue = Linear(dim, dim)
        self._v.wrappedValue = Linear(dim, dim)
        self._o.wrappedValue = Linear(dim, dim)
        self._normQ.wrappedValue = qkNorm ? HeliosRMSNorm(dim, eps: eps) : nil
        self._normK.wrappedValue = qkNorm ? HeliosRMSNorm(dim, eps: eps) : nil
    }

    /// `ropeCosSin` is applied to the full (history+current) sequence; for S1
    /// (no history) it's the current-chunk cos/sin. `originalContextLength` =
    /// current-token count (history = s - originalContextLength).
    func callAsFunction(
        _ x: MLXArray, ropeCosSin: (MLXArray, MLXArray)?, originalContextLength: Int
    ) -> MLXArray {
        let (b, s) = (x.dim(0), x.dim(1))
        let (n, d) = (numHeads, headDim)
        let historyLen = s - originalContextLength

        let wDtype = linearDtype(q)
        let xW = x.asType(wDtype)
        var qP = q(xW)
        var kP = k(xW)
        if let normQ { qP = normQ(qP) }
        if let normK { kP = normK(kP) }

        var qh = qP.reshaped(b, s, n, d)
        var kh = kP.reshaped(b, s, n, d)
        let vh = v(xW).reshaped(b, s, n, d)
        if let (cos, sin) = ropeCosSin {
            qh = applyRope(qh, cos, sin)
            kh = applyRope(kh, cos, sin)
        }

        if restrictSelfAttn && historyLen > 0 {
            // history attends to history only; current attends to history+current
            let qHist = qh[0..., ..<historyLen].asType(wDtype)
            let qCurr = qh[0..., historyLen...].asType(wDtype)
            let kHist = kh[0..., ..<historyLen].asType(wDtype)
            let kCurr = kh[0..., historyLen...].asType(wDtype)
            let vHist = vh[0..., ..<historyLen]
            let vCurr = vh[0..., historyLen...]

            let histOut = MLXFast.scaledDotProductAttention(
                queries: qHist.transposed(0, 2, 1, 3), keys: kHist.transposed(0, 2, 1, 3),
                values: vHist.transposed(0, 2, 1, 3), scale: scale, mask: nil
            ).transposed(0, 2, 1, 3).reshaped(b, historyLen, -1)

            let kAll = concatenated([kHist, kCurr], axis: 1).transposed(0, 2, 1, 3)
            let vAll = concatenated([vHist, vCurr], axis: 1).transposed(0, 2, 1, 3)
            let currOut = MLXFast.scaledDotProductAttention(
                queries: qCurr.transposed(0, 2, 1, 3), keys: kAll, values: vAll,
                scale: scale, mask: nil
            ).transposed(0, 2, 1, 3).reshaped(b, originalContextLength, -1)

            return o(concatenated([histOut, currOut], axis: 1))
        } else {
            let qT = qh.asType(wDtype).transposed(0, 2, 1, 3)
            let kT = kh.asType(wDtype).transposed(0, 2, 1, 3)
            let vT = vh.transposed(0, 2, 1, 3)
            let out = MLXFast.scaledDotProductAttention(
                queries: qT, keys: kT, values: vT, scale: scale, mask: nil
            ).transposed(0, 2, 1, 3).reshaped(b, s, -1)
            return o(out)
        }
    }
}

final class HeliosCrossAttention: Module, @unchecked Sendable {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "norm_q") var normQ: HeliosRMSNorm?
    @ModuleInfo(key: "norm_k") var normK: HeliosRMSNorm?

    init(_ dim: Int, _ numHeads: Int, qkNorm: Bool = true, eps: Float = 1e-6) {
        precondition(dim % numHeads == 0)
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self._q.wrappedValue = Linear(dim, dim)
        self._k.wrappedValue = Linear(dim, dim)
        self._v.wrappedValue = Linear(dim, dim)
        self._o.wrappedValue = Linear(dim, dim)
        self._normQ.wrappedValue = qkNorm ? HeliosRMSNorm(dim, eps: eps) : nil
        self._normK.wrappedValue = qkNorm ? HeliosRMSNorm(dim, eps: eps) : nil
    }

    func prepareKV(_ context: MLXArray) -> (MLXArray, MLXArray) {
        let b = context.dim(0)
        let (n, d) = (numHeads, headDim)
        let wDtype = linearDtype(k)
        let ctx = context.asType(wDtype)
        var kP = k(ctx)
        if let normK { kP = normK(kP) }
        let kOut = kP.reshaped(b, -1, n, d).transposed(0, 2, 1, 3)
        let vOut = v(ctx).reshaped(b, -1, n, d).transposed(0, 2, 1, 3)
        return (kOut, vOut)
    }

    func callAsFunction(
        _ x: MLXArray, context: MLXArray, kvCache: (MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let b = x.dim(0)
        let (n, d) = (numHeads, headDim)
        let wDtype = linearDtype(q)
        var qP = q(x.asType(wDtype))
        if let normQ { qP = normQ(qP) }
        let qT = qP.reshaped(b, -1, n, d).transposed(0, 2, 1, 3)

        let kT: MLXArray
        let vT: MLXArray
        if let (kc, vc) = kvCache {
            (kT, vT) = (kc, vc)
        } else {
            (kT, vT) = prepareKV(context)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: qT, keys: kT, values: vT, scale: scale, mask: nil
        ).transposed(0, 2, 1, 3).reshaped(b, -1, n * d)
        return o(out)
    }
}
