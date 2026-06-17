// HeliosModel — the Helios-Distilled DiT. 1:1 with PR#21 `transformer.py`.
// Backbone is Wan2.2-A14B-shaped; RoPE tables REUSE wan-core's public `ropeParams`
// (Helios per-axis freqs == wan-core's, verified). Self-attn is FULL over
// [history+current] (Helios-Distilled sets restrict_self_attn=False); cross-attn
// is current-only (guidance_cross_attn); history tokens use the t=0 modulation.

import Foundation
import MLX
import MLXNN
import WanCore

/// Non-Module buffers (per-axis RoPE freq tables + sinusoidal inv_freq). Kept OUT
/// of the Module so reflection doesn't register them as bogus parameters.
final class HeliosBuffers: @unchecked Sendable {
    let ft: MLXArray
    let fh: MLXArray
    let fw: MLXArray
    let invFreq: MLXArray

    init(_ config: HeliosConfig) {
        (ft, fh, fw) = HeliosRoPE.freqs(config)
        let half = config.freqDim / 2
        invFreq = MLXArray((0..<half).map { Float(pow(10000.0, -Double($0) / Double(half))) })
    }
}

/// Output head: parameterless norm + AdaLN table (`head.modulation`) + projection (`head.head`).
final class HeliosHead: Module, @unchecked Sendable {
    let outputNorm: HeliosLayerNorm
    @ModuleInfo(key: "head") var head: Linear
    @ParameterInfo(key: "modulation") var modulation: MLXArray  // [1, 2, dim]

    init(dim: Int, outDim: Int, patchSize: [Int], eps: Float) {
        self.outputNorm = HeliosLayerNorm(dim, eps)
        self._head.wrappedValue = Linear(dim, patchSize.reduce(1, *) * outDim)
        self._modulation.wrappedValue = MLXRandom.normal([1, 2, dim]) * pow(Float(dim), -0.5)
    }

    func callAsFunction(_ x: MLXArray, _ temb: MLXArray) -> MLXArray {
        let wDtype = linearDtype(head)
        let tembExp = broadcast(temb[0..., .newAxis, 0...], to: [x.dim(0), x.dim(1), x.dim(2)])
        let modOut = (modulation.expandedDimensions(axis: 0) + tembExp.expandedDimensions(axis: 2)).asType(wDtype)
        let out = (outputNorm(x) * (1 + modOut[0..., 0..., 1]) + modOut[0..., 0..., 0]).asType(wDtype)
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
    private var t0Cache: MLXArray?

    public init(_ config: HeliosConfig) {
        self.config = config
        self.dim = config.dim
        self.patchSize = config.patchSize
        let dim = config.dim
        self._patchEmbedding.wrappedValue = Linear(config.inDim * config.patchSize.reduce(1, *), dim)
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

    private func arange(_ n: Int) -> MLXArray { MLXArray((0..<n).map { Int32($0) }) }

    // Env-gated bisection trace (HELIOS_TRACE=1) → stderr (unbuffered, survives a crash).
    private static let trace = ProcessInfo.processInfo.environment["HELIOS_TRACE"] != nil
    private func dbg(_ s: String, _ a: MLXArray? = nil) {
        guard Self.trace else { return }
        if let a { eval(a) }
        FileHandle.standardError.write(Data("[trace] \(s)\(a.map { " \($0.shape)" } ?? "")\n".utf8))
    }

    /// Patchify current latents [C,F,H,W] → ([1, L, dim], grid (f',h',w')).
    func patchify(_ x: MLXArray) -> (MLXArray, (Int, Int, Int)) {
        let c = x.dim(0)
        let (pt, ph, pw) = (patchSize[0], patchSize[1], patchSize[2])
        let f = (x.dim(1) / pt) * pt, h = (x.dim(2) / ph) * ph, w = (x.dim(3) / pw) * pw
        let (fo, ho, wo) = (f / pt, h / ph, w / pw)
        let xc = x[0..., ..<f, ..<h, ..<w]
            .reshaped(c, fo, pt, ho, ph, wo, pw)
            .transposed(1, 3, 5, 0, 2, 4, 6)
            .reshaped(fo * ho * wo, -1)
        return (patchEmbedding(xc).asType(linearDtype(patchEmbedding)).expandedDimensions(axis: 0),
                (fo, ho, wo))
    }

    /// Multi-scale history patchify: edge-pad to kernel, reshape/transpose/flatten, project.
    func patchifyHistory(_ x: MLXArray, _ kernel: (Int, Int, Int), _ proj: Linear) -> MLXArray {
        var x = x
        let c = x.dim(0)
        let (kt, kh, kw) = kernel
        let (f0, h0, w0) = (x.dim(1), x.dim(2), x.dim(3))
        let (pt, ph, pw) = ((kt - f0 % kt) % kt, (kh - h0 % kh) % kh, (kw - w0 % kw) % kw)
        if pt > 0 || ph > 0 || pw > 0 {
            x = padded(x, widths: [.init((0, 0)), .init((0, pt)), .init((0, ph)), .init((0, pw))], mode: .edge)
        }
        let (f, h, w) = (x.dim(1), x.dim(2), x.dim(3))
        let (fo, ho, wo) = (f / kt, h / kh, w / kw)
        let xc = x.reshaped(c, fo, kt, ho, kh, wo, kw)
            .transposed(1, 3, 5, 0, 2, 4, 6)
            .reshaped(fo * ho * wo, -1)
        return proj(xc).asType(linearDtype(proj)).expandedDimensions(axis: 0)
    }

    func unpatchify(_ x: MLXArray, _ grid: (Int, Int, Int)) -> MLXArray {
        let cOut = config.outDim
        let (pt, ph, pw) = (patchSize[0], patchSize[1], patchSize[2])
        let (f, h, w) = grid
        return x[0, ..<(f * h * w)]
            .reshaped(f, h, w, pt, ph, pw, cOut)
            .transposed(6, 0, 3, 1, 4, 2, 5)
            .reshaped(cOut, f * pt, h * ph, w * pw)
    }

    public func embedText(_ context: [MLXArray]) -> MLXArray {
        let modelDtype = linearDtype(patchEmbedding)
        let padded = context.map { ctx -> MLXArray in
            let padLen = config.textLen - ctx.dim(0)
            guard padLen > 0 else { return ctx }
            return concatenated([ctx, MLXArray.zeros([padLen, ctx.dim(1)], dtype: ctx.dtype)], axis: 0)
        }
        return textEmbedding1(geluApproximate(textEmbedding0(stacked(padded)))).asType(modelDtype)
    }

    func crossKVCaches(_ context: MLXArray) -> [(MLXArray, MLXArray)] {
        blocks.map { $0.crossAttn.prepareKV(context) }
    }

    /// Cached t=0 time-projection [1,6,dim] for history tokens (computed after load).
    private func t0Projection() -> MLXArray {
        if let c = t0Cache { return c }
        var e = MLXArray([Float(0)]) * buffers.invFreq
        e = concatenated([cos(e), sin(e)], axis: -1).expandedDimensions(axis: 0)
        let temb = timeEmbedding1(silu(timeEmbedding0(e)))
        let tp = timeProjection(silu(temb)).reshaped(1, 6, -1)
        eval(tp)
        t0Cache = tp
        return tp
    }

    /// Forward. With history args nil → S1 no-history path; with history →
    /// multi-scale prepend + concatenated RoPE + zero-history t0 modulation.
    public func callAsFunction(
        _ latents: MLXArray, timestep: MLXArray, encoderHiddenStates: MLXArray,
        frameIndices: MLXArray? = nil,
        historyShort: MLXArray? = nil, historyMid: MLXArray? = nil, historyLong: MLXArray? = nil,
        historyShortIndices: MLXArray? = nil, historyMidIndices: MLXArray? = nil,
        historyLongIndices: MLXArray? = nil,
        crossKVCaches: [(MLXArray, MLXArray)]? = nil
    ) -> MLXArray {
        var (hidden, grid) = patchify(latents)
        let (gf, gh, gw) = grid
        let L = hidden.dim(1)
        let dtype = hidden.dtype
        let (ft, fh, fw) = (buffers.ft, buffers.fh, buffers.fw)

        dbg("patchify", hidden)
        let fi = frameIndices ?? arange(gf)
        var (rCos, rSin) = HeliosRoPE.precompute(fi, gh, gw, ft, fh, fw, downsample: nil, dtype: dtype)
        dbg("current rope", rCos)

        var historyLen = 0
        if let hShort = historyShort, let hMid = historyMid, let hLong = historyLong {
            let histS = patchifyHistory(hShort, (1, 2, 2), patchShort)
            let histM = patchifyHistory(hMid, (2, 4, 4), patchMid)
            let histL = patchifyHistory(hLong, (4, 8, 8), patchLong)
            dbg("histS", histS); dbg("histM", histM); dbg("histL", histL)
            // history RoPE is computed at the SHORT output spatial resolution
            let (hsH, hsW) = (hShort.dim(2) / 2, hShort.dim(3) / 2)
            let (cs, ss) = HeliosRoPE.precompute(historyShortIndices!, hsH, hsW, ft, fh, fw, downsample: nil, dtype: dtype)
            let (cm, sm) = HeliosRoPE.precompute(historyMidIndices!, hsH, hsW, ft, fh, fw, downsample: (2, 2, 2), dtype: dtype)
            let (cl, sl) = HeliosRoPE.precompute(historyLongIndices!, hsH, hsW, ft, fh, fw, downsample: (4, 4, 4), dtype: dtype)
            dbg("history rope", cl)
            // prepend history (long, mid, short) before the current chunk — hidden
            // and RoPE in the SAME order. rCos/rSin still hold the current values.
            hidden = concatenated([histL, histM, histS, hidden], axis: 1)
            rCos = concatenated([cl, cm, cs, rCos], axis: 0)
            rSin = concatenated([sl, sm, ss, rSin], axis: 0)
            historyLen = histL.dim(1) + histM.dim(1) + histS.dim(1)
            dbg("prepended hidden", hidden)
        }

        // Time embedding + 6-vec modulation → [1, total, 6, dim].
        var tEmb = timestep.asType(.float32) * buffers.invFreq
        tEmb = concatenated([cos(tEmb), sin(tEmb)], axis: -1)
        if tEmb.ndim == 1 { tEmb = tEmb.expandedDimensions(axis: 0) }
        let temb = timeEmbedding1(silu(timeEmbedding0(tEmb)))
        let tproj = timeProjection(silu(temb)).reshaped(1, 6, -1)
        var tprojExp = broadcast(tproj[0..., 0..., .newAxis, 0...], to: [1, 6, L, dim])
        if historyLen > 0 {
            let t0Exp = broadcast(t0Projection()[0..., 0..., .newAxis, 0...], to: [1, 6, historyLen, dim])
            tprojExp = concatenated([t0Exp, tprojExp], axis: 2)
        }
        tprojExp = tprojExp.transposed(0, 2, 1, 3)  // [1, total, 6, dim]
        dbg("tproj", tprojExp)

        for (i, block) in blocks.enumerated() {
            hidden = block(
                hidden, context: encoderHiddenStates, tProj: tprojExp,
                ropeCosSin: (rCos, rSin), originalContextLength: L,
                crossKVCache: crossKVCaches?[i])
            // Per-block eval bounds the lazy graph / Metal command-buffer size —
            // the wan-core WanModel watchdog discipline (a 40-block graph built
            // whole overruns the ~10s GPU command-buffer ceiling at video seqLen).
            eval(hidden)
            if i == 0 || i == 39 { dbg("block \(i)", hidden) }
        }

        return unpatchify(head(hidden[0..., (hidden.dim(1) - L)...], temb), grid)
    }
}
