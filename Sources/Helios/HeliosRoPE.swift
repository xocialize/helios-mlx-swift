// HeliosRoPE — the AR-specific RoPE compute (1:1 with PR#21 `rope.py`). The
// per-axis freq TABLES are wan-core's public `ropeParams` (verified bit-identical
// to `helios_rope_params`); this adds the Helios-local pieces wan-core's
// `ropePrecomputeCosSin` can't do: explicit (non-contiguous) frame indices — the
// current chunk is offset past the history prefix — and the avg-pool spatial
// downsample for mid/long history scales.

import Foundation
import MLX
import WanCore

enum HeliosRoPE {
    /// Per-axis freq tables (t,h,w) — wan-core ropeParams reuse.
    static func freqs(_ config: HeliosConfig) -> (MLXArray, MLXArray, MLXArray) {
        let theta = Double(config.ropeTheta)
        return (ropeParams(1024, config.ropeDim[0], theta: theta),
                ropeParams(1024, config.ropeDim[1], theta: theta),
                ropeParams(1024, config.ropeDim[2], theta: theta))
    }

    /// [F,H,W,halfD,2] from explicit frame indices (gathered) + contiguous h/w.
    static func compute5d(
        _ frameIndices: MLXArray, _ h: Int, _ w: Int,
        _ ft: MLXArray, _ fh: MLXArray, _ fw: MLXArray, dtype: DType
    ) -> MLXArray {
        let (ftc, fhc, fwc) = (ft.asType(dtype), fh.asType(dtype), fw.asType(dtype))
        let f = frameIndices.dim(0)
        let (dt, dh, dw) = (ftc.dim(1), fhc.dim(1), fwc.dim(1))
        let ftB = broadcast(
            take(ftc, frameIndices, axis: 0).reshaped(f, 1, 1, dt, 2), to: [f, h, w, dt, 2])
        let fhB = broadcast(fhc[..<h].reshaped(1, h, 1, dh, 2), to: [f, h, w, dh, 2])
        let fwB = broadcast(fwc[..<w].reshaped(1, 1, w, dw, 2), to: [f, h, w, dw, 2])
        return concatenated([ftB, fhB, fwB], axis: 3)
    }

    /// Edge-pad to kernel-divisible, then avg-pool over (kt,kh,kw) blocks.
    static func padDownsample(_ rope5d: MLXArray, _ kt: Int, _ kh: Int, _ kw: Int) -> MLXArray {
        var r = rope5d
        let (f, h, w) = (r.dim(0), r.dim(1), r.dim(2))
        let (d, c) = (r.dim(3), r.dim(4))
        let (pt, ph, pw) = ((kt - f % kt) % kt, (kh - h % kh) % kh, (kw - w % kw) % kw)
        if pt > 0 || ph > 0 || pw > 0 {
            r = padded(
                r,
                widths: [.init((0, pt)), .init((0, ph)), .init((0, pw)), .init((0, 0)), .init((0, 0))],
                mode: .edge)
        }
        let (f2, h2, w2) = (r.dim(0), r.dim(1), r.dim(2))
        r = r.reshaped(f2 / kt, kt, h2 / kh, kh, w2 / kw, kw, d, c)
        return r.mean(axes: [1, 3, 5])
    }

    /// Flatten [F,H,W,halfD,2] → (cos,sin) each [F*H*W, 1, halfD].
    static func flatten(_ rope5d: MLXArray) -> (MLXArray, MLXArray) {
        let (f, h, w, d) = (rope5d.dim(0), rope5d.dim(1), rope5d.dim(2), rope5d.dim(3))
        let flat = rope5d.reshaped(f * h * w, 1, d, 2)
        return (flat[.ellipsis, 0], flat[.ellipsis, 1])
    }

    /// Current-chunk OR history cos/sin. `downsample` = nil (current/short) or
    /// (kt,kh,kw) (mid/long history, computed at short spatial res then pooled).
    static func precompute(
        _ frameIndices: MLXArray, _ h: Int, _ w: Int,
        _ ft: MLXArray, _ fh: MLXArray, _ fw: MLXArray,
        downsample: (Int, Int, Int)? = nil, dtype: DType
    ) -> (MLXArray, MLXArray) {
        var r = compute5d(frameIndices, h, w, ft, fh, fw, dtype: dtype)
        if let (kt, kh, kw) = downsample { r = padDownsample(r, kt, kh, kw) }
        return flatten(r)
    }
}
