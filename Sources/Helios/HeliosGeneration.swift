// HeliosGeneration — the autoregressive chunk loop. 1:1 with PR#21
// `generate_helios.py` (distilled default: no CFG / no anti-drift / no boundary
// blending). 33-frame chunks; a rolling 19-latent-frame multi-scale history
// buffer; a 3-stage coarse→fine spatial pyramid where each stage denoises the
// current chunk via `HeliosScheduler.stepDmd`, then nearest-upsamples and mixes
// in correlated block noise before the next stage.
//
// Noise is supplied through `HeliosNoiseSource` so the parity gate can INJECT the
// oracle's exact realizations: the per-chunk initial noise is `mx.random` (bit-
// identical Py↔Swift) but `sampleBlockNoise` draws from numpy (Cholesky-correlated),
// which is NOT reproducible across runtimes — hence injection for bit-parity, and a
// native `sampleBlockNoise` (correct distribution, different realization) for real runs.

import Foundation
import MLX
import MLXRandom
import WanCore

// MARK: - Spatial helpers (operate on [F, C, H, W]; latents live as [C, F, H, W])

/// [C, F, H, W] → [F, C, H, W] for spatial ops.
public func spatialReshape(_ x: MLXArray) -> MLXArray { x.transposed(1, 0, 2, 3) }
/// [F, C, H, W] → [C, F, H, W].
public func spatialUnreshape(_ x: MLXArray) -> MLXArray { x.transposed(1, 0, 2, 3) }

/// Area-mean 2× downsample. For an integer scale this equals F.interpolate(
/// mode="bilinear", align_corners=False) — a uniform tent filter over the 2×2
/// neighborhood reduces to a plain average. Input [F, C, H, W].
public func bilinearDownsample2d(_ x: MLXArray, _ targetH: Int, _ targetW: Int) -> MLXArray {
    let (f, c, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let (sh, sw) = (h / targetH, w / targetW)
    return x.reshaped(f, c, targetH, sh, targetW, sw).mean(axes: [3, 5])
}

/// Nearest-neighbor upsample by integer factor (repeat). Input [F, C, H, W].
public func nearestUpsample2d(_ x: MLXArray, _ targetH: Int, _ targetW: Int) -> MLXArray {
    let (sh, sw) = (targetH / x.dim(2), targetW / x.dim(3))
    return repeated(repeated(x, count: sh, axis: 2), count: sw, axis: 3)
}

// MARK: - Block noise

/// Structured per-patch noise via a correlated multivariate normal — reduces
/// block artifacts so spatially adjacent latents within a patch are correlated.
/// Returns [C, F, H, W]. Distribution matches the oracle; the realization differs
/// (MLX RNG ≠ numpy), so this is for real generation, not the bit-parity gate.
func sampleBlockNoise(
    channels: Int, numFrames: Int, height: Int, width: Int,
    patchSize: [Int], gamma: Double, key: MLXArray? = nil
) -> MLXArray {
    let (ph, pw) = (patchSize[1], patchSize[2])
    let blockSize = ph * pw

    // cov = I*(1+gamma) - ones*gamma + eps*I  →  lower-triangular Cholesky L.
    var cov = [[Double]](repeating: [Double](repeating: 0, count: blockSize), count: blockSize)
    for i in 0..<blockSize {
        for j in 0..<blockSize {
            cov[i][j] = (i == j ? (1 + gamma) + 1e-6 : 0) - gamma
        }
    }
    let L = cholesky(cov)
    let lT = MLXArray((0..<blockSize).flatMap { i in (0..<blockSize).map { j in Float(L[j][i]) } },
                      [blockSize, blockSize])  // L^T, row-major

    let blockCount = channels * numFrames * (height / ph) * (width / pw)
    let z = key.map { MLXRandom.normal([blockCount, blockSize], key: $0) }
        ?? MLXRandom.normal([blockCount, blockSize])
    var s = matmul(z, lT)  // [blockCount, blockSize]
    // reshape (C, F, H/ph, W/pw, ph, pw) → interleave → (C, F, H, W)
    s = s.reshaped(channels, numFrames, height / ph, width / pw, ph, pw)
        .transposed(0, 1, 2, 4, 3, 5)
        .reshaped(channels, numFrames, height, width)
    return s
}

/// Cholesky of a small SPD matrix (lower-triangular), plain Double — the cov here
/// is tiny (blockSize = ph*pw = 4), so a hand rolled factorization avoids a linalg dep.
private func cholesky(_ a: [[Double]]) -> [[Double]] {
    let n = a.count
    var l = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
    for i in 0..<n {
        for j in 0...i {
            var sum = a[i][j]
            for k in 0..<j { sum -= l[i][k] * l[j][k] }
            l[i][j] = i == j ? sum.squareRoot() : sum / l[j][j]
        }
    }
    return l
}

// MARK: - Noise source

/// Supplies the noise the AR loop consumes. `injected(...)` feeds oracle fixtures
/// (bit-parity gate); `live(...)` draws fresh noise for real generation.
public struct HeliosNoiseSource {
    /// (chunk, shape [C,F,H,W]) → full-res initial latent noise.
    public var initial: (Int, [Int]) -> MLXArray
    /// (chunk, stage, shape [C,F,H,W]) → correlated block noise.
    public var block: (Int, Int, [Int]) -> MLXArray

    public init(initial: @escaping (Int, [Int]) -> MLXArray,
                block: @escaping (Int, Int, [Int]) -> MLXArray) {
        self.initial = initial
        self.block = block
    }

    /// Real generation: mx.random initial noise + native correlated block noise.
    public static func live(patchSize: [Int], gamma: Double) -> HeliosNoiseSource {
        HeliosNoiseSource(
            initial: { _, shape in MLXRandom.normal(shape) },
            block: { _, _, shape in
                sampleBlockNoise(channels: shape[0], numFrames: shape[1],
                                 height: shape[2], width: shape[3],
                                 patchSize: patchSize, gamma: gamma)
            })
    }

    /// Parity gate: inject the oracle's captured realizations.
    public static func injected(
        initial: @escaping (Int) -> MLXArray, block: @escaping (Int, Int) -> MLXArray
    ) -> HeliosNoiseSource {
        HeliosNoiseSource(initial: { c, _ in initial(c) }, block: { c, s, _ in block(c, s) })
    }
}

// MARK: - Generation

public final class HeliosGeneration {
    let model: HeliosModel
    let scheduler: HeliosScheduler
    let config: HeliosConfig

    public init(model: HeliosModel, config: HeliosConfig) {
        self.model = model
        self.config = config
        self.scheduler = HeliosScheduler(
            numTrainTimesteps: config.numTrainTimesteps, shift: Double(config.shift),
            stages: config.stages, stageRange: config.stageRange.map { Double($0) },
            gamma: Double(config.gamma), useDynamicShifting: true)
    }

    /// Pixel→latent alignment + frame rounding (mirrors generate_video).
    public func align(width: Int, height: Int, numFrames: Int, pyramidStages: Int)
        -> (width: Int, height: Int, numChunks: Int) {
        let framesPerChunk = (config.numLatentFramesPerChunk - 1) * config.vaeStride[0] + 1  // 33
        let numChunks = max(1, (numFrames + framesPerChunk - 1) / framesPerChunk)
        let pf = 1 << (pyramidStages - 1)
        let alignH = config.patchSize[1] * pf * config.vaeStride[1]
        let alignW = config.patchSize[2] * pf * config.vaeStride[2]
        let h = ((height + alignH - 1) / alignH) * alignH
        let w = ((width + alignW - 1) / alignW) * alignW
        return (w, h, numChunks)
    }

    /// Run the AR loop. Returns per-chunk clean latents [C, npc, hLat, wLat].
    /// `contextEmbedded` = model.embedText([...]); `crossKV` = model.crossKVCaches(contextEmbedded).
    public func generate(
        contextEmbedded: MLXArray, crossKV: [(MLXArray, MLXArray)],
        hLatent: Int, wLatent: Int, numChunks: Int,
        pyramidSteps: [Int], amplifyFirstChunk: Bool, noise: HeliosNoiseSource,
        computeDType: DType = .bfloat16,
        teacher: ((Int) -> MLXArray)? = nil,
        onStage: ((Int, Int, MLXArray) -> Void)? = nil
    ) -> [MLXArray] {
        let c = config.inDim
        let npc = config.numLatentFramesPerChunk
        let hs = config.historySizes
        let nHist = hs.reduce(0, +)  // 19
        let stages = scheduler.stages

        var historyLatents = MLXArray.zeros([c, nHist, hLatent, wLatent])

        // Constant frame indices: prefix 0 | long 1..16 | mid 17..18 | 1x 19 | current 20..28.
        let total = 1 + nHist + npc
        let idx = MLXArray((0..<total).map { Int32($0) })
        let idxLong = idx[1 ..< (1 + hs[0])]
        let idxMid = idx[(1 + hs[0]) ..< (1 + hs[0] + hs[1])]
        let idx1x = idx[(1 + hs[0] + hs[1]) ..< (1 + nHist)]
        let idxShort = concatenated([idx[0 ..< 1], idx1x])
        let idxCurrent = idx[(1 + nHist)...]

        var chunks: [MLXArray] = []
        var imagePrefix: MLXArray? = nil

        for chunkIdx in 0..<numChunks {
            let isFirst = chunkIdx == 0

            let recent = historyLatents[0..., (historyLatents.dim(1) - nHist)...]
            let parts = split(recent, indices: [hs[0], hs[0] + hs[1]], axis: 1)
            let (histLong, histMid, hist1x) = (parts[0], parts[1], parts[2])

            let prefix = isFirst ? MLXArray.zeros([c, 1, hLatent, wLatent]) : imagePrefix!
            let histShort = concatenated([prefix, hist1x], axis: 1)

            // Initial noise → downsample to 1/4 resolution.
            let initNoise = noise.initial(chunkIdx, [c, npc, hLatent, wLatent])
            var curH = hLatent, curW = wLatent
            var latents = spatialReshape(initNoise)
            for _ in 0 ..< (stages - 1) {
                curH /= 2; curW /= 2
                latents = bilinearDownsample2d(latents, curH, curW) * 2
            }
            latents = spatialUnreshape(latents)
            eval(latents)

            var startPoints = [latents]

            for iS in 0..<stages {
                let imageSeqLen = npc * curH * curW / config.patchSize.reduce(1, *)
                scheduler.setTimesteps(
                    numInferenceSteps: pyramidSteps[iS], stageIndex: iS,
                    imageSeqLen: imageSeqLen,
                    isAmplifyFirstChunk: amplifyFirstChunk && isFirst)

                if iS > 0 {
                    curH *= 2; curW *= 2
                    latents = spatialReshape(latents)
                    latents = nearestUpsample2d(latents, curH, curW)
                    latents = spatialUnreshape(latents)

                    let oriSigma = 1 - scheduler.oriStartSigmas[iS]
                    let g = scheduler.gamma
                    let alpha = 1 / ((1 + 1 / g).squareRoot() * (1 - oriSigma) + oriSigma)
                    let beta = alpha * (1 - oriSigma) / g.squareRoot()
                    let bn = noise.block(chunkIdx, iS, [c, npc, curH, curW])
                    latents = Float(alpha) * latents + Float(beta) * bn
                    startPoints.append(latents)
                }

                let hShort = histShort.asType(computeDType)
                let hMid = histMid.asType(computeDType)
                let hLong = histLong.asType(computeDType)
                let ts = scheduler.timesteps
                let sig = scheduler.sigmas

                for i in 0..<ts.count {
                    let timestep = MLXArray(Int32(Int(ts[i])))
                    let noisePred = model(
                        latents.asType(computeDType), timestep: timestep,
                        encoderHiddenStates: contextEmbedded, frameIndices: idxCurrent,
                        historyShort: hShort, historyMid: hMid, historyLong: hLong,
                        historyShortIndices: idxShort, historyMidIndices: idxMid,
                        historyLongIndices: idxLong, crossKVCaches: crossKV)
                    let sigmaNext: Double? = i < ts.count - 1 ? sig[i + 1] : nil
                    latents = scheduler.stepDmd(
                        modelOutput: noisePred, sample: latents, curStep: i,
                        noisyStart: startPoints[iS], sigmaT: sig[i], sigmaNext: sigmaNext)
                    eval(latents)
                }
                onStage?(chunkIdx, iS, latents)
            }

            eval(latents)
            chunks.append(latents)
            // Parity harness: teacher-force history from the oracle's clean chunk so
            // each chunk is a per-chunk forward+loop parity test, decoupled from the
            // (sensitivity-amplified) drift of our own prior-chunk rounding. nil →
            // free-running autoregression (production).
            let carried = teacher?(chunkIdx) ?? latents
            historyLatents = concatenated([historyLatents, carried], axis: 1)
            if isFirst { imagePrefix = carried[0..., 0 ..< 1] }
        }
        return chunks
    }

    /// Decode per-chunk latents → video [1, 3, Ttot, H, W] in [-1, 1], reusing
    /// wan-core's 16-ch `WanVAE` (`decodeStreaming` for flat-memory chunked decode).
    /// Mirrors generate_helios: decode each chunk independently (avoids cross-chunk
    /// causal-conv artifacts), trim the VAE warmup frames, then drop the first pixel
    /// frame (the overlap/conditioning duplicate of the prior chunk's last frame).
    public func decode(_ chunks: [MLXArray], vae: WanVAE) -> MLXArray {
        let strideT = config.vaeStride[0]
        let valid = (config.numLatentFramesPerChunk - 1) * strideT + 1  // 33
        var videos: [MLXArray] = []
        for chunk in chunks {
            var v = decodeStreaming(vae: vae, chunk.expandedDimensions(axis: 0))  // [1,3,T,H,W]
            let t = v.dim(2)
            if t > valid { v = v[0..., 0..., (t - valid)...] }
            v = v[0..., 0..., 1...]  // drop overlap frame (33 → 32)
            eval(v)
            videos.append(v)
        }
        return concatenated(videos, axis: 2)
    }
}
