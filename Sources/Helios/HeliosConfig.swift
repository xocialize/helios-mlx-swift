// HeliosConfig — resolved Helios-Distilled architecture config.
//
// Values verified against the checkpoint's transformer/config.json (2026-06-14).
// The transformer block is Wan2.2-A14B-shaped (dim 5120, 40×40, head_dim 128,
// rope (44,42,42)); the Helios-only fields drive the autoregressive history delta.
// 1:1 with the oracle `mlx_video/models/helios/config.py` HeliosModelConfig.

import Foundation

public struct HeliosConfig: Sendable, Codable {
    // — Transformer (identical to Wan2.2-A14B) —
    public var dim: Int = 5120
    public var ffnDim: Int = 13824
    public var numHeads: Int = 40
    public var numLayers: Int = 40
    public var patchSize: [Int] = [1, 2, 2]
    public var inDim: Int = 16
    public var outDim: Int = 16
    public var textDim: Int = 4096
    public var freqDim: Int = 256
    public var textLen: Int = 512
    public var eps: Float = 1e-6
    public var qkNorm: Bool = true
    public var crossAttnNorm: Bool = true

    // — RoPE (per-axis; bit-identical reuse of wan-core RoPE) —
    public var ropeDim: [Int] = [44, 42, 42]
    public var ropeTheta: Float = 10000.0

    // — Helios-only: multi-scale autoregressive history memory —
    public var historySizes: [Int] = [16, 2, 1]
    public var numLatentFramesPerChunk: Int = 9
    public var hasMultiTermMemoryPatch: Bool = true
    public var historyScaleMode: String = "per_head"
    public var guidanceCrossAttn: Bool = true
    public var zeroHistoryTimestep: Bool = true
    public var isAmplifyHistory: Bool = false

    // — VAE / T5 (reused from wan-core, listed for completeness) —
    public var vaeStride: [Int] = [4, 8, 8]
    public var vaeZDim: Int = 16

    // — DMD distilled scheduler —
    public var numTrainTimesteps: Int = 1000
    public var shift: Float = 1.0
    public var stages: Int = 3
    public var stageRange: [Float] = [0, 1.0 / 3.0, 2.0 / 3.0, 1]
    public var gamma: Float = 1.0 / 3.0

    public var headDim: Int { dim / numHeads }

    public init() {}

    /// Helios-Distilled: x0-prediction, no CFG, DMD scheduler, 2–3 steps/stage.
    public static func heliosDistilled() -> HeliosConfig {
        var c = HeliosConfig()
        c.shift = 1.0
        return c
    }
}
