// S2 gate: Helios model forward WITH history (the AR delta) vs PR#21 oracle.
// Exercises multi-scale history patchify, full self-attn over [history+current],
// current-only cross-attn, zero-history t0 modulation, frame-offset/downsampled
// history RoPE. Real canonical weights, CPU stream. Fixtures: dump_helios_s2.py.

import Foundation
import MLX
import MLXNN
import WanCore
import Helios

func runS2Gate(mlxModel: URL, fixtures: URL) -> Bool {
    func fx(_ name: String) throws -> MLXArray {
        try loadNumpy(url: fixtures.appending(path: "\(name).npy"))
    }
    func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
    do {
        return try Device.withDefaultDevice(.cpu) {
            let model = HeliosModel(HeliosConfig.heliosDistilled())
            let weights = try MLX.loadArrays(url: mlxModel)
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
            eval(model)

            let ctxEmb = model.embedText([try fx("ctx_raw")])
            let out = model(
                try fx("latents"), timestep: try fx("timestep"), encoderHiddenStates: ctxEmb,
                frameIndices: try fx("idx_current").asType(.int32),
                historyShort: try fx("hist_short"), historyMid: try fx("hist_mid"),
                historyLong: try fx("hist_long"),
                historyShortIndices: try fx("idx_short").asType(.int32),
                historyMidIndices: try fx("idx_mid").asType(.int32),
                historyLongIndices: try fx("idx_long").asType(.int32))
            eval(out)
            let exp = try fx("out")
            let d = maxAbs(out, exp)
            // bf16-accumulation tolerance: 40 blocks + history tokens drift a bit
            // more than S1's no-history 0.0195 (production fp32 DiT is far tighter).
            let pass = out.shape == exp.shape && d <= 3e-2
            print("  [fwd+history] shape \(out.shape) (exp \(exp.shape)) max_abs=\(d) \(pass ? "OK" : "FAIL")")
            print(pass ? "[s2-gate] PASS" : "[s2-gate] FAIL")
            return pass
        }
    } catch {
        print("  [s2-gate] ERROR: \(error)")
        return false
    }
}
