// S6 gate: int4 DiT quality vs full precision. Loads the int4 transformer (block
// Linears quantized, the oracle's _quantize_predicate scope) and runs the SAME
// history forward as S2, then cosine-compares to the S2 `out` fixture (the bf16
// full-precision reference for identical inputs). int4 is not bit-exact; the gate
// is the oracle's own quantize bar (cosine ≥ 0.99). Reuses Tests/.../Fixtures/s2.
//
// STREAM DISCIPLINE (mlx-swift-integration: swift-port-parity §2): the 8.4 GB int4
// WEIGHT LOAD rides the CPU stream (else a multi-GB read holds one Metal buffer past
// the watchdog), but the FORWARD runs on the GPU stream — quantized matmuls route to
// Metal regardless, so a CPU-pinned quantized graph becomes ONE Metal buffer fenced on
// CPU ops at every block (observed: 100+ min CPU, no progress). GPU float noise ~1e-3
// is negligible against int4 error at a 0.99 gate.

import Foundation
import MLX
import MLXNN
import WanCore
import Helios

private func cosineSim(_ a: MLXArray, _ b: MLXArray) -> Float {
    let af = a.asType(.float32).flattened(), bf = b.asType(.float32).flattened()
    let dot = sum(af * bf)
    let norm = sqrt(sum(af * af)) * sqrt(sum(bf * bf))
    return (dot / (norm + 1e-12)).item(Float.self)
}

func runS6Gate(int4Model: URL, fixtures: URL) -> Bool {
    func fx(_ name: String) throws -> MLXArray {
        try loadNumpy(url: fixtures.appending(path: "\(name).npy"))
    }
    do {
        let config = HeliosConfig.heliosDistilled()
        // Construct + quantize-slot + LOAD on the CPU stream (multi-GB read discipline).
        let model = try Device.withDefaultDevice(.cpu) { () -> HeliosModel in
            let m = HeliosModel(config)
            WeightLoader.applyQuantization(to: m, quantization: WanQuantization(groupSize: 64, bits: 4))
            let weights = try MLX.loadArrays(url: int4Model)
            try m.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
            eval(m)
            return m
        }
        // Forward on the GPU stream (quantized matmuls → Metal; intra-block eval bounds it).
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
        let ref = try fx("out")
        let cos = cosineSim(out, ref)
        let pass = out.shape == ref.shape && cos >= 0.99
        print("  [int4 vs bf16] shape \(out.shape) cosine=\(cos) \(pass ? "OK" : "FAIL")")
        print(pass ? "[s6-gate] PASS (int4 cosine ≥ 0.99)" : "[s6-gate] FAIL")
        return pass
    } catch {
        print("  [s6-gate] ERROR: \(error)")
        return false
    }
}
