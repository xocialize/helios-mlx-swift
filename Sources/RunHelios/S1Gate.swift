// S1 component parity gate (CLI; the metallib-in-xctest lesson). Real canonical
// weights, CPU stream, vs PR#21 oracle fixtures (tools/dump_helios_s1.py).
//   1. key contract: model param keys == converted safetensors headers (0/0).
//   2. RoPE table: wan-core ropeParams(44/42/42) == oracle helios_rope_params.
//   3. full no-history forward: HeliosModel(latents,t,ctx) == oracle fwd_out
//      (exercises patchify + time/text embed + 40 blocks + output head).

import Foundation
import MLX
import MLXNN
import WanCore
import Helios

private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

func runS1Gate(mlxModel: URL, fixtures: URL) -> Bool {
    func fx(_ name: String) throws -> MLXArray {
        try loadNumpy(url: fixtures.appending(path: "\(name).npy"))
    }
    var allPass = true
    do {
        try Device.withDefaultDevice(.cpu) {
            let cfg = HeliosConfig.heliosDistilled()
            let model = HeliosModel(cfg)

            // 1. key contract vs real converted headers
            let weights = try MLX.loadArrays(url: mlxModel)
            let modelKeys = Set(model.parameters().flattened().map { $0.0 })
            let weightKeys = Set(weights.keys)
            let missing = modelKeys.subtracting(weightKeys)
            let unused = weightKeys.subtracting(modelKeys)
            let keyOK = missing.isEmpty && unused.isEmpty
            if !keyOK {
                print("  [keys] MISSING \(missing.sorted().prefix(8)) | UNUSED \(unused.sorted().prefix(8))")
            }
            print("  [keys] model \(modelKeys.count) vs weights \(weightKeys.count): \(keyOK ? "OK" : "MISMATCH")")
            allPass = allPass && keyOK
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
            eval(model)

            // 2. RoPE table parity (wan-core public ropeParams)
            let mine = concatenated(
                [ropeParams(1024, cfg.ropeDim[0], theta: Double(cfg.ropeTheta)),
                 ropeParams(1024, cfg.ropeDim[1], theta: Double(cfg.ropeTheta)),
                 ropeParams(1024, cfg.ropeDim[2], theta: Double(cfg.ropeTheta))], axis: 1)
            let ref = try concatenated([fx("rope_freqs_t"), fx("rope_freqs_h"), fx("rope_freqs_w")], axis: 1)
            let dRope = maxAbs(mine, ref)
            let ropeOK = dRope <= 1e-5
            print("  [rope] wan-core ropeParams vs oracle: max_abs=\(dRope) \(ropeOK ? "OK" : "FAIL")")
            allPass = allPass && ropeOK

            // 3. full no-history forward
            let latents = try fx("fwd_latents")
            let timestep = try fx("fwd_timestep")
            let ctxRaw = try fx("fwd_ctx_raw")
            let expOut = try fx("fwd_out")
            let ctxEmb = model.embedText([ctxRaw])
            let out = model(latents, timestep: timestep, encoderHiddenStates: ctxEmb)
            eval(out)
            let dFwd = maxAbs(out, expOut)
            let fwdOK = out.shape == expOut.shape && dFwd <= 2e-2
            print("  [fwd] shape \(out.shape) (exp \(expOut.shape)) max_abs=\(dFwd) \(fwdOK ? "OK" : "FAIL")")
            allPass = allPass && fwdOK
        }
    } catch {
        print("  [s1-gate] ERROR: \(error)")
        return false
    }
    print(allPass ? "[s1-gate] PASS" : "[s1-gate] FAIL")
    return allPass
}
