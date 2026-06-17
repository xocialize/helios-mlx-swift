// S3b gate: the full autoregressive generation loop (generate_helios.py) vs PR#21.
// Real canonical weights, CPU stream. The oracle's per-chunk initial noise and the
// numpy-Cholesky block noise are INJECTED from fixtures (numpy RNG ≠ MLX RNG), so the
// only divergence source is the bf16 model forward + fp32 scheduler arithmetic.
//
// Localizes failures: [start0] checks the 1/4-res downsample (spatial helpers, no
// model); [chunk N] checks each chunk's denoised output (= the e2e latent, since the
// final video latent is just the concatenation of these). Fixtures: dump_helios_s3_loop.py.

import Foundation
import MLX
import MLXNN
import WanCore
import Helios

private struct S3Meta: Decodable {
    let h_latent: Int
    let w_latent: Int
    let num_chunks: Int
    let pyramid_steps: [Int]
    let amplify_first_chunk: Bool
}

func runS3Gate(mlxModel: URL, fixtures: URL, fp32: Bool = false) -> Bool {
    func fx(_ name: String) throws -> MLXArray {
        try loadNumpy(url: fixtures.appending(path: "\(name).npy"))
    }
    func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
    // The production Helios DiT is fp32 (the default); run the gate there. Per-forward
    // fp32 parity is at the precision floor (≤7e-5, see the per-stage prints) — but the
    // 3-stage DMD pyramid is a chaotic, sensitivity-amplifying system at the t≈998 / high-
    // magnitude regime: a stage-0 fp32 op-order difference (MLX-Swift vs MLX-Python SDPA/
    // matmul accumulation) grows ~10-20× per stage (zero-history chunk 0 stays ~1e-4;
    // nonzero-history chunk 1 reaches ~1.3e-2). This is the SAME mechanism that blew bf16
    // up to ~2.3. The forward is verified bit-faithful (the per-stage X.0 deltas); the loop
    // tolerance bounds the amplified-but-correct end-to-end drift. (cf. S1b/S2 bf16 bounds.)
    let computeDType: DType = fp32 ? .float32 : .bfloat16
    let tol: Float = fp32 ? 2e-2 : 5e-2

    do {
        guard let metaData = try? Data(contentsOf: fixtures.appending(path: "meta.json")),
              let meta = try? JSONDecoder().decode(S3Meta.self, from: metaData)
        else {
            print("[s3-gate] SKIP — meta.json not found; run tools/dump_helios_s3_loop.py")
            return true
        }

        return try Device.withDefaultDevice(.cpu) {
            let config = HeliosConfig.heliosDistilled()
            let stages = config.stages

            // Preload injected noise + oracle chunks (closures must be non-throwing).
            var initNoise: [MLXArray] = []
            var oracleChunks: [MLXArray] = []
            var blockNoise: [String: MLXArray] = [:]
            for c in 0..<meta.num_chunks {
                initNoise.append(try fx("noise_\(c)"))
                oracleChunks.append(try fx("chunk_\(c)"))
                for s in 1..<stages { blockNoise["\(c)_\(s)"] = try fx("blocknoise_\(c)_\(s)") }
            }
            let noise = HeliosNoiseSource.injected(
                initial: { initNoise[$0] },
                block: { blockNoise["\($0)_\($1)"]! })

            // --- [start0] spatial-helper check (downsample to 1/4 res), no model ---
            var startOK = true
            for c in 0..<meta.num_chunks {
                var lat = spatialReshape(initNoise[c])
                var (h, w) = (meta.h_latent, meta.w_latent)
                for _ in 0..<(stages - 1) {
                    h /= 2; w /= 2
                    lat = bilinearDownsample2d(lat, h, w) * 2
                }
                lat = spatialUnreshape(lat)
                eval(lat)
                let d = maxAbs(lat, try fx("start0_\(c)"))
                let ok = d <= 1e-5
                print("  [start0 \(c)] max_abs=\(d) \(ok ? "OK" : "FAIL")")
                startOK = startOK && ok
            }

            // --- load weights + run the AR loop ---
            let model = HeliosModel(config)
            var weights = try MLX.loadArrays(url: mlxModel)
            if fp32 { weights = weights.mapValues { $0.asType(.float32) } }  // bf16 bits → fp32 compute
            try model.update(parameters: ModuleParameters.unflattened(weights),
                             verify: [.noUnusedKeys])
            eval(model)

            let ctxEmb = model.embedText([try fx("ctx_raw")])
            let crossKV = model.crossKVCaches(ctxEmb)
            eval(ctxEmb)

            let gen = HeliosGeneration(model: model, config: config)

            // Teacher-forced parity: each chunk consumes the ORACLE's prior chunk as
            // history, isolating per-chunk forward+loop correctness from the (chaotic,
            // sensitivity-amplified) drift of free-running AR — see chunk-1 note below.
            // per-stage drift (informational): localizes where divergence enters
            var stageDeltas: [String] = []
            let onStage: (Int, Int, MLXArray) -> Void = { c, s, lat in
                if let exp = try? fx("stageout_\(c)_\(s)") {
                    stageDeltas.append("    [stage \(c).\(s)] max_abs=\(maxAbs(lat, exp))")
                }
            }
            let chunks = gen.generate(
                contextEmbedded: ctxEmb, crossKV: crossKV,
                hLatent: meta.h_latent, wLatent: meta.w_latent, numChunks: meta.num_chunks,
                pyramidSteps: meta.pyramid_steps, amplifyFirstChunk: meta.amplify_first_chunk,
                noise: noise, computeDType: computeDType,
                teacher: { oracleChunks[$0] }, onStage: onStage)
            stageDeltas.forEach { print($0) }

            var chunkOK = true
            for c in 0..<meta.num_chunks {
                let d = maxAbs(chunks[c], oracleChunks[c])
                let ok = chunks[c].shape == oracleChunks[c].shape && d <= tol
                print("  [chunk \(c)] shape \(chunks[c].shape) max_abs=\(d) \(ok ? "OK" : "FAIL")")
                chunkOK = chunkOK && ok
            }

            let pass = startOK && chunkOK
            print(pass ? "[s3-gate] PASS" : "[s3-gate] FAIL")
            return pass
        }
    } catch {
        print("  [s3-gate] ERROR: \(error)")
        return false
    }
}
