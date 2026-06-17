// S3b VAE-decode smoke: wire HeliosGeneration.decode through wan-core's 16-ch
// WanVAE on the GPU and confirm it produces finite, in-range frames from the
// oracle's chunk latents (the s3loop fixtures). This proves the decode path; it is
// a SMOKE, not parity. Uses the family-shared Wan 16-ch VAE (Helios reuses Wan's
// AutoencoderKLWan, vae_stride 4/8/8) pending a Helios-own VAE convert in the full
// S2b GPU pipeline. Runs on the default (Metal) stream — eval per chunk (watchdog).

import Foundation
import MLX
import MLXNN
import WanCore
import Helios

func runS3Decode(vaePath: URL, fixtures: URL) -> Bool {
    func fx(_ name: String) throws -> MLXArray {
        try loadNumpy(url: fixtures.appending(path: "\(name).npy"))
    }
    do {
        guard let metaData = try? Data(contentsOf: fixtures.appending(path: "meta.json")),
              let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any],
              let numChunks = meta["num_chunks"] as? Int
        else { print("[s3-decode] SKIP — meta.json not found"); return true }

        let config = HeliosConfig.heliosDistilled()
        // encoder:true so the full VAE checkpoint (enc+dec) loads with .noUnusedKeys;
        // decode() only exercises the decoder path.
        let vae = WanVAE(zDim: config.vaeZDim, encoder: true)
        let weights = try MLX.loadArrays(url: vaePath)
        try vae.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
        eval(vae)

        var chunks: [MLXArray] = []
        for c in 0..<numChunks { chunks.append(try fx("chunk_\(c)")) }

        let gen = HeliosGeneration(model: HeliosModel(config), config: config)
        let video = gen.decode(chunks, vae: vae)
        eval(video)

        let f = video.asType(.float32)
        let lo = f.min().item(Float.self)
        let hi = f.max().item(Float.self)
        let anyNaN = notEqual(f, f).any().item(Bool.self)  // NaN != NaN
        // [-1,1]-clamped, finite, non-degenerate (real spread, not a flat frame).
        let perChunk = (config.numLatentFramesPerChunk - 1) * config.vaeStride[0] + 1 - 1  // 32
        let expFrames = numChunks * perChunk
        let ok = !anyNaN && lo >= -1.001 && hi <= 1.001 && (hi - lo) > 0.05
            && video.dim(2) == expFrames && video.dim(1) == 3
        print("  [decode] video \(video.shape) range=[\(lo), \(hi)] nan=\(anyNaN)")
        print(ok ? "[s3-decode] PASS (finite, in-range, \(expFrames) frames)"
                 : "[s3-decode] FAIL")
        return ok
    } catch {
        print("  [s3-decode] ERROR: \(error)")
        return false
    }
}
