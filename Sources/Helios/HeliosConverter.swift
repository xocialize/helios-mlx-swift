// HeliosConverter — re-port of mlx-video `convert_helios.sanitize_helios_transformer_weights`.
//
// Loads the HF-diffusers Helios transformer shards and writes ONE canonical MLX
// `model.safetensors` whose keys match `HeliosWeightKeys.ditKeys()` (wan-core /
// Bernini canonical), so `HeliosModel` loads strictly with no remap layer.
//
// The transformer conversion is RENAME-ONLY (via HeliosWeightKeys.canonicalize)
// plus the patch-embedding Conv3d→Linear reshape ([O,I,kd,kh,kw] → [O, I*kd*kh*kw]);
// there are NO weight transposes. Defensive skips (timesteps_proj / dropout
// to_out.1 / rope buffers / norm_added_* / add_*_proj) mirror the oracle — none
// appear in Helios-Distilled's 1101-key checkpoint, but a future diffusers variant
// might carry them. Runs on the CPU stream (Metal-watchdog: a multi-GB save must
// not hold a Metal command buffer open).

import Foundation
import MLX
import MLXNN

public enum HeliosConverter {

    /// The int4 recipe's scope — the 10 heavyweight Linears per block (attn q/k/v/o +
    /// cross-attn q/k/v/o + ffn fc1/fc2). Norms / modulation / patch-embeds / head stay
    /// full precision. 1:1 with the oracle `convert_helios._quantize_predicate`.
    static let blockLinearSuffixes = [
        ".self_attn.q", ".self_attn.k", ".self_attn.v", ".self_attn.o",
        ".cross_attn.q", ".cross_attn.k", ".cross_attn.v", ".cross_attn.o",
        ".ffn.fc1", ".ffn.fc2",
    ]

    /// Quantize the canonical bf16 transformer → int4 `model.safetensors` (+scales/biases
    /// on the block Linears). Loads the real weights first, then `MLXNN.quantize` (so the
    /// produced scales reflect the trained weights, not zero-fill). CPU stream (watchdog).
    /// - Returns: number of Linears quantized.
    @discardableResult
    public static func quantizeTransformer(
        canonicalURL: URL, outURL: URL, config: HeliosConfig, bits: Int = 4, groupSize: Int = 64
    ) throws -> Int {
        var nQuant = 0
        try Device.withDefaultDevice(.cpu) {
            let model = HeliosModel(config)
            let weights = try MLX.loadArrays(url: canonicalURL)
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
            eval(model)
            MLXNN.quantize(model: model, groupSize: groupSize, bits: bits) { path, m in
                m is Linear && blockLinearSuffixes.contains { path.hasSuffix($0) }
            }
            let flat = model.parameters().flattened()
            nQuant = flat.filter { $0.0.hasSuffix(".scales") }.count
            let out = Dictionary(uniqueKeysWithValues: flat)
            eval(Array(out.values))
            try MLX.save(arrays: out, url: outURL, stream: .cpu)
        }
        return nQuant
    }

    /// HF keys that carry no canonical weight (consumed-and-dropped by the oracle).
    static func isSkippable(_ key: String) -> Bool {
        key.contains("timesteps_proj")
            || key.contains(".to_out.1.")        // dropout
            || key.hasPrefix("rope.")             // computed in-model
            || key.contains("norm_added_q") || key.contains("norm_added_k")
            || key.contains("add_k_proj") || key.contains("add_v_proj")
    }

    static let patchWeightKeys: Set<String> = [
        "patch_embedding.weight", "patch_short.weight", "patch_mid.weight", "patch_long.weight",
    ]

    /// Convert `<srcDir>/transformer/*.safetensors` → `outURL` (canonical MLX).
    /// - Returns: the canonical key set actually written.
    @discardableResult
    public static func convertTransformer(
        srcDir: URL, outURL: URL, dtype: DType = .bfloat16
    ) throws -> Set<String> {
        var written = Set<String>()
        try Device.withDefaultDevice(.cpu) {
            let transformerDir = srcDir.appending(path: "transformer")
            let shards = try FileManager.default
                .contentsOfDirectory(at: transformerDir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            guard !shards.isEmpty else {
                throw HeliosConverterError.noShards(transformerDir.path)
            }

            // Metal-watchdog discipline: cast + save run EXPLICITLY on the CPU
            // stream (a 14B-element fp32→bf16 cast + multi-GB disk write must not
            // ride a GPU command buffer — that times out the watchdog). Eval each
            // shard's bf16 result before the next mmap so the fp32 source is freed
            // (bounds peak ≈ accumulated bf16 + one shard's fp32, not the full fp32 set).
            var out: [String: MLXArray] = [:]
            for shard in shards {
                let dict = try MLX.loadArrays(url: shard)  // mmap-backed, lazy
                var shardOut: [MLXArray] = []
                for (hfKey, value) in dict {
                    if isSkippable(hfKey) { continue }
                    var v = value
                    // Conv3d patch weight [O,I,kd,kh,kw] → Linear [O, I*kd*kh*kw].
                    if patchWeightKeys.contains(hfKey), v.ndim == 5 {
                        v = v.reshaped(v.dim(0), -1)
                    }
                    let cast = v.asType(dtype, stream: .cpu)
                    out[HeliosWeightKeys.canonicalize(hfKey)] = cast
                    shardOut.append(cast)
                }
                eval(shardOut)  // CPU-materialize this shard; releases its fp32 mmap
            }
            try MLX.save(arrays: out, url: outURL, stream: .cpu)
            written = Set(out.keys)
        }
        return written
    }

    public enum HeliosConverterError: Error, CustomStringConvertible {
        case noShards(String)
        public var description: String {
            switch self {
            case .noShards(let p): return "no transformer/*.safetensors shards under \(p)"
            }
        }
    }
}
