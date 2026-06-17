// HeliosVAEConverter — diffusers `AutoencoderKLWan` → wan-core canonical `WanVAE`.
//
// Helios ships its VAE in HuggingFace-diffusers format (`down_blocks`/`up_blocks`/
// `mid_block`/`quant_conv`, PyTorch conv layout), whereas wan-core's `WanVAE` expects
// the original-Wan canonical names (`downsamples`/`upsamples`/`middle`/`head`/`conv1`/
// `conv2`, channels-last convs). The two are the SAME 16-ch Wan VAE — verified by a
// permutation-invariant value match: all 194 tensors are bit-identical to the
// already-canonical Bernini `vae.safetensors`, with ZERO fine-tuning. So this
// converter is a pure RENAME + the standard conv transpose (no reshapes), and its
// output is bit-exact to Bernini's file (the `--convert-vae` gate proves it).
//
// Why convert at all if it equals Bernini's file: a shipped Helios ModelPackage (S7)
// should carry its OWN weights rather than depend on a sibling's checkpoint. The
// mapping itself is the reusable lesson — diffusers-sourced Wan VAEs all need it.
//
// The structural map (derived empirically from the value-matched pairs):
//   quant_conv → conv1 · post_quant_conv → conv2
//   {enc,dec}.conv_in → conv1 · conv_out → head.2 · norm_out → head.0
//   mid_block.resnets.{0,1} → middle.{0,2}.residual · mid_block.attentions.0 → middle.1
//   resnet internals: norm1→residual.0 · conv1→residual.2 · norm2→residual.3 ·
//                     conv2→residual.6 · conv_shortcut→shortcut
//   encoder.down_blocks.N → downsamples.N           (diffusers encoder is already flat)
//   decoder.up_blocks.X   → upsamples.M             (FLATTENED: resnet Y→4X+Y, upsampler→4X+3)
//   resample.1.* / time_conv.* are kept verbatim.
// Conv transpose: 5D Conv3d [O,I,kt,kh,kw]→[O,kt,kh,kw,I]; 4D Conv2d [O,I,kh,kw]→[O,kh,kw,I]
// (a no-op on the [C,1,1,1] norm `gamma`s, so the ndim rule is safe to apply blanket).

import Foundation
import MLX

public enum HeliosVAEConverter {

    /// Map one resnet/residual sub-path (`norm1.gamma`, `conv1.weight`, …) to canonical.
    /// Passes `resample.*` / `time_conv.*` through unchanged.
    private static func resnetSub(_ sub: [String]) -> [String] {
        let tail = Array(sub.dropFirst())
        switch sub[0] {
        case "norm1": return ["residual", "0"] + tail
        case "conv1": return ["residual", "2"] + tail
        case "norm2": return ["residual", "3"] + tail
        case "conv2": return ["residual", "6"] + tail
        case "conv_shortcut": return ["shortcut"] + tail
        default: return sub  // resample.1.*, time_conv.*
        }
    }

    /// diffusers AutoencoderKLWan key → wan-core canonical key.
    public static func canonicalizeVAEKey(_ key: String) -> String {
        let c = key.split(separator: ".").map(String.init)
        if c[0] == "post_quant_conv" { return (["conv2"] + c.dropFirst()).joined(separator: ".") }
        if c[0] == "quant_conv" { return (["conv1"] + c.dropFirst()).joined(separator: ".") }

        let pfx = c[0]                       // encoder | decoder
        let rest = Array(c.dropFirst())
        var mapped: [String]
        switch rest[0] {
        case "conv_in": mapped = ["conv1"] + rest.dropFirst()
        case "conv_out": mapped = ["head", "2"] + rest.dropFirst()
        case "norm_out": mapped = ["head", "0"] + rest.dropFirst()
        case "mid_block":
            let kind = rest[1], idx = rest[2], sub = Array(rest.dropFirst(3))
            if kind == "attentions" {
                mapped = ["middle", "1"] + sub
            } else {  // resnets: 0→middle.0, 1→middle.2
                mapped = ["middle", idx == "0" ? "0" : "2"] + resnetSub(sub)
            }
        case "down_blocks":  // encoder, already flat
            mapped = ["downsamples", rest[1]] + resnetSub(Array(rest.dropFirst(2)))
        case "up_blocks":    // decoder, flatten X*4 + (Y | 3 for upsampler)
            let x = Int(rest[1])!
            if rest[2] == "resnets" {
                let y = Int(rest[3])!
                mapped = ["upsamples", "\(x * 4 + y)"] + resnetSub(Array(rest.dropFirst(4)))
            } else {  // upsamplers.0.(resample.1.*|time_conv.*)
                mapped = ["upsamples", "\(x * 4 + 3)"] + rest.dropFirst(4)
            }
        default:
            mapped = rest
        }
        return ([pfx] + mapped).joined(separator: ".")
    }

    /// PyTorch conv → channels-last (no-op on norm gammas / 1-D biases).
    private static func transposed(_ v: MLXArray) -> MLXArray {
        switch v.ndim {
        case 5: return v.transposed(0, 2, 3, 4, 1)
        case 4: return v.transposed(0, 2, 3, 1)
        default: return v
        }
    }

    /// Convert `<srcDir>/vae/diffusion_pytorch_model.safetensors` → `outURL` (canonical MLX).
    /// fp32 by default (matches the published Wan VAE precision). CPU stream (watchdog).
    /// - Returns: the canonical key set written.
    @discardableResult
    public static func convertVAE(
        srcDir: URL, outURL: URL, dtype: DType = .float32
    ) throws -> Set<String> {
        var written = Set<String>()
        try Device.withDefaultDevice(.cpu) {
            let src = srcDir.appending(path: "vae").appending(path: "diffusion_pytorch_model.safetensors")
            let dict = try MLX.loadArrays(url: src)
            var out: [String: MLXArray] = [:]
            for (hfKey, value) in dict {
                out[canonicalizeVAEKey(hfKey)] = transposed(value).asType(dtype, stream: .cpu)
            }
            eval(Array(out.values))
            try MLX.save(arrays: out, url: outURL, stream: .cpu)
            written = Set(out.keys)
        }
        return written
    }
}
