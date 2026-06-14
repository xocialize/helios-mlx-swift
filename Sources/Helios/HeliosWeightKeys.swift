// HeliosWeightKeys — the Helios-Distilled transformer key contract.
//
// Two halves, both verified against the real checkpoint headers (2026-06-14,
// BestWishYsh/Helios-Distilled, 1101 HF keys = 21 global + 40×27):
//   1. `ditKeys(...)`     — the CANONICAL (post-conversion) key set the Swift
//                           loader checks against. Reuses wan-core / Bernini
//                           canonical names so HeliosModel reuses wan-core block
//                           components, PLUS the 3 net-new history patchifiers.
//   2. `canonicalize(_:)` — the HF-diffusers → canonical rename applied by our
//                           Swift converter (and by the S0 gate, to prove the HF
//                           headers map 1:1 onto `ditKeys`). Re-port of the four
//                           renames in the oracle `convert_helios.sanitize_*`,
//                           adopting Bernini-canonical names (program decision):
//                             scale_shift_table      → modulation
//                             norm2.{w,b}            → norm3.{w,b}   (affine X-norm)
//                             norm_out.scale_shift_table → head.modulation
//                             proj_out               → head.head
//                           (+ attn1/attn2/ffn/time/text embedder renames).
//
// Loaders must refuse partial loads (0 missing / 0 unused) against `ditKeys`.

import Foundation

public enum HeliosWeightKeys {

    /// Quantized Linear paths within one block (same int4 scope as Bernini's
    /// recipe: attention q/k/v/o + FFN; norms / modulation / embeds / head bf16).
    static let blockLinearPaths = [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        "ffn.fc1", "ffn.fc2",
    ]

    /// Net-new Helios globals with no Wan equivalent: the 3 history patchifiers
    /// (Conv3d→Linear) + the current-chunk patch embedding.
    static let patchGlobals = ["patch_embedding", "patch_short", "patch_mid", "patch_long"]

    /// CANONICAL transformer key set (post-conversion). 21 global + layers×27.
    public static func ditKeys(layers: Int = 40, quantized: Bool = false) -> Set<String> {
        var keys = Set<String>()
        for i in 0..<layers {
            let b = "blocks.\(i)"
            for path in blockLinearPaths {
                keys.insert("\(b).\(path).weight")
                keys.insert("\(b).\(path).bias")
                if quantized {
                    keys.insert("\(b).\(path).scales")
                    keys.insert("\(b).\(path).biases")
                }
            }
            for attn in ["self_attn", "cross_attn"] {
                keys.insert("\(b).\(attn).norm_q.weight")
                keys.insert("\(b).\(attn).norm_k.weight")
            }
            keys.insert("\(b).modulation")          // ← scale_shift_table
            keys.insert("\(b).norm3.weight")        // ← norm2 (affine cross-attn norm)
            keys.insert("\(b).norm3.bias")
        }
        for g in patchGlobals + [
            "text_embedding_0", "text_embedding_1",
            "time_embedding_0", "time_embedding_1", "time_projection",
            "head.head",                            // ← proj_out
        ] {
            keys.insert("\(g).weight")
            keys.insert("\(g).bias")
        }
        keys.insert("head.modulation")              // ← norm_out.scale_shift_table
        return keys
    }

    /// Map ONE HuggingFace-diffusers transformer key to its canonical name.
    /// Pure string transform (the converter additionally reshapes the Conv3d
    /// patch weights 5D→2D, which does not change keys). Ordered prefix rules.
    public static func canonicalize(_ hfKey: String) -> String {
        var k = hfKey

        // — per-block attention / ffn / norm / modulation —
        if k.hasPrefix("blocks.") {
            k = k.replacingOccurrences(of: ".attn1.to_q", with: ".self_attn.q")
            k = k.replacingOccurrences(of: ".attn1.to_k", with: ".self_attn.k")
            k = k.replacingOccurrences(of: ".attn1.to_v", with: ".self_attn.v")
            k = k.replacingOccurrences(of: ".attn1.to_out.0", with: ".self_attn.o")
            k = k.replacingOccurrences(of: ".attn1.norm_q", with: ".self_attn.norm_q")
            k = k.replacingOccurrences(of: ".attn1.norm_k", with: ".self_attn.norm_k")
            k = k.replacingOccurrences(of: ".attn2.to_q", with: ".cross_attn.q")
            k = k.replacingOccurrences(of: ".attn2.to_k", with: ".cross_attn.k")
            k = k.replacingOccurrences(of: ".attn2.to_v", with: ".cross_attn.v")
            k = k.replacingOccurrences(of: ".attn2.to_out.0", with: ".cross_attn.o")
            k = k.replacingOccurrences(of: ".attn2.norm_q", with: ".cross_attn.norm_q")
            k = k.replacingOccurrences(of: ".attn2.norm_k", with: ".cross_attn.norm_k")
            k = k.replacingOccurrences(of: ".ffn.net.0.proj", with: ".ffn.fc1")
            k = k.replacingOccurrences(of: ".ffn.net.2", with: ".ffn.fc2")
            k = k.replacingOccurrences(of: ".norm2.", with: ".norm3.")  // affine X-attn norm
            k = k.replacingOccurrences(of: ".scale_shift_table", with: ".modulation")
            return k
        }

        // — globals —
        k = k.replacingOccurrences(
            of: "condition_embedder.time_embedder.linear_1", with: "time_embedding_0")
        k = k.replacingOccurrences(
            of: "condition_embedder.time_embedder.linear_2", with: "time_embedding_1")
        k = k.replacingOccurrences(
            of: "condition_embedder.time_proj", with: "time_projection")
        k = k.replacingOccurrences(
            of: "condition_embedder.text_embedder.linear_1", with: "text_embedding_0")
        k = k.replacingOccurrences(
            of: "condition_embedder.text_embedder.linear_2", with: "text_embedding_1")
        k = k.replacingOccurrences(of: "norm_out.scale_shift_table", with: "head.modulation")
        k = k.replacingOccurrences(of: "proj_out.", with: "head.head.")
        // patch_embedding / patch_short / patch_mid / patch_long pass through unchanged.
        return k
    }
}
