import Testing
@testable import Helios

// S0 structural unit tests — pure string/Set logic, no MLX eval, so they run
// under `swift test` regardless of the metallib. The full contract-vs-real-headers
// check is `RunHelios --s0-gate` (CLI, reads the checkpoint index.json).

@Suite struct HeliosWeightKeysTests {

    @Test func contractShape() {
        // 21 global + 40 blocks × 27 per-block = 1101 (bf16).
        let keys = HeliosWeightKeys.ditKeys(layers: 40)
        #expect(keys.count == 1101)
        let global = keys.filter { !$0.hasPrefix("blocks.") }
        #expect(global.count == 21)
        let block0 = keys.filter { $0.hasPrefix("blocks.0.") }
        #expect(block0.count == 27)
    }

    @Test func quantizedAddsScalesBiases() {
        // +4 keys (scales+biases) on each of the 10 block Linears × 40 blocks = +800.
        let bf16 = HeliosWeightKeys.ditKeys(layers: 40).count
        let q = HeliosWeightKeys.ditKeys(layers: 40, quantized: true).count
        #expect(q - bf16 == 10 * 2 * 40)
    }

    @Test func canonicalizeRenames() {
        let cases: [(String, String)] = [
            ("blocks.7.attn1.to_q.weight", "blocks.7.self_attn.q.weight"),
            ("blocks.7.attn1.to_out.0.bias", "blocks.7.self_attn.o.bias"),
            ("blocks.7.attn1.norm_k.weight", "blocks.7.self_attn.norm_k.weight"),
            ("blocks.7.attn2.to_v.weight", "blocks.7.cross_attn.v.weight"),
            ("blocks.7.ffn.net.0.proj.weight", "blocks.7.ffn.fc1.weight"),
            ("blocks.7.ffn.net.2.bias", "blocks.7.ffn.fc2.bias"),
            ("blocks.7.norm2.weight", "blocks.7.norm3.weight"),       // affine X-attn norm swap
            ("blocks.7.scale_shift_table", "blocks.7.modulation"),
            ("condition_embedder.time_embedder.linear_1.weight", "time_embedding_0.weight"),
            ("condition_embedder.time_proj.bias", "time_projection.bias"),
            ("condition_embedder.text_embedder.linear_2.weight", "text_embedding_1.weight"),
            ("norm_out.scale_shift_table", "head.modulation"),
            ("proj_out.weight", "head.head.weight"),
            ("patch_short.weight", "patch_short.weight"),             // net-new, unchanged
        ]
        for (hf, want) in cases {
            #expect(HeliosWeightKeys.canonicalize(hf) == want, "\(hf) -> \(HeliosWeightKeys.canonicalize(hf)) != \(want)")
        }
    }
}
