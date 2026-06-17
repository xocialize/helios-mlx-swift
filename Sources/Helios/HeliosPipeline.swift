// HeliosPipeline — the high-level prompt → frames path, the reusable core the
// MLXHelios ModelPackage drives (and the RunHelios --generate CLI). Owns the
// component loads (DiT + VAE resident; umT5 paged per-request + evicted before the
// AR loop, the §2.4 lever) and the encode → AR generate → decode flow.
//
// Stream discipline (learned the hard way): weight loads + the bf16→fp32 upcast ride
// the CPU stream and are eval'd before any forward (else the lazy cast folds into the
// forward's command buffer → watchdog); int4 weights load on CPU but the forward runs
// on GPU. The DiT runs fp32 (production default; bf16 amplifies through the DMD pyramid).
//
// umT5 is the family-shared umT5-XXL — loaded from `textEncoderDir` (a sibling Wan
// checkpoint) for now; a Helios-own converted `t5_encoder.safetensors` is the follow-up
// (the VAE got this treatment in S5; the encoder is the same shared model).

import Foundation
import MLX
import MLXNN
import MLXRandom
import WanCore
import Tokenizers

public final class HeliosPipeline: @unchecked Sendable {
    public let config: HeliosConfig
    let model: HeliosModel
    let vae: WanVAE
    let gen: HeliosGeneration
    let textEncoderDir: URL          // shared umT5-XXL source (sibling Wan checkpoint)
    let wanConfig: WanConfig          // for UMT5EncoderModel.fromConfig
    let tokenizer: any Tokenizer

    init(config: HeliosConfig, model: HeliosModel, vae: WanVAE, textEncoderDir: URL,
         wanConfig: WanConfig, tokenizer: any Tokenizer) {
        self.config = config
        self.model = model
        self.vae = vae
        self.gen = HeliosGeneration(model: model, config: config)
        self.textEncoderDir = textEncoderDir
        self.wanConfig = wanConfig
        self.tokenizer = tokenizer
    }

    /// Load DiT (bf16→fp32 or int4) + VAE from `modelDir` (Helios canonical:
    /// model.safetensors + vae.safetensors). umT5 stays unloaded (paged per request).
    public static func fromPretrained(
        modelDir: URL, textEncoderDir: URL, quantized: Bool = false,
        tokenizerRepo: String = umt5TokenizerRepo
    ) async throws -> HeliosPipeline {
        let config = HeliosConfig.heliosDistilled()
        let tokenizer = try await AutoTokenizer.from(pretrained: tokenizerRepo)
        let wanConfig = try WanConfig.load(from: textEncoderDir.appending(path: "config.json"))

        let (model, vae) = try Device.withDefaultDevice(.cpu) { () -> (HeliosModel, WanVAE) in
            let m = HeliosModel(config)
            if quantized {
                WeightLoader.applyQuantization(to: m, quantization: WanQuantization(groupSize: 64, bits: 4))
            }
            var weights = try MLX.loadArrays(url: modelDir.appending(path: "model.safetensors"))
            if !quantized { weights = weights.mapValues { $0.asType(.float32) } }  // production fp32 DiT
            try m.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
            eval(m)  // materialize the upcast NOW (watchdog)

            let v = WanVAE(zDim: config.vaeZDim, encoder: true)
            let vw = try MLX.loadArrays(url: modelDir.appending(path: "vae.safetensors"))
            try v.update(parameters: ModuleParameters.unflattened(vw), verify: [.noUnusedKeys])
            eval(v)
            return (m, v)
        }
        return HeliosPipeline(config: config, model: model, vae: vae,
                              textEncoderDir: textEncoderDir, wanConfig: wanConfig, tokenizer: tokenizer)
    }

    /// Encode a prompt with the shared umT5-XXL, then EVICT the encoder (§2.4) so it
    /// never co-resides with the DiT during the AR loop. Returns raw umT5 features.
    func encode(_ prompt: String) throws -> MLXArray {
        var enc: UMT5EncoderModel? = try Device.withDefaultDevice(.cpu) { () -> UMT5EncoderModel in
            let e = UMT5EncoderModel.fromConfig(wanConfig)
            let t5 = try MLX.loadArrays(url: textEncoderDir.appending(path: "t5_encoder.safetensors"))
                .mapValues { $0.asType(.float32) }
            try e.update(parameters: ModuleParameters.unflattened(t5), verify: [.noUnusedKeys])
            eval(e)
            return e
        }
        let ctx = encodeText(encoder: enc!, tokenizer: tokenizer, prompt: prompt, textLen: config.textLen)
        eval(ctx)
        enc = nil
        MLX.Memory.clearCache()
        return ctx
    }

    /// Text → video latents → frames [1, 3, T, H, W] in [-1, 1].
    /// `onChunk` fires after each chunk (cancellation seam for the engine).
    public func t2v(
        prompt: String, width: Int = 128, height: Int = 128, numFrames: Int = 33,
        pyramidSteps: [Int] = [2, 2, 2], amplify: Bool = true, seed: UInt64 = 42,
        onChunk: ((Int, Int) throws -> Void)? = nil
    ) throws -> MLXArray {
        let context = try encode(prompt)
        let ctxEmb = model.embedText([context])
        let crossKV = model.crossKVCaches(ctxEmb)
        eval(ctxEmb)

        let (w, h, numChunks) = gen.align(
            width: width, height: height, numFrames: numFrames, pyramidStages: pyramidSteps.count)
        let hLat = h / config.vaeStride[1], wLat = w / config.vaeStride[2]
        MLXRandom.seed(seed)
        let chunks = gen.generate(
            contextEmbedded: ctxEmb, crossKV: crossKV, hLatent: hLat, wLatent: wLat,
            numChunks: numChunks, pyramidSteps: pyramidSteps, amplifyFirstChunk: amplify,
            noise: .live(patchSize: config.patchSize, gamma: Double(config.gamma)),
            computeDType: .float32,
            onStage: { c, s, _ in if s == 0 { try? onChunk?(c, numChunks) } })
        eval(chunks)
        return gen.decode(chunks, vae: vae)
    }
}
