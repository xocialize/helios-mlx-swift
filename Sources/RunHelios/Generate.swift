// S2b: the first REAL Helios run in Swift — prompt → video, end-to-end on GPU.
// Not the shipping pipeline (that's the S7 MLXHelios ModelPackage); this is the
// CLI eyeball that proves umT5 encode + embedText + the AR loop + VAE decode
// compose on real (non-fixture) inputs.
//
// Component reuse: the umT5-XXL text encoder and the 16-ch WanVAE are shared
// across the whole Wan family (per the wan-video skill), so we load them from the
// already-canonical bernini weights; only the Helios transformer is Helios-own.
// fp32 DiT (the production default) — bf16 amplifies through the DMD pyramid to
// garbage (the S3b finding). umT5 is evicted after encode (it never co-resides
// with the DiT).

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXNN
import MLXRandom
import WanCore
import Helios
import Tokenizers

/// Write [3, H, W] in [-1, 1] → PNG.
func writePNG(_ frame: MLXArray, to url: URL) throws {
    let h = frame.dim(1), w = frame.dim(2)
    let rgb = clip((frame.asType(.float32) + 1) * 127.5, min: 0, max: 255)
        .asType(.uint8).transposed(1, 2, 0)
    eval(rgb)
    let bytes: [UInt8] = rgb.asArray(UInt8.self)
    let data = CFDataCreate(nil, bytes, bytes.count)!
    let image = CGImage(
        width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: w * 3,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
        provider: CGDataProvider(data: data)!, decode: nil, shouldInterpolate: false,
        intent: .defaultIntent)!
    let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

final class UncheckedBox<T>: @unchecked Sendable { var value: T; init(_ v: T) { value = v } }

/// Sync entry for the script: bridge to the async tokenizer load. The forwarded
/// closure captures only Sendable params, so it satisfies Swift 6's sending check.
func runGenerate(
    prompt: String, width: Int, height: Int, numFrames: Int, seed: UInt64,
    pyramidSteps: [Int], amplify: Bool, mlxModel: URL, berniniDir: URL, outDir: URL
) -> Bool {
    let sema = DispatchSemaphore(value: 0)
    let box = UncheckedBox(false)
    Task { @Sendable in
        box.value = await generateImpl(
            prompt: prompt, width: width, height: height, numFrames: numFrames, seed: seed,
            pyramidSteps: pyramidSteps, amplify: amplify,
            mlxModel: mlxModel, berniniDir: berniniDir, outDir: outDir)
        sema.signal()
    }
    sema.wait()
    return box.value
}

func generateImpl(
    prompt: String, width: Int, height: Int, numFrames: Int, seed: UInt64,
    pyramidSteps: [Int], amplify: Bool, mlxModel: URL, berniniDir: URL, outDir: URL
) async -> Bool {
    do {
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let config = HeliosConfig.heliosDistilled()
        let wanConfig = try WanConfig.load(from: berniniDir.appending(path: "config.json"))

        do {
            // 1. Tokenizer + umT5 encode (shared umT5-XXL), then EVICT the encoder.
            let tokenizer = try await AutoTokenizer.from(pretrained: umt5TokenizerRepo)
            print("Encoding prompt with umT5 …")
            let tEnc = Date()
            let context: MLXArray = try {
                var enc: UMT5EncoderModel? = UMT5EncoderModel.fromConfig(wanConfig)
                let t5 = try MLX.loadArrays(url: berniniDir.appending(path: "t5_encoder.safetensors"))
                    .mapValues { $0.asType(.float32) }
                try enc!.update(parameters: ModuleParameters.unflattened(t5), verify: [.noUnusedKeys])
                eval(enc!)  // materialize the bf16→fp32 upcast NOW; else the lazy 11GB cast
                            // folds into the encode command buffer and trips the GPU watchdog
                let c = encodeText(encoder: enc!, tokenizer: tokenizer, prompt: prompt, textLen: config.textLen)
                eval(c)
                enc = nil
                MLX.GPU.clearCache()
                return c
            }()
            print(String(format: "  context %@  (%.1fs)", "\(context.shape)", -tEnc.timeIntervalSinceNow))

            // 2. Helios transformer (fp32 = production DiT).
            print("Loading Helios transformer (fp32) …")
            let tLoad = Date()
            let model = HeliosModel(config)
            let weights = try MLX.loadArrays(url: mlxModel).mapValues { $0.asType(.float32) }
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
            eval(model)
            let ctxEmb = model.embedText([context])
            let crossKV = model.crossKVCaches(ctxEmb)
            eval(ctxEmb)
            print(String(format: "  load+embed (%.1fs)", -tLoad.timeIntervalSinceNow))

            // 3. AR generation (live noise, fp32, GPU).
            let gen = HeliosGeneration(model: model, config: config)
            let (w, h, numChunks) = gen.align(
                width: width, height: height, numFrames: numFrames, pyramidStages: pyramidSteps.count)
            let hLat = h / config.vaeStride[1], wLat = w / config.vaeStride[2]
            print("Generating \(numChunks) chunk(s) @ \(w)x\(h) (latent \(hLat)x\(wLat)), "
                + "pyramid \(pyramidSteps)\(amplify ? "+amplify" : ""), seed \(seed)")
            MLXRandom.seed(seed)
            let tGen = Date()
            let chunks = gen.generate(
                contextEmbedded: ctxEmb, crossKV: crossKV, hLatent: hLat, wLatent: wLat,
                numChunks: numChunks, pyramidSteps: pyramidSteps, amplifyFirstChunk: amplify,
                noise: .live(patchSize: config.patchSize, gamma: Double(config.gamma)),
                computeDType: .float32)
            eval(chunks)
            print(String(format: "  generate (%.1fs)  peak %.1f GB", -tGen.timeIntervalSinceNow,
                         Double(MLX.GPU.peakMemory) / 1e9))

            // 4. VAE decode (shared 16-ch WanVAE).
            print("Decoding with VAE …")
            let vae = WanVAE(zDim: config.vaeZDim, encoder: true)
            let vaeW = try MLX.loadArrays(url: berniniDir.appending(path: "vae.safetensors"))
            try vae.update(parameters: ModuleParameters.unflattened(vaeW), verify: [.noUnusedKeys])
            eval(vae)
            let video = gen.decode(chunks, vae: vae)  // [1, 3, T, H, W] in [-1, 1]
            eval(video)

            let t = video.dim(2)
            for i in 0..<t {
                try writePNG(video[0, 0..., i, 0..., 0...],
                             to: outDir.appending(path: String(format: "frame_%03d.png", i)))
            }
            print("Wrote \(t) frame(s) → \(outDir.path)")
            let lo = video.min().item(Float.self), hi = video.max().item(Float.self)
            print("  video \(video.shape) range=[\(lo), \(hi)]")
            return true
        }
    } catch {
        print("[generate] ERROR: \(error)")
        return false
    }
}
