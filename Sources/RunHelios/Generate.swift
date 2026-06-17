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
        // `mlxModel` points at <dir>/model.safetensors → the Helios canonical dir (DiT + VAE).
        // umT5 comes from the shared sibling Wan checkpoint (`berniniDir`).
        let modelDir = mlxModel.deletingLastPathComponent()
        print("Loading Helios pipeline (DiT \(modelDir.lastPathComponent), umT5 \(berniniDir.lastPathComponent)) …")
        let tLoad = Date()
        let pipe = try await HeliosPipeline.fromPretrained(modelDir: modelDir, textEncoderDir: berniniDir)
        print(String(format: "  load (%.1fs)", -tLoad.timeIntervalSinceNow))

        print("Generating @ \(width)x\(height), \(numFrames)f, pyramid \(pyramidSteps)"
            + "\(amplify ? "+amplify" : ""), seed \(seed)")
        let tGen = Date()
        let video = try pipe.t2v(
            prompt: prompt, width: width, height: height, numFrames: numFrames,
            pyramidSteps: pyramidSteps, amplify: amplify, seed: seed)
        eval(video)
        print(String(format: "  generate+decode (%.1fs)  peak %.1f GB", -tGen.timeIntervalSinceNow,
                     Double(MLX.GPU.peakMemory) / 1e9))

        let t = video.dim(2)
        for i in 0..<t {
            try writePNG(video[0, 0..., i, 0..., 0...],
                         to: outDir.appending(path: String(format: "frame_%03d.png", i)))
        }
        let lo = video.min().item(Float.self), hi = video.max().item(Float.self)
        print("Wrote \(t) frame(s) → \(outDir.path)  video \(video.shape) range=[\(lo), \(hi)]")
        return true
    } catch {
        print("[generate] ERROR: \(error)")
        return false
    }
}
