// RunHelios — CLI gate modes for the Helios port (the metallib-in-xctest lesson:
// Metal-context gates run as executable modes, not SPM tests).
//
//   RunHelios --s0-gate   key contract: HF-diffusers transformer headers, run
//                         through HeliosWeightKeys.canonicalize, must equal
//                         HeliosWeightKeys.ditKeys() bijectively (0 missing / 0
//                         unused). Reads the checkpoint index.json (no big shards).
//
// S1+ gates (component forwards, AR chunk, DMD) land here as they're ported.

import Foundation
import MLX
import Helios

let defaultCheckpoint =
    "/Volumes/DEV_ARCHIVE/weights/Helios-Distilled"

func argValue(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

/// S0: the transformer key contract, against the real HF checkpoint headers.
func runS0Gate(checkpointDir: String) -> Bool {
    let idxURL = URL(filePath: checkpointDir)
        .appending(path: "transformer")
        .appending(path: "diffusion_pytorch_model.safetensors.index.json")
    guard let data = try? Data(contentsOf: idxURL),
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        let weightMap = json["weight_map"] as? [String: Any]
    else {
        print("[s0-gate] SKIP — index.json not found at \(idxURL.path)")
        return true
    }
    let hfKeys = Array(weightMap.keys)
    let canonical = Set(hfKeys.map { HeliosWeightKeys.canonicalize($0) })
    let expected = HeliosWeightKeys.ditKeys(layers: 40)

    let missing = expected.subtracting(canonical)   // expected but no HF source
    let unused = canonical.subtracting(expected)     // HF keys mapping outside contract

    print("[s0-gate] HF keys: \(hfKeys.count) -> canonical: \(canonical.count) | expected: \(expected.count)")
    if !missing.isEmpty {
        print("[s0-gate] MISSING (\(missing.count)): \(missing.sorted().prefix(12))")
    }
    if !unused.isEmpty {
        print("[s0-gate] UNUSED  (\(unused.count)): \(unused.sorted().prefix(12))")
    }
    let pass = missing.isEmpty && unused.isEmpty && hfKeys.count == 1101
    print(pass ? "[s0-gate] PASS (1101-key contract bijective)" : "[s0-gate] FAIL")
    return pass
}

/// S1a: convert the HF transformer → canonical MLX, then verify the written
/// headers equal HeliosWeightKeys.ditKeys() bijectively (the S0 contract, now
/// against the REAL converted output rather than the HF index).
func runConvert(checkpointDir: String, outURL: URL) -> Bool {
    do {
        print("[convert] \(checkpointDir)/transformer → \(outURL.path) (bf16, CPU stream)…")
        let t0 = Date()
        let written = try HeliosConverter.convertTransformer(
            srcDir: URL(filePath: checkpointDir), outURL: outURL)
        print(String(format: "[convert] wrote %d keys in %.1fs", written.count, -t0.timeIntervalSinceNow))

        let expected = HeliosWeightKeys.ditKeys(layers: 40)
        let missing = expected.subtracting(written)
        let unused = written.subtracting(expected)
        if !missing.isEmpty { print("[convert] MISSING (\(missing.count)): \(missing.sorted().prefix(12))") }
        if !unused.isEmpty { print("[convert] UNUSED  (\(unused.count)): \(unused.sorted().prefix(12))") }
        let pass = missing.isEmpty && unused.isEmpty
        print(pass ? "[convert] PASS (canonical headers == contract)" : "[convert] FAIL")
        return pass
    } catch {
        print("[convert] ERROR: \(error)")
        return false
    }
}

/// Convert the Helios diffusers VAE → canonical MLX, then GATE the output bit-exactly
/// against Bernini's already-canonical vae.safetensors (the perfect oracle: the VAE is
/// the same shared Wan 16-ch VAE). Proves the diffusers→canonical mapping is correct.
func runConvertVAE(checkpointDir: String, outURL: URL, oracleVAE: URL) -> Bool {
    do {
        print("[convert-vae] \(checkpointDir)/vae → \(outURL.path) (fp32, CPU stream)…")
        let t0 = Date()
        let written = try HeliosVAEConverter.convertVAE(srcDir: URL(filePath: checkpointDir), outURL: outURL)
        print(String(format: "[convert-vae] wrote %d keys in %.1fs", written.count, -t0.timeIntervalSinceNow))

        guard FileManager.default.fileExists(atPath: oracleVAE.path) else {
            print("[convert-vae] DONE (no oracle at \(oracleVAE.path) to gate against)")
            return true
        }
        return try Device.withDefaultDevice(.cpu) {
            let got = try MLX.loadArrays(url: outURL)
            let exp = try MLX.loadArrays(url: oracleVAE)
            let gk = Set(got.keys), ek = Set(exp.keys)
            let missing = ek.subtracting(gk), extra = gk.subtracting(ek)
            var worst: Float = 0
            for k in gk.intersection(ek) {
                let d = MLX.abs(got[k]!.asType(.float32) - exp[k]!.asType(.float32)).max().item(Float.self)
                worst = Swift.max(worst, d)
            }
            if !missing.isEmpty { print("[convert-vae] MISSING (\(missing.count)): \(missing.sorted().prefix(8))") }
            if !extra.isEmpty { print("[convert-vae] EXTRA (\(extra.count)): \(extra.sorted().prefix(8))") }
            let pass = missing.isEmpty && extra.isEmpty && gk.count == 194 && worst == 0
            print("[convert-vae] keys \(gk.count) vs oracle \(ek.count) | max_abs=\(worst)")
            print(pass ? "[convert-vae] PASS (194 keys, bit-exact vs Bernini canonical → shared VAE confirmed)"
                       : "[convert-vae] FAIL")
            return pass
        }
    } catch {
        print("[convert-vae] ERROR: \(error)")
        return false
    }
}

let checkpoint = argValue("--checkpoint") ?? defaultCheckpoint
let convertOut = URL(filePath:
    argValue("--out") ?? "/Volumes/DEV_ARCHIVE/weights/Helios-Distilled-MLX/model.safetensors")

let mlxModelURL = URL(filePath:
    argValue("--mlx") ?? "/Volumes/DEV_ARCHIVE/weights/Helios-Distilled-MLX/model.safetensors")

if CommandLine.arguments.contains("--s0-gate") {
    exit(runS0Gate(checkpointDir: checkpoint) ? 0 : 1)
}
if CommandLine.arguments.contains("--s1-gate") {
    let fixtures = URL(filePath: argValue("--fixtures") ?? "Tests/HeliosTests/Fixtures/s1")
    exit(runS1Gate(mlxModel: mlxModelURL, fixtures: fixtures) ? 0 : 1)
}
if CommandLine.arguments.contains("--s2-gate") {
    let fixtures = URL(filePath: argValue("--fixtures") ?? "Tests/HeliosTests/Fixtures/s2")
    exit(runS2Gate(mlxModel: mlxModelURL, fixtures: fixtures) ? 0 : 1)
}
if CommandLine.arguments.contains("--s3-sched-gate") {
    let fixtures = URL(filePath: argValue("--fixtures") ?? "Tests/HeliosTests/Fixtures/s3")
    exit(runS3SchedGate(fixtures: fixtures) ? 0 : 1)
}
// The production Helios DiT runs fp32 (per-forward parity at the precision floor);
// gate there by default. --bf16 opts into the loose functional bound.
let fp32 = !CommandLine.arguments.contains("--bf16")
if CommandLine.arguments.contains("--s3-gate") {
    let dir = fp32 ? "Tests/HeliosTests/Fixtures/s3loop_fp32" : "Tests/HeliosTests/Fixtures/s3loop"
    let fixtures = URL(filePath: argValue("--fixtures") ?? dir)
    exit(runS3Gate(mlxModel: mlxModelURL, fixtures: fixtures, fp32: fp32) ? 0 : 1)
}
if CommandLine.arguments.contains("--s3-localize") {
    let dir = fp32 ? "Tests/HeliosTests/Fixtures/s3loop_fp32" : "Tests/HeliosTests/Fixtures/s3loop"
    let fixtures = URL(filePath: argValue("--fixtures") ?? dir)
    exit(runS3Localize(mlxModel: mlxModelURL, fixtures: fixtures, fp32: fp32) ? 0 : 1)
}
if CommandLine.arguments.contains("--s3-decode") {
    let dir = fp32 ? "Tests/HeliosTests/Fixtures/s3loop_fp32" : "Tests/HeliosTests/Fixtures/s3loop"
    let fixtures = URL(filePath: argValue("--fixtures") ?? dir)
    let vae = URL(filePath: argValue("--vae")
        ?? "/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16/vae.safetensors")
    exit(runS3Decode(vaePath: vae, fixtures: fixtures) ? 0 : 1)
}
if CommandLine.arguments.contains("--generate") {
    let steps = argValue("--pyramid").map { $0.split(separator: ",").compactMap { Int($0) } } ?? [2, 2, 2]
    exit(runGenerate(
        prompt: argValue("--prompt") ?? "a cat playing the piano, cinematic",
        width: argValue("--width").flatMap(Int.init) ?? 128,
        height: argValue("--height").flatMap(Int.init) ?? 128,
        numFrames: argValue("--frames").flatMap(Int.init) ?? 33,
        seed: argValue("--seed").flatMap(UInt64.init) ?? 42,
        pyramidSteps: steps,
        amplify: !CommandLine.arguments.contains("--no-amplify"),
        mlxModel: mlxModelURL,
        berniniDir: URL(filePath: argValue("--bernini")
            ?? "/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16"),
        outDir: URL(filePath: argValue("--out-dir") ?? "/tmp/helios")) ? 0 : 1)
}
// (runGenerate bridges to async internally; no top-level await.)
if CommandLine.arguments.contains("--convert-vae") {
    let out = URL(filePath: argValue("--vae-out")
        ?? "/Volumes/DEV_ARCHIVE/weights/Helios-Distilled-MLX/vae.safetensors")
    let oracle = URL(filePath: argValue("--vae")
        ?? "/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16/vae.safetensors")
    try? FileManager.default.createDirectory(
        at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
    exit(runConvertVAE(checkpointDir: checkpoint, outURL: out, oracleVAE: oracle) ? 0 : 1)
}
if CommandLine.arguments.contains("--convert") {
    try? FileManager.default.createDirectory(
        at: convertOut.deletingLastPathComponent(), withIntermediateDirectories: true)
    exit(runConvert(checkpointDir: checkpoint, outURL: convertOut) ? 0 : 1)
}
if CommandLine.arguments.contains("--convert-int4") {
    let out = URL(filePath: argValue("--int4-out")
        ?? "/Volumes/DEV_ARCHIVE/weights/Helios-Distilled-MLX-int4/model.safetensors")
    try? FileManager.default.createDirectory(
        at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
    do {
        print("[convert-int4] \(mlxModelURL.path) → \(out.path) (int4 g64, CPU stream)…")
        let t0 = Date()
        let n = try HeliosConverter.quantizeTransformer(
            canonicalURL: mlxModelURL, outURL: out, config: .heliosDistilled())
        print(String(format: "[convert-int4] %d Linears quantized in %.1fs → PASS", n, -t0.timeIntervalSinceNow))
        exit(n == 40 * 10 ? 0 : 1)
    } catch { print("[convert-int4] ERROR: \(error)"); exit(1) }
}
if CommandLine.arguments.contains("--s6-gate") {
    let int4 = URL(filePath: argValue("--int4")
        ?? "/Volumes/DEV_ARCHIVE/weights/Helios-Distilled-MLX-int4/model.safetensors")
    let fixtures = URL(filePath: argValue("--fixtures") ?? "Tests/HeliosTests/Fixtures/s2")
    exit(runS6Gate(int4Model: int4, fixtures: fixtures) ? 0 : 1)
}

print("RunHelios — Helios-Distilled port gates.")
print("  --s0-gate              key contract vs HF index.json")
print("  --s1-gate [--mlx <f>]  component parity vs oracle fixtures (real weights)")
print("  --s2-gate [--mlx <f>]  forward+history parity vs oracle fixtures")
print("  --s3-sched-gate        DMD scheduler trajectories (offline)")
print("  --s3-gate [--bf16]     AR generation loop parity (fp32 default; injected noise)")
print("  --s3-decode [--vae <f>] VAE-decode smoke (reuse wan-core WanVAE; GPU)")
print("  --generate --prompt <s> [--width/--height/--frames/--seed/--pyramid a,b,c/--no-amplify/--out-dir]")
print("                         real t2v run on GPU (fp32 DiT, shared umT5+VAE) → PNG frames")
print("  --convert [--out <f>]  HF transformer → canonical MLX + header check")
print("  --convert-vae          diffusers VAE → canonical MLX + bit-exact gate vs Bernini")
print("  --convert-int4         quantize canonical transformer → int4 (block Linears)")
print("  --s6-gate [--int4 <f>] int4 vs bf16 forward cosine (≥0.99) on the S2 fixture")
print("  [--checkpoint <dir>]")
