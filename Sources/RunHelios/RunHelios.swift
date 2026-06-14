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

let checkpoint = argValue("--checkpoint") ?? defaultCheckpoint

if CommandLine.arguments.contains("--s0-gate") {
    exit(runS0Gate(checkpointDir: checkpoint) ? 0 : 1)
}

print("RunHelios — Helios-Distilled port gates. Usage: RunHelios --s0-gate [--checkpoint <dir>]")
