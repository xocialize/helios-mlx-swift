// S3 scheduler gate: HeliosScheduler trajectories vs PR#21, OFFLINE (no weights,
// pure scalar math → bit-exact). Gates global schedule + per-stage init +
// set_timesteps for 3 stages × {first-chunk amplify, later chunk}.

import Foundation
import MLX
import WanCore
import Helios

func runS3SchedGate(fixtures: URL) -> Bool {
    func fx(_ name: String) throws -> MLXArray {
        try loadNumpy(url: fixtures.appending(path: "\(name).npy"))
    }
    func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
    func arr(_ xs: [Double]) -> MLXArray { MLXArray(xs.map { Float($0) }) }

    let tol: Float = 1e-5
    var allPass = true
    do {
        let sched = HeliosScheduler()

        let dG = maxAbs(arr(sched.globalSigmas), try fx("global_sigmas"))
        let dT = maxAbs(arr(sched.globalTimesteps), try fx("global_timesteps"))
        let gOK = dG <= tol && dT <= tol
        print("  [global] sigmas max_abs=\(dG) timesteps max_abs=\(dT) \(gOK ? "OK" : "FAIL")")
        allPass = allPass && gOK

        let seqLens = [36, 144, 576]
        for stage in 0..<3 {
            for amplify in [false, true] {
                sched.setTimesteps(numInferenceSteps: 2, stageIndex: stage,
                                   imageSeqLen: seqLens[stage], isAmplifyFirstChunk: amplify)
                let tag = "s\(stage)_\(amplify ? "amp" : "plain")"
                let tsExp = try fx("ts_\(tag)")
                let sigExp = try fx("sig_\(tag)")
                let dts = maxAbs(sched.timestepsArray, tsExp)
                let dsig = maxAbs(sched.sigmasArray, sigExp)
                let ok = dts <= tol && dsig <= tol
                    && sched.timesteps.count == tsExp.dim(0)
                    && sched.sigmas.count == sigExp.dim(0)
                print("  [\(tag)] ts max_abs=\(dts) (\(sched.timesteps.count)) sig max_abs=\(dsig) (\(sched.sigmas.count)) \(ok ? "OK" : "FAIL")")
                allPass = allPass && ok
            }
        }
    } catch {
        print("  [s3-sched-gate] ERROR: \(error)")
        return false
    }
    print(allPass ? "[s3-sched-gate] PASS" : "[s3-sched-gate] FAIL")
    return allPass
}
