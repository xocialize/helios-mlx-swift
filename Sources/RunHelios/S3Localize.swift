// S3b localization: isolate the FIRST forward of the AR loop (chunk 0, stage 0)
// — current latents at small res (2×2 grid → 1×1) with FULL-res (8×8) ZERO history.
// This is the unequal current/history resolution the S2 gate never exercised (S2 had
// both at 16×16). If pred matches, the forward is fine and the bug is loop/scheduler.

import Foundation
import MLX
import MLXNN
import WanCore
import Helios

func runS3Localize(mlxModel: URL, fixtures: URL, fp32: Bool = false) -> Bool {
    func fx(_ name: String) throws -> MLXArray {
        try loadNumpy(url: fixtures.appending(path: "\(name).npy"))
    }
    func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }
    let cdt: DType = fp32 ? .float32 : .bfloat16
    let tol: Float = fp32 ? 2e-3 : 5e-2
    do {
        return try Device.withDefaultDevice(.cpu) {
            let config = HeliosConfig.heliosDistilled()
            let c = config.inDim, npc = config.numLatentFramesPerChunk
            let hs = config.historySizes, nHist = hs.reduce(0, +)
            let hLat = 8, wLat = 8

            let model = HeliosModel(config)
            var weights = try MLX.loadArrays(url: mlxModel)
            if fp32 { weights = weights.mapValues { $0.asType(.float32) } }
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
            eval(model)

            let ctxEmb = model.embedText([try fx("ctx_raw")])
            let crossKV = model.crossKVCaches(ctxEmb)

            // constant indices (prefix 0 | long 1..16 | mid 17..18 | 1x 19 | current 20..28)
            let total = 1 + nHist + npc
            let idx = MLXArray((0..<total).map { Int32($0) })
            let idxLong = idx[1 ..< (1 + hs[0])]
            let idxMid = idx[(1 + hs[0]) ..< (1 + hs[0] + hs[1])]
            let idx1x = idx[(1 + hs[0] + hs[1]) ..< (1 + nHist)]
            let idxShort = concatenated([idx[0 ..< 1], idx1x])
            let idxCurrent = idx[(1 + nHist)...]

            // zero history at full res (chunk 0)
            let histLong = MLXArray.zeros([c, hs[0], hLat, wLat]).asType(cdt)
            let histMid = MLXArray.zeros([c, hs[1], hLat, wLat]).asType(cdt)
            let histShort = MLXArray.zeros([c, 1 + hs[2], hLat, wLat]).asType(cdt)

            // stage-0 timestep for the chunk-0 config (res 2×2 → imageSeqLen 9, no amplify)
            let sched = HeliosScheduler(
                numTrainTimesteps: config.numTrainTimesteps, shift: Double(config.shift),
                stages: config.stages, stageRange: config.stageRange.map { Double($0) },
                gamma: Double(config.gamma), useDynamicShifting: true)
            let imageSeqLen = npc * 2 * 2 / config.patchSize.reduce(1, *)
            sched.setTimesteps(numInferenceSteps: 1, stageIndex: 0,
                               imageSeqLen: imageSeqLen, isAmplifyFirstChunk: false)
            let timestep = MLXArray(Int32(Int(sched.timesteps[0])))
            print("  stage-0 t=\(sched.timesteps[0]) seqLen=\(imageSeqLen)")

            let latents = try fx("predin_0_0_0")  // == start0_0
            let pred = model(
                latents.asType(cdt), timestep: timestep,
                encoderHiddenStates: ctxEmb, frameIndices: idxCurrent,
                historyShort: histShort, historyMid: histMid, historyLong: histLong,
                historyShortIndices: idxShort, historyMidIndices: idxMid,
                historyLongIndices: idxLong, crossKVCaches: crossKV)
            eval(pred)
            let exp = try fx("pred_0_0_0")
            let d = maxAbs(pred, exp)
            let pass = pred.shape == exp.shape && d <= tol
            print("  [first-fwd] shape \(pred.shape) (exp \(exp.shape)) max_abs=\(d) \(pass ? "OK" : "FAIL")")
            print(pass ? "[s3-localize] PASS — forward fine; bug is loop/scheduler"
                       : "[s3-localize] FAIL — forward diverges at small-current/full-history res")
            return pass
        }
    } catch {
        print("  [s3-localize] ERROR: \(error)")
        return false
    }
}
