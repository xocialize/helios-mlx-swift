// HeliosScheduler — DMD flow-matching with a 3-stage coarse→fine pyramid.
// 1:1 with PR#21 `scheduler.py`. All schedule math is computed in Swift `Double`
// (== numpy float64) before the float32 cast, so trajectories are bit-exact.
// Distilled: x0-pred from flow, re-noise toward the stage's noisy_start, no CFG.

import Foundation
import MLX

private func linspace(_ a: Double, _ b: Double, _ count: Int) -> [Double] {
    guard count > 1 else { return [a] }
    return (0..<count).map { a + (b - a) * Double($0) / Double(count - 1) }
}

public final class HeliosScheduler: @unchecked Sendable {
    let numTrainTimesteps: Int
    let shift: Double
    let stages: Int
    let stageRange: [Double]
    let gamma: Double
    let useDynamicShifting: Bool
    let baseImageSeqLen: Int
    let maxImageSeqLen: Int
    let baseShift: Double
    let maxShift: Double

    public private(set) var globalSigmas: [Double] = []
    public private(set) var globalTimesteps: [Double] = []
    private var timestepsPerStage: [[Double]] = []
    private var sigmasPerStage: [[Double]] = []
    private(set) var startSigmas: [Double] = []
    private(set) var endSigmas: [Double] = []
    private(set) var oriStartSigmas: [Double] = []
    private var timestepRatios: [(Double, Double)] = []

    // runtime (set by setTimesteps)
    public private(set) var timesteps: [Double] = []
    public private(set) var sigmas: [Double] = []
    private var stepIndex = 0

    public var sigmaMin: Double { globalSigmas.last ?? 0 }
    public var sigmaMax: Double { globalSigmas.first ?? 0 }
    public var timestepsArray: MLXArray { MLXArray(timesteps.map { Float($0) }) }
    public var sigmasArray: MLXArray { MLXArray(sigmas.map { Float($0) }) }

    public init(
        numTrainTimesteps: Int = 1000, shift: Double = 1.0, stages: Int = 3,
        stageRange: [Double] = [0, 1.0 / 3, 2.0 / 3, 1], gamma: Double = 1.0 / 3,
        useDynamicShifting: Bool = true, baseImageSeqLen: Int = 256,
        maxImageSeqLen: Int = 4096, baseShift: Double = 0.5, maxShift: Double = 1.15
    ) {
        self.numTrainTimesteps = numTrainTimesteps
        self.shift = shift
        self.stages = stages
        self.stageRange = stageRange
        self.gamma = gamma
        self.useDynamicShifting = useDynamicShifting
        self.baseImageSeqLen = baseImageSeqLen
        self.maxImageSeqLen = maxImageSeqLen
        self.baseShift = baseShift
        self.maxShift = maxShift
        initSigmas()
        initSigmasPerStage()
    }

    private func calculateShift(_ imageSeqLen: Int) -> Double {
        let m = (maxShift - baseShift) / Double(maxImageSeqLen - baseImageSeqLen)
        let b = baseShift - m * Double(baseImageSeqLen)
        return Double(imageSeqLen) * m + b
    }

    private static func timeShift(_ mu: Double, _ t: Double) -> Double {
        mu * t / (1 + (mu - 1) * t)
    }

    private func initSigmas() {
        let n = numTrainTimesteps
        let alphas = linspace(1, 1.0 / Double(n), n + 1)
        let raw = alphas.map { a -> Double in
            let s = 1.0 - a
            return shift * s / (1 + (shift - 1) * s)
        }
        // np.flip(...)[:-1]: reverse, then drop the last (post-flip) element.
        let flipped = Array(raw.reversed())
        globalSigmas = Array(flipped[0..<(flipped.count - 1)])  // length n
        globalTimesteps = globalSigmas.map { $0 * Double(n) }
    }

    private func initSigmasPerStage() {
        let n = numTrainTimesteps
        var stageDistance: [Double] = []
        startSigmas = .init(repeating: 0, count: stages)
        endSigmas = .init(repeating: 0, count: stages)
        oriStartSigmas = .init(repeating: 0, count: stages)

        for i in 0..<stages {
            let startIdx = max(Int(stageRange[i] * Double(n)), 0)
            let endIdx = min(Int(stageRange[i + 1] * Double(n)), n)
            var startSigma = globalSigmas[startIdx]
            let endSigma = endIdx < n ? globalSigmas[endIdx] : 0.0
            oriStartSigmas[i] = startSigma
            if i != 0 {
                let oriSigma = 1 - startSigma
                let corrected = (1 / ((1 + 1 / gamma).squareRoot() * (1 - oriSigma) + oriSigma)) * oriSigma
                startSigma = 1 - corrected
            }
            stageDistance.append(startSigma - endSigma)
            startSigmas[i] = startSigma
            endSigmas[i] = endSigma
        }

        let totDistance = stageDistance.reduce(0, +)
        for i in 0..<stages {
            let startRatio = i == 0 ? 0.0 : stageDistance[0..<i].reduce(0, +) / totDistance
            let endRatio = i == stages - 1 ? 1.0 - 1e-16 : stageDistance[0...i].reduce(0, +) / totDistance
            timestepRatios.append((startRatio, endRatio))
        }

        for i in 0..<stages {
            let (r0, r1) = timestepRatios[i]
            let tMax = Swift.min(globalTimesteps[Int(r0 * Double(n))], 999)
            let tMin = globalTimesteps[Swift.min(Int(r1 * Double(n)), n - 1)]
            timestepsPerStage.append(Array(linspace(tMax, tMin, n + 1)[0..<n]))
            sigmasPerStage.append(Array(linspace(0.999, 0, n + 1)[0..<n]))
        }
    }

    public func setTimesteps(
        numInferenceSteps: Int, stageIndex: Int, imageSeqLen: Int? = nil,
        isAmplifyFirstChunk: Bool = false
    ) {
        let nSteps = isAmplifyFirstChunk ? numInferenceSteps * 2 + 1 : numInferenceSteps + 1
        let stageTs = timestepsPerStage[stageIndex]
        var ts = linspace(stageTs.first!, stageTs.last!, nSteps)
        let stageSig = sigmasPerStage[stageIndex]
        var sig = linspace(stageSig.first!, stageSig.last!, nSteps)
        sig.append(0.0)

        // DMD trim: drop last timestep; sigmas = [sig[:-2], sig[-1:]]
        ts = Array(ts[0..<(ts.count - 1)])
        sig = Array(sig[0..<(sig.count - 2)]) + [sig.last!]

        if useDynamicShifting, let imageSeqLen {
            let mu = calculateShift(imageSeqLen)
            sig = sig.map { Self.timeShift(mu, $0) }
            let tMin = stageTs.min()!, tMax = stageTs.max()!
            ts = sig[0..<(sig.count - 1)].map { tMin + $0 * (tMax - tMin) }
        }
        timesteps = ts
        sigmas = sig
        stepIndex = 0
    }

    /// DMD step: x0 = sample - sigma_t*flow; re-noise toward noisy_start (except last).
    public func stepDmd(
        modelOutput: MLXArray, sample: MLXArray, curStep: Int, noisyStart: MLXArray,
        sigmaT: Double? = nil, sigmaNext: Double? = nil
    ) -> MLXArray {
        let flow = modelOutput.asType(.float32)
        let x = sample.asType(.float32)
        let st = sigmaT ?? sigmas[curStep]
        let x0 = x - Float(st) * flow
        if curStep < timesteps.count - 1 {
            let sn = sigmaNext ?? sigmas[curStep + 1]
            return (1 - Float(sn)) * x0 + Float(sn) * noisyStart.asType(.float32)
        }
        return x0
    }

    /// Euler step: x_{t-1} = x_t + (sigma_next - sigma)*v.
    public func step(modelOutput: MLXArray, sample: MLXArray, sigma: Double, sigmaNext: Double) -> MLXArray {
        sample + Float(sigmaNext - sigma) * modelOutput
    }

    /// Flow-matching noising: (1-sigma)*original + sigma*noise.
    public func addNoise(_ original: MLXArray, _ noise: MLXArray, _ sigma: Double) -> MLXArray {
        (1 - Float(sigma)) * original + Float(sigma) * noise
    }
}
