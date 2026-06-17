// Helios layer primitives — 1:1 with the PR#21 oracle `attention.py` /
// `transformer.py`. Mirror wan-core's WanRMSNorm/WanLayerNorm idioms but kept
// LOCAL (the AR attention/block diverge, so Helios owns its primitives — only
// the public RoPE is reused from wan-core). See PORTING-SPEC.md §S1b.

import Foundation
import MLX
import MLXFast
import MLXNN
import WanCore

/// RMS norm over the FULL dim (qk_norm "rms_norm_across_heads": applied to the
/// [B,L,dim] projection before the head reshape, NOT per-head).
final class HeliosRMSNorm: Module, @unchecked Sendable {
    let eps: Float
    let weight: MLXArray

    init(_ dim: Int, eps: Float = 1e-6) {
        self.eps = eps
        self.weight = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// LayerNorm in float32 (mx.fast.layer_norm), optional affine. The parameterless
/// variant (norm1 self, norm-ffn) carries no checkpoint keys; only the affine
/// cross-attn norm does (canonical key `norm3`, oracle `norm2`).
final class HeliosLayerNorm: Module, @unchecked Sendable {
    let eps: Float
    let elementwiseAffine: Bool
    @ParameterInfo(key: "weight") var weight: MLXArray?
    @ParameterInfo(key: "bias") var bias: MLXArray?

    init(_ dim: Int, _ eps: Float = 1e-6, elementwiseAffine: Bool = false) {
        self.eps = eps
        self.elementwiseAffine = elementwiseAffine
        if elementwiseAffine {
            self._weight.wrappedValue = MLXArray.ones([dim])
            self._bias.wrappedValue = MLXArray.zeros([dim])
        } else {
            self._weight.wrappedValue = nil
            self._bias.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        elementwiseAffine
            ? MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
            : MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)
    }
}

/// Feed-forward: fc1 → GELU(tanh) → fc2. Oracle uses `nn.GELU(approx="tanh")`.
final class HeliosFFN: Module, @unchecked Sendable {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ dim: Int, _ ffnDim: Int) {
        self._fc1.wrappedValue = Linear(dim, ffnDim)
        self._fc2.wrappedValue = Linear(ffnDim, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xW = x.asType(linearDtype(fc1))
        return fc2(geluApproximate(fc1(xW)))
    }
}
