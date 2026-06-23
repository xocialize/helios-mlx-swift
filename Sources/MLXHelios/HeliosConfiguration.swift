import Foundation
import WanCore
import MLXToolKit

/// Init-time configuration for `HeliosPackage` (C9): which variant, where the canonical
/// Helios checkpoint lives (DiT + VAE), and where the shared umT5-XXL comes from. Per-request
/// prompt/size/frames ride the canonical `T2VRequest`, not here.
///
/// Resolution at `load()`:
///   1. `modelDirectory` (canonical Helios dir: `model.safetensors` + `vae.safetensors`)
///   2. HF download of `repo`.
/// `textEncoderDirectory` supplies the family-shared umT5-XXL (`t5_encoder.safetensors` +
/// `config.json`) — Helios reuses the same encoder (S5 proved the VAE bit-identical to the
/// sibling Wan checkpoint; the encoder is likewise shared). A Helios-OWN converted
/// `t5_encoder.safetensors` co-located in the model dir is the follow-up.
public struct HeliosConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// Canonical Helios checkpoint repo / provenance source.
    public var repo: String
    public var revision: String?
    /// Backbone quant of the chosen variant (bf16 = fp32-run DiT, or int4).
    public var quant: Quant
    /// Resolved canonical Helios dir (DiT + VAE). Environment-specific → not Codable.
    public var modelDirectory: URL?
    /// Shared umT5-XXL source dir. Environment-specific → not Codable.
    public var textEncoderDirectory: URL?
    /// Engine-chosen models root (future auto-materialization). Not Codable.
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "mlx-community/Helios-Distilled-MLX",
        revision: String? = nil,
        quant: Quant = .bf16,
        modelDirectory: URL? = nil,
        textEncoderDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.modelDirectory = modelDirectory
        self.textEncoderDirectory = textEncoderDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// The int4 variant (block Linears quantized; ~8.4 GB DiT on disk).
    public static var int4: HeliosConfiguration {
        HeliosConfiguration(repo: "mlx-community/Helios-Distilled-MLX-int4", quant: .int4)
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant
    }
}
