import Foundation
import WanCore
import MLX
import MLXToolKit
import Helios

/// MLXEngine package: Helios-Distilled (PKU-YuanGroup, 14B autoregressive minute-scale
/// text-to-video on the Wan2.x substrate) exposing the canonical `textToVideo` surface.
/// Co-registers with Bernini-R under the same capability (multi-package-per-capability):
/// Bernini = high-quality short clips, Helios = long autoregressive video.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `HeliosConfiguration`, pages
/// the working set in with `load()`, drives `run(_:)`, reclaims with `unload()`. Lifecycle is
/// `InferenceActor`-isolated; the non-`Sendable` `HeliosPipeline` never crosses the boundary.
/// Cancellation is honored at chunk boundaries via the pipeline's `onChunk` seam.
@InferenceActor
public final class HeliosPackage: ModelPackage {
    public typealias Configuration = HeliosConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Helios-Distilled weights are Apache-2.0 (finetune of Wan2.1-T2V-14B, Apache);
            // this port code is Apache-2.0. Both layers permissive → C7/C8 pass.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "BestWishYsh/Helios-Distilled",
                revision: "main",
                tier: 1
            ),
            requirements: RequirementsManifest(
                // Footprints GROUNDED by the WANVideoTesting live app (Xcode agent, 2026-06-17,
                // M5 Max / 128 GB, peak phys_footprint). int4: 37 GB @128² → 54 GB @640×384 (native;
                // the AR loop's per-chunk working set is bounded by the fixed 19-frame history, so
                // longer videos don't raise the peak) → declare 56 GB. This is the viable production
                // path. bf16/fp32: 88 GB @128² and extrapolates WELL PAST 128 GB @640×384 (OOMs on a
                // 128 GB box — int4 is the only native-res path there) → declare 160 GB to gate bf16
                // to >128 GB pro hardware (safe under-admission; never admit-then-OOM; exact @640
                // peak unmeasured). Engine W1: the variant-unaware governor charged 48 GB for ALL
                // three runs (bf16 actually used 88) → config-aware footprint selection still needed.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 160_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 56_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max
            ),
            specialties: [
                SpecialtyWeight(.general, strength: 0.5),
            ],
            surfaces: [
                T2VContract.descriptor(
                    name: "helios-t2v",
                    summary: "Helios-Distilled autoregressive long-form text-to-video (14B, MLX). "
                        + "33-frame chunks with a rolling multi-scale history buffer; a 3-stage DMD "
                        + "pyramid denoises each chunk (no CFG, ~6 forwards/chunk). Native 640×384; "
                        + "minute-scale clips. fp32 DiT (the production default).",
                    modes: []
                ),
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: HeliosPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// Page the DiT + VAE in (umT5 stays paged-per-request inside the pipeline). Resolution:
    /// explicit `modelDirectory` → HF download of `repo`. The shared umT5 comes from
    /// `textEncoderDirectory` (or `HELIOS_T5_DIR`).
    public func load() async throws {
        guard pipeline == nil else { return }
        let modelDir: URL
        if let explicit = configuration.modelDirectory {
            modelDir = explicit
        } else {
            modelDir = try await WeightLoader.snapshotDownload(repoID: configuration.repo)
        }
        let t5Dir = configuration.textEncoderDirectory
            ?? ProcessInfo.processInfo.environment["HELIOS_T5_DIR"].map { URL(filePath: $0) }
            ?? modelDir
        pipeline = try await HeliosPipeline.fromPretrained(
            modelDir: modelDir, textEncoderDir: t5Dir, quantized: configuration.quant == .int4)
    }

    public func unload() async {
        pipeline = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let pipeline else { throw PackageError.notLoaded }
        switch request.capability {
        case .textToVideo:
            guard let t2v = request as? T2VRequest else {
                throw PackageError.configurationMismatch(
                    expected: "T2VRequest", got: String(describing: type(of: request)))
            }
            return try await runT2V(t2v, pipeline: pipeline)
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
    }

    private func runT2V(_ request: T2VRequest, pipeline: HeliosPipeline) async throws -> T2VResponse {
        guard request.initImage == nil, request.referenceImages?.isEmpty ?? true else {
            throw PackageError.configurationMismatch(
                expected: "text-only t2v (image/reference conditioning not in Helios v1)",
                got: "initImage/referenceImages")
        }
        try Task.checkCancellation()
        let fps = request.fps ?? 16
        let frames = try pipeline.t2v(
            prompt: request.prompt,
            width: request.width ?? 640,
            height: request.height ?? 384,
            numFrames: request.numFrames ?? 33,
            seed: request.seed ?? 42
        ) { _, _ in try Task.checkCancellation() }  // C13: per-chunk cancellation
        let mp4 = try await encodeMP4(frames: frames, fps: fps)
        return T2VResponse(
            video: Video(format: .mp4, data: mp4,
                         durationSeconds: Double(frames.dim(2)) / fps, frameRate: fps))
    }
}

extension HeliosPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(HeliosPackage.self)
    }
}
