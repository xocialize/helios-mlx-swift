// CancellationTests.swift — Helios through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before the
// notLoaded guard or weights); CAN-3 is the document of record for the checkpoint cadence:
// the core's AR loop bails cooperatively on `Task.isCancelled` per chunk AND per DMD
// denoise step (HeliosGeneration.generate), the per-chunk VAE decode bails per chunk
// (HeliosGeneration.decode + wan-core decodeStreaming), and the throwing seams —
// HeliosPipeline.t2v post-generate + the wrapper's post-core checkpoint
// (HeliosPackage.runT2V) — rethrow the CancellationError unchanged. (The pipeline's
// `try? onChunk` swallow is documented in code; it is NOT the cancellation mechanism.)

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXHelios

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = HeliosPackage(configuration: HeliosConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2VRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // textToVideo is a long-run capability — the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: HeliosPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: HeliosPackage.manifest,
            posture: .cadence([
                // Per DMD denoise step (and per AR chunk): Task.isCancelled bails in
                // HeliosGeneration.generate; HeliosPipeline.t2v's post-generate
                // checkpoint rethrows.
                .init(phase: .denoise, unit: .step),
                // Per decoded chunk: HeliosGeneration.decode bails per chunk and
                // wan-core decodeStreaming bails per temporal chunk; the wrapper's
                // post-core checkpoint discards the truncated video.
                .init(phase: .decode, unit: .chunk),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
