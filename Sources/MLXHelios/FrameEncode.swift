// Canonical-artifact encoder for the Helios wrapper: decoded frames
// ([1, 3, T, H, W] in [-1, 1]) → H.264 MP4 `Video` (textToVideo). Pure
// AVFoundation/CoreVideo — no MLX beyond reading the frame tensor out.
// (Mirrors MLXBerniniR/FrameEncode.swift; PNG/t2i path omitted — Helios is t2v-only.)

import AVFoundation
import CoreVideo
import Foundation
import MLX
import MLXToolKit

enum FrameEncodeError: Error {
    case pixelBufferAllocation
    case writerSetup(String)
    case badFrames(String)
    case appendFailed(String)
    case writeIncomplete(String)
}

/// Frame tensor [3, H, W] in [-1, 1] → interleaved RGB bytes [H, W, 3].
private func rgbBytes(_ frame: MLXArray) -> (bytes: [UInt8], width: Int, height: Int) {
    let h = frame.dim(1), w = frame.dim(2)
    let rgb = clip((frame.asType(.float32) + 1) * Float(127.5), min: 0, max: 255)
        .asType(.uint8).transposed(1, 2, 0)
    eval(rgb)
    return (rgb.asArray(UInt8.self), w, h)
}

private func pixelBuffer(
    rgb: [UInt8], width: Int, height: Int, pool: CVPixelBufferPool
) throws -> CVPixelBuffer {
    var bufferOut: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &bufferOut)
    guard let buffer = bufferOut else { throw FrameEncodeError.pixelBufferAllocation }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<height {
        for x in 0..<width {
            let src = (y * width + x) * 3, dst = y * stride + x * 4
            base[dst + 0] = rgb[src + 2]  // B
            base[dst + 1] = rgb[src + 1]  // G
            base[dst + 2] = rgb[src + 0]  // R
            base[dst + 3] = 255           // A
        }
    }
    return buffer
}

/// Encode frames [1, 3, T, H, W] in [-1, 1] as H.264 MP4 at `fps` → bytes.
/// `@InferenceActor` so the non-`Sendable` frame tensor never crosses isolation.
@InferenceActor
func encodeMP4(frames: MLXArray, fps: Double) async throws -> Data {
    let t = frames.dim(2), h = frames.dim(3), w = frames.dim(4)
    guard frames.ndim == 5, t > 0, h > 0, w > 0 else {
        throw FrameEncodeError.badFrames("expected [1,3,T,H,W] with T>0, got \(frames.shape)")
    }
    let url = FileManager.default.temporaryDirectory
        .appending(path: "helios-\(UUID().uuidString).mp4")
    defer { try? FileManager.default.removeItem(at: url) }

    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(
        mediaType: .video,
        outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: w, AVVideoHeightKey: h,
        ])
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
        ])
    guard writer.canAdd(input) else { throw FrameEncodeError.writerSetup("cannot add video input") }
    writer.add(input)
    guard writer.startWriting() else {
        throw FrameEncodeError.writerSetup(writer.error?.localizedDescription ?? "startWriting")
    }
    writer.startSession(atSourceTime: .zero)

    let frameDuration = CMTime(value: CMTimeValue((600.0 / fps).rounded()), timescale: 600)
    for i in 0..<t {
        let (bytes, fw, fh) = rgbBytes(frames[0, 0..., i, 0..., 0...])
        guard let pool = adaptor.pixelBufferPool else {
            throw FrameEncodeError.writerSetup("no pixel buffer pool")
        }
        let buffer = try pixelBuffer(rgb: bytes, width: fw, height: fh, pool: pool)
        while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(5)) }
        guard adaptor.append(buffer, withPresentationTime:
                  CMTimeMultiply(frameDuration, multiplier: Int32(i))) else {
            throw FrameEncodeError.appendFailed(
                "frame \(i)/\(t), status=\(writer.status.rawValue), err=\(String(describing: writer.error))")
        }
    }
    input.markAsFinished()
    await writer.finishWriting()
    let exists = FileManager.default.fileExists(atPath: url.path)
    guard writer.status == .completed, exists else {
        throw FrameEncodeError.writeIncomplete(
            "status=\(writer.status.rawValue) err=\(String(describing: writer.error)) exists=\(exists)")
    }
    return try Data(contentsOf: url)
}
