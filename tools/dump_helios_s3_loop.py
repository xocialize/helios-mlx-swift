#!/usr/bin/env python3
"""S3b fixture: Helios autoregressive generation loop (generate_helios.py), oracle = PR#21.

Replicates the distilled-default AR loop body INLINE (no T5/VAE/tokenizer) so we can
CAPTURE the noise it consumes and inject the same realizations into the Swift loop.

Why inject: `sample_block_noise` draws from NUMPY (Cholesky-correlated) — NOT mx.random —
so it is not reproducible in Swift. The per-chunk initial `noise = mx.random.normal(...)`
IS mx.random (bit-identical Py↔Swift) but we inject it too so the gate is hermetic.

Reduced workload for CPU tractability (the per-step math is identical to production):
  64×64 px → 8×8 latent, 2 chunks (tests history carry + first-frame prefix),
  pyramid_steps=[1,1,1] (1 forward/stage, still exercises all 3 stages + upsample +
  block-noise mix + DMD step), amplify_first_chunk=False.

Dumps per chunk c: noise_c (full-res init), start0_c (after 1/4 downsample checkpoint),
blocknoise_c_s (stages 1,2), chunk_c (output latents). Plus ctx_raw + meta.json.

    cd /Volumes/DEV_ARCHIVE/helios-mlx-ref && \
    PYTHONPATH=. /Volumes/DEV_ARCHIVE/bernini-r-mlx/.venv/bin/python \
        /…/helios-mlx-swift/tools/dump_helios_s3_loop.py
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

# HELIOS_S3_FP32=1 → run the loop in fp32 (bf16 weight VALUES upcast to fp32 +
# fp32 activations), matching the production fp32-DiT path and removing the bf16
# op-order confound. Weights use the same bf16 bits as the bf16 fixtures so the
# only delta vs the Swift fp32 gate is implementation, not weight precision.
FP32 = os.environ.get("HELIOS_S3_FP32") == "1"
CDT = mx.float32 if FP32 else mx.bfloat16

HF = Path("/Volumes/DEV_ARCHIVE/weights/Helios-Distilled")
OUT = Path(__file__).resolve().parents[1] / (
    "Tests/HeliosTests/Fixtures/s3loop_fp32" if FP32 else "Tests/HeliosTests/Fixtures/s3loop")
OUT.mkdir(parents=True, exist_ok=True)

from mlx_video.models.helios.config import HeliosModelConfig
from mlx_video.models.helios.transformer import HeliosModel
from mlx_video.models.helios.scheduler import HeliosScheduler
from mlx_video.convert_helios import sanitize_helios_transformer_weights
from mlx_video.generate_helios import (
    sample_block_noise,
    _spatial_reshape,
    _spatial_unreshape,
    _bilinear_downsample_2d,
    _nearest_upsample_2d,
)

# ---- reduced gate config ----
SEED = 1234
WIDTH, HEIGHT = 64, 64
NUM_CHUNKS = 2
PYRAMID_STEPS = [1, 1, 1]
AMPLIFY_FIRST_CHUNK = False


def save(name, arr):
    a = arr.astype(mx.float32) if arr.dtype == mx.bfloat16 else arr
    np.save(OUT / f"{name}.npy", np.array(a))
    print(f"  {name}: {tuple(arr.shape)} {arr.dtype}")


def main():
    cfg = HeliosModelConfig.helios_distilled()
    model = HeliosModel(cfg)
    print("loading + sanitizing HF transformer…")
    weights = {}
    for shard in sorted((HF / "transformer").glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    weights = {k: v.astype(mx.bfloat16).astype(CDT)
               for k, v in sanitize_helios_transformer_weights(weights).items()}
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    # seeded ctx (stand-in for T5 output), embedded + cross-kv prepared as generate does
    rng_ctx = np.random.default_rng(7)
    ctx_raw = mx.array(rng_ctx.standard_normal((24, cfg.text_dim)).astype(np.float32) * 0.5)
    save("ctx_raw", ctx_raw)
    context_embedded = model.embed_text([ctx_raw])
    mx.eval(context_embedded)
    cross_kv_caches = model.prepare_cross_kv(context_embedded)
    mx.eval(*[v for kv in cross_kv_caches for v in kv])

    # --- dimension alignment (mirror generate_video) ---
    vae_stride_t, vae_stride_h, vae_stride_w = cfg.vae_stride
    num_latent_per_chunk = cfg.num_latent_frames_per_chunk  # 9
    num_stages = len(PYRAMID_STEPS)
    pyramid_factor = 2 ** (num_stages - 1)
    align_h = cfg.patch_size[1] * pyramid_factor * vae_stride_h
    align_w = cfg.patch_size[2] * pyramid_factor * vae_stride_w
    height = ((HEIGHT + align_h - 1) // align_h) * align_h
    width = ((WIDTH + align_w - 1) // align_w) * align_w
    h_latent = height // vae_stride_h
    w_latent = width // vae_stride_w
    print(f"  px {width}x{height} → latent {h_latent}x{w_latent}, {NUM_CHUNKS} chunks")

    mx.random.seed(SEED)

    history_sizes = cfg.history_sizes  # [16, 2, 1]
    num_history_frames = sum(history_sizes)  # 19
    history_latents = mx.zeros((cfg.in_dim, num_history_frames, h_latent, w_latent))

    total_indices = 1 + num_history_frames + num_latent_per_chunk
    indices = mx.arange(total_indices)
    idx_prefix = indices[:1]
    idx_long = indices[1:1 + history_sizes[0]]
    idx_mid = indices[1 + history_sizes[0]:1 + history_sizes[0] + history_sizes[1]]
    idx_1x = indices[1 + history_sizes[0] + history_sizes[1]:1 + num_history_frames]
    idx_short = mx.concatenate([idx_prefix, idx_1x])
    idx_current = indices[1 + num_history_frames:]

    scheduler = HeliosScheduler(
        num_train_timesteps=1000, shift=1.0, stages=3, gamma=1 / 3,
        use_dynamic_shifting=True,
    )

    all_latent_chunks = []
    image_latents_prefix = None

    for chunk_idx in range(NUM_CHUNKS):
        is_first = chunk_idx == 0

        hist_long, hist_mid, hist_1x = mx.split(
            history_latents[:, -num_history_frames:],
            [history_sizes[0], history_sizes[0] + history_sizes[1]], axis=1)

        if is_first:
            latents_prefix = mx.zeros((cfg.in_dim, 1, h_latent, w_latent))
        else:
            latents_prefix = image_latents_prefix
        hist_short = mx.concatenate([latents_prefix, hist_1x], axis=1)

        # --- initial noise (mx.random) — CAPTURED for injection ---
        noise = mx.random.normal((cfg.in_dim, num_latent_per_chunk, h_latent, w_latent))
        mx.eval(noise)
        save(f"noise_{chunk_idx}", noise)

        cur_h, cur_w = h_latent, w_latent
        latents = _spatial_reshape(noise, num_latent_per_chunk, cfg.in_dim)
        for _ in range(scheduler.stages - 1):
            cur_h //= 2
            cur_w //= 2
            latents = _bilinear_downsample_2d(latents, cur_h, cur_w) * 2
        latents = _spatial_unreshape(latents, num_latent_per_chunk, cfg.in_dim, cur_h, cur_w)
        mx.eval(latents)
        save(f"start0_{chunk_idx}", latents)

        start_point_list = [latents]

        for i_s in range(scheduler.stages):
            image_seq_len = (num_latent_per_chunk * cur_h * cur_w
                             // math.prod(cfg.patch_size))
            scheduler.set_timesteps(
                PYRAMID_STEPS[i_s], stage_index=i_s, image_seq_len=image_seq_len,
                is_amplify_first_chunk=(AMPLIFY_FIRST_CHUNK and is_first))
            timesteps = scheduler.timesteps

            if i_s > 0:
                cur_h *= 2
                cur_w *= 2
                latents = _spatial_reshape(latents, num_latent_per_chunk, cfg.in_dim)
                latents = _nearest_upsample_2d(latents, cur_h, cur_w)
                latents = _spatial_unreshape(latents, num_latent_per_chunk, cfg.in_dim, cur_h, cur_w)

                ori_sigma = 1 - scheduler.ori_start_sigmas[i_s]
                gamma = scheduler.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                block_noise = sample_block_noise(
                    1, cfg.in_dim, num_latent_per_chunk, cur_h, cur_w,
                    cfg.patch_size, gamma)
                mx.eval(block_noise)
                save(f"blocknoise_{chunk_idx}_{i_s}", block_noise)
                latents = alpha * latents + beta * block_noise
                start_point_list.append(latents)

            h_short, h_mid, h_long = hist_short, hist_mid, hist_long
            h_short_bf16 = h_short.astype(CDT)
            h_mid_bf16 = h_mid.astype(CDT)
            h_long_bf16 = h_long.astype(CDT)

            timestep_list = [int(t) for t in timesteps.tolist()]
            sigma_list = scheduler.sigmas.tolist()

            for idx, t_val in enumerate(timestep_list):
                timestep = mx.array(t_val, dtype=mx.int32)
                # localization: dump the FIRST forward's input + prediction (chunk 0)
                if chunk_idx == 0 and i_s == 0 and idx == 0:
                    save("predin_0_0_0", latents)
                noise_pred = model(
                    latents=latents.astype(CDT),
                    timestep=timestep,
                    encoder_hidden_states=context_embedded,
                    frame_indices=idx_current,
                    history_short=h_short_bf16, history_mid=h_mid_bf16, history_long=h_long_bf16,
                    history_short_indices=idx_short, history_mid_indices=idx_mid,
                    history_long_indices=idx_long,
                    cross_kv_caches=cross_kv_caches)
                if chunk_idx == 0 and i_s == 0 and idx == 0:
                    mx.eval(noise_pred)
                    save("pred_0_0_0", noise_pred)
                sigma_next = sigma_list[idx + 1] if idx < len(timestep_list) - 1 else None
                latents = scheduler.step_dmd(
                    model_output=noise_pred, sample=latents, cur_step=idx,
                    noisy_start=start_point_list[i_s],
                    sigma_t=sigma_list[idx], sigma_next=sigma_next)
                mx.eval(latents)

            mx.eval(latents)
            save(f"stageout_{chunk_idx}_{i_s}", latents)

        mx.eval(latents)
        all_latent_chunks.append(latents)
        save(f"chunk_{chunk_idx}", latents)

        history_latents = mx.concatenate([history_latents, latents], axis=1)
        if is_first and image_latents_prefix is None:
            image_latents_prefix = latents[:, 0:1, :, :]

    meta = {
        "seed": SEED, "width": width, "height": height,
        "h_latent": h_latent, "w_latent": w_latent,
        "num_chunks": NUM_CHUNKS, "pyramid_steps": PYRAMID_STEPS,
        "amplify_first_chunk": AMPLIFY_FIRST_CHUNK,
        "in_dim": cfg.in_dim, "num_latent_per_chunk": num_latent_per_chunk,
        "history_sizes": history_sizes,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"OK -> S3b loop fixtures in {OUT}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
