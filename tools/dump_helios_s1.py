#!/usr/bin/env python3
"""S1 component fixtures for the Helios Swift port (oracle = mlx-video PR #21).

Loads the oracle HeliosModel (HF transformer weights sanitized to ORACLE names
in-memory — no second on-disk convert), seeds inputs, and dumps:
  - rope freqs tables (freqs_t/h/w)             → gate vs wan-core ropeParams
  - one block forward (NO history, restrict OFF) → must equal a Wan block
  - full no-history forward (latents→output)     → patchify+embeds+blocks+output

Small grid (F=5, 16×16 → L=320) keeps the 40-block CPU forward quick. CPU stream
(the Python oracle trips the GPU watchdog otherwise — dev-machine note).

    /Volumes/DEV_ARCHIVE/helios-mlx-ref/.venv/bin/python tools/dump_helios_s1.py
"""
from pathlib import Path
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

HF = Path("/Volumes/DEV_ARCHIVE/weights/Helios-Distilled")
OUT = Path(__file__).resolve().parents[1] / "Tests/HeliosTests/Fixtures/s1"
OUT.mkdir(parents=True, exist_ok=True)

from mlx_video.models.helios.config import HeliosModelConfig
from mlx_video.models.helios.transformer import HeliosModel
from mlx_video.convert_helios import sanitize_helios_transformer_weights


def save(name, arr):
    a = arr.astype(mx.float32) if arr.dtype == mx.bfloat16 else arr
    np.save(OUT / f"{name}.npy", np.array(a))
    print(f"  {name}: {tuple(arr.shape)} {arr.dtype}")


def load_oracle():
    cfg = HeliosModelConfig.helios_distilled()
    model = HeliosModel(cfg)
    print("loading + sanitizing HF transformer (fp32)…")
    weights = {}
    for shard in sorted((HF / "transformer").glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    weights = sanitize_helios_transformer_weights(weights)
    weights = {k: v.astype(mx.bfloat16) for k, v in weights.items()}
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return cfg, model


def main():
    cfg, model = load_oracle()
    dim = cfg.dim

    # --- 1. RoPE freq tables (deterministic; gate vs wan-core ropeParams) ---
    ft, fh, fw = model.rope_freqs
    save("rope_freqs_t", ft)
    save("rope_freqs_h", fh)
    save("rope_freqs_w", fw)

    # --- inputs (seeded; RNG is bit-identical Python↔Swift) ---
    rng = np.random.default_rng(13)
    F, H, W = 5, 16, 16                       # latent grid → patch (1,2,2) → L=5*8*8=320
    latents = mx.array(rng.standard_normal((cfg.in_dim, F, H, W)).astype(np.float32) * 0.5)
    save("fwd_latents", latents)
    timestep = mx.array([900.0])
    save("fwd_timestep", timestep)
    ctx_raw = mx.array(rng.standard_normal((24, cfg.text_dim)).astype(np.float32) * 0.5)
    save("fwd_ctx_raw", ctx_raw)              # [24, 4096] raw text features (pre-embed)

    # --- 2. One block forward (no history, restrict OFF) ---
    hidden, grid = model._patchify(latents)   # [1, L, dim]
    L = hidden.shape[1]
    save("block_hidden_in", hidden)
    ctx_emb = model.embed_text([ctx_raw])     # [1, text_len, dim]
    save("block_ctx_emb", ctx_emb)
    frame_indices = mx.arange(grid[0])
    rope_cs = model.rope_freqs  # (tables)
    from mlx_video.models.helios.rope import helios_rope_precompute_cos_sin
    cos_f, sin_f = helios_rope_precompute_cos_sin(frame_indices, grid, model.rope_freqs, dtype=mx.bfloat16)
    save("block_rope_cos", cos_f)
    save("block_rope_sin", sin_f)
    # timestep_proj_expanded [1, L, 6, dim] (no history)
    t_emb = timestep.astype(mx.float32) * model._inv_freq
    t_emb = mx.concatenate([mx.cos(t_emb), mx.sin(t_emb)], axis=-1)[None, :]
    temb = model.time_embedding_1(model.time_embedding_act(model.time_embedding_0(t_emb)))
    tproj = model.time_projection(model.time_projection_act(temb)).reshape(1, 6, -1)
    tproj_exp = mx.broadcast_to(tproj[:, :, None, :], (1, 6, L, dim)).transpose(0, 2, 1, 3)
    save("block_tproj", tproj_exp)
    blk0 = model.blocks[0]
    blk_out = blk0(hidden, ctx_emb, tproj_exp, rotary_emb=(cos_f, sin_f),
                   original_context_length=L, frame_indices=frame_indices,
                   grid_size=grid, freqs=model.rope_freqs, cross_kv_cache=None)
    mx.eval(blk_out)
    save("block_out", blk_out)

    # --- 3. Full no-history forward ---
    out = model(latents, timestep, ctx_emb)
    mx.eval(out)
    save("fwd_out", out)
    print(f"OK -> S1 fixtures in {OUT} (grid {grid}, L={L})")


if __name__ == "__main__":
    main()
