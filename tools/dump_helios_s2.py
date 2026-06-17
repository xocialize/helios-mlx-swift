#!/usr/bin/env python3
"""S2 fixture: Helios model forward WITH history (the AR delta), oracle = PR#21.

Replicates generate_helios's history construction (seeded), then one model()
call with history_{short,mid,long} + indices → dumps inputs + output. Exercises:
multi-scale history patchify, restricted self-attn, selective cross-attn,
zero-history t0 projection, and frame-offset/downsampled history RoPE.

  history_sizes=[16,2,1] → hist_long 16f, hist_mid 2f, hist_1x 1f;
  hist_short = [prefix, hist_1x] = 2f. Indices: prefix 0, long 1..16, mid 17..18,
  1x 19, short [0,19], current 20..28. Small spatial (16×16), current 9 frames.

    cd /Volumes/DEV_ARCHIVE/helios-mlx-ref && \
    PYTHONPATH=. /Volumes/DEV_ARCHIVE/bernini-r-mlx/.venv/bin/python \
        /…/helios-mlx-swift/tools/dump_helios_s2.py
"""
from pathlib import Path
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

HF = Path("/Volumes/DEV_ARCHIVE/weights/Helios-Distilled")
OUT = Path(__file__).resolve().parents[1] / "Tests/HeliosTests/Fixtures/s2"
OUT.mkdir(parents=True, exist_ok=True)

from mlx_video.models.helios.config import HeliosModelConfig
from mlx_video.models.helios.transformer import HeliosModel
from mlx_video.convert_helios import sanitize_helios_transformer_weights


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
    weights = {k: v.astype(mx.bfloat16) for k, v in sanitize_helios_transformer_weights(weights).items()}
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    rng = np.random.default_rng(29)
    C, H, W = cfg.in_dim, 16, 16
    npc = cfg.num_latent_frames_per_chunk          # 9
    hs = cfg.history_sizes                          # [16, 2, 1]
    nhist = sum(hs)                                 # 19

    def seeded(shape):
        return mx.array(rng.standard_normal(shape).astype(np.float32) * 0.5)

    history_latents = seeded((C, nhist, H, W))
    hist_long, hist_mid, hist_1x = mx.split(history_latents, [hs[0], hs[0] + hs[1]], axis=1)
    prefix = seeded((C, 1, H, W))                   # non-first chunk image prefix
    hist_short = mx.concatenate([prefix, hist_1x], axis=1)   # 2 frames
    latents = seeded((C, npc, H, W))
    timestep = mx.array([900.0])
    ctx_raw = seeded((24, cfg.text_dim))

    # indices: prefix 0 | long 1..16 | mid 17..18 | 1x 19 | current 20..28
    total = 1 + nhist + npc
    idx = mx.arange(total)
    idx_long = idx[1:1 + hs[0]]
    idx_mid = idx[1 + hs[0]:1 + hs[0] + hs[1]]
    idx_1x = idx[1 + hs[0] + hs[1]:1 + nhist]
    idx_short = mx.concatenate([idx[:1], idx_1x])   # [0, 19]
    idx_current = idx[1 + nhist:]                    # [20..28]

    for n, a in [("latents", latents), ("timestep", timestep), ("ctx_raw", ctx_raw),
                 ("hist_short", hist_short), ("hist_mid", hist_mid), ("hist_long", hist_long),
                 ("idx_short", idx_short), ("idx_mid", idx_mid), ("idx_long", idx_long),
                 ("idx_current", idx_current)]:
        save(n, a)

    ctx_emb = model.embed_text([ctx_raw])
    out = model(
        latents, timestep, ctx_emb,
        frame_indices=idx_current,
        history_short=hist_short, history_mid=hist_mid, history_long=hist_long,
        history_short_indices=idx_short, history_mid_indices=idx_mid,
        history_long_indices=idx_long,
    )
    mx.eval(out)
    save("out", out)
    print(f"OK -> S2 history fixtures in {OUT}")


if __name__ == "__main__":
    main()
