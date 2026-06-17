#!/usr/bin/env python3
"""S3 scheduler fixtures: HeliosScheduler timestep/sigma trajectories (PR#21).

Offline (NO weights) — the scheduler is pure scalar math, gated bit-exact. Dumps
set_timesteps(timesteps, sigmas) for the real distilled cases: 3 pyramid stages ×
{first chunk (amplify), later chunk}, at the 1/4-res pyramid image_seq_lens, plus
the global sigma schedule. pyramid_steps=[2,2,2], 3 stages.

    cd /Volumes/DEV_ARCHIVE/helios-mlx-ref && PYTHONPATH=. \
      /Volumes/DEV_ARCHIVE/bernini-r-mlx/.venv/bin/python /…/tools/dump_helios_s3_sched.py
"""
from pathlib import Path
import json
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
OUT = Path(__file__).resolve().parents[1] / "Tests/HeliosTests/Fixtures/s3"
OUT.mkdir(parents=True, exist_ok=True)

from mlx_video.models.helios.scheduler import HeliosScheduler


def save(name, arr):
    np.save(OUT / f"{name}.npy", np.array(arr.astype(mx.float32)))


def main():
    sched = HeliosScheduler(num_train_timesteps=1000, shift=1.0, stages=3,
                            stage_range=[0, 1 / 3, 2 / 3, 1], gamma=1 / 3)

    # global schedule + per-stage start/end (gate the init math)
    save("global_sigmas", mx.array(np.asarray(sched.global_sigmas)))
    save("global_timesteps", mx.array(np.asarray(sched.global_timesteps)))
    meta = {
        "start_sigmas": [sched.start_sigmas[i] for i in range(3)],
        "end_sigmas": [sched.end_sigmas[i] for i in range(3)],
        "ori_start_sigmas": [sched.ori_start_sigmas[i] for i in range(3)],
        "sigma_min": sched.sigma_min, "sigma_max": sched.sigma_max,
    }
    (OUT / "init_meta.json").write_text(json.dumps(meta, indent=2))

    # set_timesteps trajectories — 1/4-res pyramid seq lens for a 16×16 latent,
    # 9 latent frames, patch (1,2,2): stage 0=36, 1=144, 2=576.
    seq_lens = [36, 144, 576]
    cases = {}
    for stage in range(3):
        for amplify in (False, True):
            sched.set_timesteps(2, stage_index=stage, image_seq_len=seq_lens[stage],
                                is_amplify_first_chunk=amplify)
            tag = f"s{stage}_{'amp' if amplify else 'plain'}"
            save(f"ts_{tag}", sched.timesteps)
            save(f"sig_{tag}", sched.sigmas)
            cases[tag] = {"stage": stage, "seq_len": seq_lens[stage], "amplify": amplify,
                          "n_timesteps": int(sched.timesteps.shape[0]),
                          "n_sigmas": int(sched.sigmas.shape[0])}
    (OUT / "cases.json").write_text(json.dumps(cases, indent=2))
    print("OK -> S3 scheduler fixtures:", sorted(cases.keys()))


if __name__ == "__main__":
    main()
