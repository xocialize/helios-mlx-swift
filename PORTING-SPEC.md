# helios-mlx-swift — Porting Spec (Phase B1)

> Swift/MLX port of **Helios-Distilled** (PKU-YuanGroup), a 14B autoregressive minute-scale
> text-to-video model, onto the `wan-core` substrate. Second consumer of `wan-core` (after
> Bernini-R). Program context: `../WAN-STACK-PLAN.md` §B1 and `../HELIOS-PORTING-SPEC.md`
> (scoping). Oracle: mlx-video PR #21 (`dmunch/mlx-video @ helios`), pinned at
> `27902e7550546c9caa3ae9707f6bfa2bd23d0816`, on disk at `/Volumes/DEV_ARCHIVE/helios-mlx-ref`.
> Discipline: one phase = one parity gate = one commit; never proceed past a red gate; Metal
> gates run as `RunHelios --sN-gate` CLI modes (the metallib-in-xctest lesson).

## Headline: backbone is 100% `wan-core` reuse; the port IS the AR delta

Helios = **Wan2.2-A14B backbone (unchanged) + an autoregressive history delta.** The transformer
config resolves to the exact Wan2.2-A14B shape, so `WanModel` / `Transformer` / `Attention` /
cross-attn KV-cache / `WanVAE` / `UMT5EncoderModel` / `WeightLoader` are reused **as-is** from
`wan-core`. The net-new Swift is the memory/history machinery + the DMD scheduler + the AR loop.

## Verified config (resolved `transformer/config.json`, 2026-06-14)

| field | value | vs Wan2.2-A14B |
|---|---|---|
| dim (heads×head_dim) | 5120 (40 × 128) | ✅ identical |
| ffn_dim | 13824 | ✅ identical |
| num_layers / num_heads | 40 / 40 | ✅ identical |
| patch_size | (1, 2, 2) | ✅ identical |
| in/out_channels | 16 / 16 | ✅ identical |
| text_dim / freq_dim | 4096 / 256 | ✅ identical |
| eps / cross_attn_norm | 1e-6 / true | ✅ identical |
| qk_norm | `rms_norm_across_heads` | ✅ matches our per-head RMS norm_q/k |
| rope_dim | (44, 42, 42) | ✅ **bit-identical reuse** (settled 2026-06-13; `RoPE.swift` already builds per-axis tables) |
| rope_theta | 10000.0 | ✅ identical |
| **has_multi_term_memory_patch** | **true** | ➕ Helios-only (AR delta) |
| **history_scale_mode** | **`per_head`** | ➕ Helios-only |
| **guidance_cross_attn** | **true** | ➕ Helios-only (current-only cross-attn) |
| **zero_history_timestep** | **true** | ➕ Helios-only (history uses cached t=0 proj) |
| **is_amplify_history** | **false** | ➕ Helios-only (Easy-Anti-Drift amplify; OFF for distilled) |

Scheduler (from `HeliosModelConfig.helios_distilled()`): DMD, `shift=1.0`, `stages=3`,
`stage_range=[0,1/3,2/3,1]`, `gamma=1/3`, ~2–3 steps/stage, **no CFG**. AR: `history_sizes=[16,2,1]`,
`num_latent_frames_per_chunk=9` (33-frame chunks @ 4× temporal stride).

## S0 key contract — VERIFIED (2026-06-14, offline, from `index.json` + oracle `sanitize`)

Source: `BestWishYsh/Helios-Distilled` (public, ~138 GB; we pulled ~75 GB — `transformer/` +
`text_encoder/` + `vae/` + `tokenizer/`, **excluding the unused `transformer_ode/` teacher
variant**) → `/Volumes/DEV_ARCHIVE/weights/Helios-Distilled`. HF diffusers keys run through the
oracle `convert_helios.sanitize_helios_transformer_weights` map **1:1, lossless: 1101 HF →
1101 MLX** keys.

**Transformer contract: 1101 keys = 21 global + 40 blocks × 27 per block.**

- **Global (21):** `patch_embedding.{weight,bias}`, `patch_short/mid/long.{weight,bias}` (the 3
  history patchifiers — Conv3d→Linear reshaped), `time_embedding_0/1.{w,b}`,
  `time_projection.{w,b}`, `text_embedding_0/1.{w,b}`, `proj_out.{w,b}`, `output_norm_table`.
- **Per-block (27):** `self_attn.{q,k,v,o}.{w,b}` + `self_attn.norm_q/k.weight` +
  `cross_attn.{q,k,v,o}.{w,b}` + `cross_attn.norm_q/k.weight` + `ffn.fc1/fc2.{w,b}` +
  `norm2.{w,b}` + `scale_shift_table`. (`norm1`/`norm3` are parameterless LayerNorms — no keys,
  same as Wan; modulation rides `scale_shift_table`.)

**Naming deltas vs Bernini-canonical `wan-core` names** (the FOUR divergences — verified against
both block forwards, 2026-06-14):

| concept | Helios MLX (oracle) | `wan-core` / Bernini canonical |
|---|---|---|
| per-block 6-vec AdaLN table | `blocks.{i}.scale_shift_table` | `blocks.{i}.modulation` |
| **affine cross-attn norm** | `blocks.{i}.norm2.{w,b}` | `blocks.{i}.norm3.{w,b}` |
| output AdaLN table | `output_norm_table` | `head.modulation` |
| output projection | `proj_out` | `head.head` |
| history patchifiers | `patch_short/mid/long` | ➕ net-new (no Wan equivalent) |

The **norm index swap** is the subtle one: both lineages have norm1 (self, parameterless), one
affine cross-attn norm, and an ffn norm (parameterless). The affine norm — the only one with
checkpoint weights — is **`norm2` in Helios (diffusers)** but **`norm3` in wan-core (mlx-video)**.
The parameterless norms carry no keys, so only this affine norm needs the `norm2`→`norm3` rename.

**6-vector modulation is bit-compatible:** Helios `scale_shift_table` (1,6,dim) order
`[shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate]` == wan-core `modulation`
`[e0..e5]` semantics exactly → pure rename, no reshuffle.

**Decision (per program spec):** adopt **Bernini-canonical** names. The Swift weight-key adapter
(our re-port of `convert_helios`) applies all four renames at convert time, so the loader matches
`wan-core` canonical with **zero load-time remap**. `heliosDitKeys` = Bernini `ditKeys`
(modulation/head/norm3 canonical) **+** the 3 `patch_*` globals.
T5 (`t5Keys`) + VAE (`vae_keys.txt`) contracts reuse Bernini's unchanged (`loading.py` confirms the
oracle reuses Wan's T5/VAE loaders verbatim).

**Block reuse caveat (S1):** the Helios block is a **superset** of `wan-core`'s `WanAttentionBlock`,
not a verbatim reuse — it adds the AR paths (restricted self-attn, history/current cross-attn split,
fp32 residual accumulation). So we write a `HeliosAttentionBlock` that *reuses wan-core's
`WanSelfAttention`/`WanCrossAttention`/`WanFFN`/`WanLayerNorm` components* and matches the
non-AR forward (verified structurally identical: norm1→self+gate, affine-norm→cross, ffn-norm+gate).
The S1 gate is "one block, restricted-attn OFF, history len 0 → bit-equals a Wan block."

## The AR delta — net-new Swift (the real work, ~2.2k LOC)

All refs into the pinned oracle `models/helios/`.

1. **Multi-Term Memory Patchification** (`transformer.py`) — 3 history scales via Conv3d-as-Linear
   kernels (1,2,2)/(2,4,4)/(4,8,8); `history_sizes=[16,2,1]`; forward prepends
   `[hist_long, hist_mid, hist_short, current]`, unpatchifies only the trailing current tokens.
2. **Restricted self-attention** (`attention.py`) — history tokens attend only to history; current
   attends to history+current. A `restrict_self_attn` path on `wan-core` `WanSelfAttention`.
3. **Selective cross-attention** (`guidance_cross_attn=true`) — only current tokens get text
   cross-attn; history skips it.
4. **Zero-history timestep** (`zero_history_timestep=true`) — history tokens use a cached t=0
   time-projection, not the current sigma's. Precompute once.
5. **Helios RoPE** — per-axis freq tables are **free reuse** (bit-identical to `RoPE.swift`).
   Net-new = ONLY history RoPE at downsampled spatial res + concat `[hist_l,hist_m,hist_s,current]`.
6. **DMD distilled scheduler** (`scheduler.py`) — 3-stage coarse→fine pyramid, gamma=1/3 boundary
   correction; DMD step = x0-pred from flow then re-noise toward a `noisy_start` anchor; no CFG.

AR loop (`generate_helios.py`): 33-frame chunks; rolling 19-frame history buffer; per chunk split
history → 3 scales → denoise current through the pyramid → append → carry last frame.

## `wan-core` reuse map

| reuse (free) | net-new (`Helios` core) |
|---|---|
| `WanModel`/`Transformer`/`Attention`, `RoPE` (bit-identical), `UMT5EncoderModel`, `WanVAE` + `StreamingDecode` (E11-corrected), `WeightLoader`/`WeightKeys`, schedulers base | `HeliosConfig`, `HeliosModel` (history patchify + prepend + zero-t0 + restricted/selective attn flags), `HeliosRoPE` (history downsample + concat), `HeliosScheduler` (DMD pyramid), `HeliosGeneration` (chunk/history loop), the weight-key adapter, `MLXHelios` ModelPackage wrapper |

## Phase gates (S0→S7, the Bernini pattern)

| phase | gate |
|---|---|
| **S0** key contract | ✅ **DONE + GATED 2026-06-14**: `HeliosWeightKeys.{ditKeys,canonicalize}` encode the 1101-key canonical contract + the 4 renames. `RunHelios --s0-gate` against the real HF `index.json` = **PASS (1101 bijective, 0 missing/0 unused)**; 3 structural `swift test` cases green. Package builds on `wan-core`. T5/VAE reuse confirmed. |
| **S1** substrate | per-component forwards vs PR #21 fixtures (real weights, CPU stream): patch embed, one block (restricted attn off → must equal Wan block), RoPE, time/text embed. |
| **S2** AR core | per-chunk denoise golden (injected noise/context) through the DMD pyramid; history-injection + zero-t0 + restricted/selective attn each fixture-gated. |
| **S2b** GPU eyeball | tokenizer wiring + ONE real chunk on GPU. |
| **S3+** AR loop | full 33-frame-chunk autoregressive generation w/ rolling history; per-chunk parity. |
| **S5** VAE | reuse `wan-core` `WanVAE` + `StreamingDecode` (chunked decode matters more at minute scale). |
| **S6** int4 | per-pass cosine vs the oracle's own quantize gate (`_quantize_predicate`, same 10 block Linears as Bernini). |
| **S7** engine wrap | `HeliosPackage` → `textToVideo`; co-registers with Bernini (multi-package-per-capability); C0–C13. |

## Package topology (mirrors `bernini-r-mlx-swift`)

`Helios` core (on `WanCore`, no engine import) · `MLXHelios` wrapper (on `Helios` + `WanCore` +
`MLXToolKit`) · `RunHelios` executable (S0–S6 CLI gate modes + generation) · test targets. Path B
(novel architecture — `mlx-swift-lm` can't load the AR machinery), per the `mlx-swift-integration`
skill.

## Weights / oracle status

- ✅ PyTorch checkpoint downloaded (75 GB, `transformer_ode` excluded) →
  `/Volumes/DEV_ARCHIVE/weights/Helios-Distilled`.
- ⏳ Convert transformer → MLX (`convert_helios.py` bf16 + int4) for the oracle parity runs and as
  the Swift load target; re-port the converter in Swift with canonical renames for the shipping path.
- Oracle pinned at PR #21; `generate_helios.py` is the parity reference.
