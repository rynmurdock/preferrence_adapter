# Plan: monitoring → verified personalization metrics → overnight sweep

*2026-10-16. Three outcomes, in order. Each has a gate; we don't start the next until the gate passes.*

**Base code:** upstream `e89a168` (bf16 + LoRA r8, trains the oct14 checkpoint). Our additions stay in new
modules (`src/eval/…`) plus small hooks in `train.py`; no rewrites of upstream files.

**Hardware:** one RTX 4090 (24 GB), 31 GB RAM. Run jobs under a `systemd-run` memory cap.

## 1. Monitoring (TensorBoard)

**Logged today:**
- `loss/train`, `loss/train_step`, `loss/val`
- `train/lr`, `train/grad_norm`
- `sys/gpu_mem`
- 2 sample images

**To add**, at each validation step, on a fixed val panel (val_unseen, fixed people, histories and seeds):

| Tag | What | Model (already cached) | When |
| --- | --- | --- | --- |
| `quality/hps` | HPSv2 | `xswu/HPSv2` (CLIP-H) | every val |
| `quality/aesthetic` | LAION aesthetic v2 | `shunk031/aesthetics-predictor-v2…` (CLIP-L) | every val |
| `quality/pickscore` | PickScore (optional) | `yuvalkirstain/PickScore_v1` | every val |
| `align/clipscore` | Prompt alignment | CLIP-L | only once prompts ≠ "The scene." |
| `pers/rho_own`, `pers/rho_swap`, `pers/rho_gap` | ρ_gen, own vs swapped history | SigLIP2 (have) | every val |
| `pers/verifier_*` | Verifier scores (outcome 2) | ours | every val |
| `pers/diversity` | Mean pairwise SigLIP cos across people, same seed (collapse alarm) | SigLIP2 | every val |
| `pers/token_between_cos` | T1: adapter-token dependence on history | none | every val (cheap) |
| `pers/viper` | ViPer proxy metric | `EPFL-VILAB/Metric-ViPer` (Idefics2-8B, ~16 GB) | **offline only**: too big to share the GPU with training |

ViPer is Salehi et al., ECCV 2024 (EPFL VILAB), "ViPer: Visual Personalization of Generative Models via
Individual Preference Learning". Its metric is a fine-tuned Idefics2 that judges whether an image matches a user's
liked and disliked images.

**Gate 1:**
- One short training run (~1 h) logs every tag.
- Val time stays under 15% of wall time.
- The scorers reproduce known values on real PAMELA images: aesthetic correlates with mean PAMELA rating,
  and ρ on real held-out images is above 0.

## 2. Personalization statistics, revised

Correlation (ρ_gen) stays as the baseline. We add a **learned verifier**: a rating predictor
f(history, image) → rating, trained on **real PAMELA images only** (train split).

- **Without ID (in-context):** sees only the 32-rating history. This works for unseen people, which is our setting.
- **With ID:** adds a learned per-person embedding. Only possible for seen people (val_seen/test_seen). It is an
  upper reference: how much of each person's taste a history fails to capture. EXP18-B started this.

**Reliability checks, on real held-out images, before we trust it on generated images (Gate 2):**

| Check | Pass rule |
| --- | --- |
| Predicts held-out ratings of unseen people | Spearman > ρ of the SigLIP-similarity rule (beats our current metric) |
| Uses the history | own-history > swapped-history prediction accuracy, sign test p < 0.05 |
| With-ID vs without-ID gap on seen people | reported, no rule |
| Agreement with ViPer metric on real images | reported, no rule |

**Metrics for generated images:**
- verifier(own history, gen) − verifier(swapped history, gen)
- 27-way identification: which person's history scores this image highest

## 3. Overnight sweep

Only after Gates 1 and 2. 3–4 runs per night, each about 2–3 h at 512 px, all with identical monitoring.
Candidate axes (we pick at most 2 per night, with the hypothesis written down first):

1. LoRA on/off, or rank: does adapting the base model help personalization or only quality?
2. CFG negative: zeroed input vs score=1 history (Ryan's todo).
3. Adapter depth (3 blocks vs 1) or adapter-token count.
4. Score-embedding use: real scores vs a shuffled-score control. Tests whether scores are used at all.

**Read-out:** a table of final `pers/*` and `quality/*` per run, plus curves. A setting wins only if it improves
the personalization metric without losing more than X on HPS/aesthetic (X fixed before the run).

## Open decisions

- Train on this 4090 (fit unverified) or on Ryan's rig?
- Pull `e89a168` into your tree, or keep using a separate checkout?
