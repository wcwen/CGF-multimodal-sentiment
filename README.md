# CGF: Text Confidence-Guided Fusion for Multimodal Sentiment Analysis

Code and per-seed results for the paper **"Text Confidence-Guided Adaptive Gated
Fusion for Multimodal Sentiment Analysis"** (V7.2).

CGF is a multimodal sentiment analysis model in which a confidence score derived
from a text-only predictive distribution modulates a text-enhancement branch and
informs gates over acoustic and visual interaction features obtained through
text-guided cross-attention.

This repository contains the complete experimental pipeline: model implementation,
training/evaluation scripts for all paper tables and figures, per-seed result
files exactly as reported in the paper, and the MOSEI data preparation script.

## Repository layout

```
CGF-multimodal-sentiment/
├── src/                  # model code + all experiment scripts
│   ├── common.py         # data loading, loaders, training/evaluation utils
│   ├── models_v2.py      # CGF model and all reimplemented baselines
│   ├── build_v3_tables.py# regenerate paper tables from progress JSONs
│   ├── run_v3_main.py    # Table 2/3: main comparison, all models, 4 seeds
│   ├── run_v3_binary.py  # Table 2/3: binary classification runs
│   ├── run_v3_ablation.py    # Table 6 + Table A1: module ablation
│   ├── run_abl_perturb_v3.py # Table 7: ablation under text corruption
│   ├── run_fair_perturb_v3.py    # Table 4: text-feature corruption, main models
│   ├── run_fair_perturb_pooled_v3.py  # pooled-input baselines under corruption
│   ├── run_cgf_pooled_v3.py    # CGF with pooled (mean) inputs, clean + noise
│   ├── run_cgf_stopgrad_v3.py  # stop-gradient variant (blocks gradient through c)
│   ├── run_constantc_v3.py     # constant-confidence control (CMU-MOSI)
│   ├── run_shuffle_control.py  # shuffled-confidence control (CMU-MOSI)
│   ├── run_v3_paddingmask.py   # padding-mask sensitivity experiment
│   ├── run_simple_baselines_v3.py  # Table A2: text-only MLP / late-fusion baselines
│   ├── run_fig2_stats.py       # statistics behind Figure 2
│   ├── run_mosei_pilot.py      # CMU-MOSEI pilot run
│   └── run_mosei_controlled.py # CMU-MOSEI controlled confidence experiments
├── analysis/             # downstream analyses (read results/, write outputs here)
│   ├── paired_bootstrap.py     # paired bootstrap CIs (reads results/)
│   ├── risk_coverage.py        # risk–coverage / selective-prediction analysis
│   ├── input_change.py         # input-representation change analysis
│   └── export_mosi_records.py  # per-sample preds+confidence (needs trained weights)
├── prep/
│   └── prep_mosei.py     # build CMU-MOSEI aligned_50.pkl from official SDK .csd
├── results/              # per-seed progress JSONs, exactly as reported in the paper
├── sucai/
│   ├── CMU-MOSI/         # place mosi_aligned_50.pkl here (not in git)
│   └── CMU-MOSEI/        # place mosei_aligned_50.pkl here (not in git)
├── requirements.txt
└── LICENSE               # MIT
```

## Environment

Verified on Python 3.12.7 / CPU.

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -r requirements.txt
```

`prep/prep_mosei.py` additionally needs `transformers` and `h5py` (BERT feature
extraction and reading official SDK `.csd` files); these are not required for
training or evaluation.

## Data preparation

The datasets themselves are not redistributed (license). Download or build the
aligned feature files, then place them at:

- `sucai/CMU-MOSI/mosi_aligned_50.pkl` — from the standard aligned release of
  CMU-MOSI (2,199 segments; 1,284/229/686 split; BERT 768-d text,
  COVAREP 5-d audio, FACET 20-d vision, 50 aligned steps), e.g. the
  CMU-MultimodalSDK or the MMSA data release.
- `sucai/CMU-MOSEI/mosei_aligned_50.pkl` — the release above distributes the
  commonly used 22,856-segment version. The paper's 22,860-segment variant is
  built from the official SDK with:

  ```bash
  # 1. download the official computational sequences into repo root:
  #      mosei_raw/COVAREP.csd, FACET 4.2.csd, TimestampedWords.csd, Labels.csd,
  #      cmu_mosei_std_folds.py
  python prep/prep_mosei.py av     # stage 1: audio+visual alignment (resumable)
  python prep/prep_mosei.py bert   # stage 2: BERT text features (resumable, long)
  python prep/prep_mosei.py pack   # stage 3: assemble final pkl -> sucai/CMU-MOSEI/
  ```

  The script retains every labeled segment of the fold videos without
  additional filtering (16,327/1,871/4,662). The four extra segments relative
  to the commonly distributed version are not missing-modality, empty-text,
  duplicate, NaN-label, or out-of-range segments; see the paper, Section 4.1.
  Segment ID lists are in `results/mosei_split_manifest.json`.

## Reproducing the reported numbers

All training scripts take an optional time-budget argument in **minutes**
(default 240) and resume from checkpoints on interruption. Each model/seed run
takes ~3–4 min on CPU. Results are appended to a `progress_*.json` file next to
the script (i.e. in `src/`). The `results/` directory holds the exact files
behind the paper; re-running a script produces a fresh file in `src/`.

| Paper table/figure | Script | Result file (in results/) |
|---|---|---|
| Table 2/3 (main comparison, binary + ternary) | `run_v3_main.py`, `run_v3_binary.py` | `progress_ternary_v3.json`, `progress_binary_v3.json` |
| Table 4 (text corruption, sequence-input models) | `run_fair_perturb_v3.py` | `progress_fair_perturb_v3.json` |
| Table 5 / §4.4 (input representation: pooled inputs) | `run_cgf_pooled_v3.py`, `run_fair_perturb_pooled_v3.py` | `progress_cgf_pooled_v3.json`, `progress_fair_perturb_pooled_v3.json` |
| Table 6 (module ablation) | `run_v3_ablation.py` | `progress_ablation_v3.json` |
| Table A1 (ablation, binary) | `run_v3_ablation.py` | `progress_ablation_v3.json` |
| Table 7 (ablation under corruption) | `run_abl_perturb_v3.py` | `progress_abl_perturb_v3.json` |
| Table 8 / §4.5 (constant-confidence control) | `run_constantc_v3.py`, `run_mosei_controlled.py` | `progress_constantc_v3.json`, `progress_mosei_controlled.json` |
| Shuffled-confidence control (§4.5) | `run_shuffle_control.py` | `progress_shuffle_control.json` |
| Stop-gradient variant (§4.5) | `run_cgf_stopgrad_v3.py` | `progress_stopgrad_v3.json` |
| Padding-mask sensitivity (Appendix) | `run_v3_paddingmask.py` | `progress_paddingmask_v3.json` |
| Table A2 (text-only / late-fusion lower bounds) | `run_simple_baselines_v3.py` | `progress_simple_baselines_v3.json` |
| Figure 2 (corruption trends) | `run_fig2_stats.py` | `progress_fig2_stats_v3.json` |
| CMU-MOSEI pilot / controlled runs | `run_mosei_pilot.py`, `run_mosei_controlled.py` | `progress_mosei_pilot_v3.json`, `progress_mosei_controlled.json` |

Example (ternary main table, four seeds):

```bash
python src/run_v3_main.py 240
```

Regenerate paper tables from result files:

```bash
python src/build_v3_tables.py          # uses results/ JSONs; prints LaTeX-ready tables
```

## Downstream analyses

```bash
python analysis/paired_bootstrap.py    # paired bootstrap CIs -> analysis/paired_bootstrap.md
                                       # (the paper's CI table; committed copy included)
python analysis/risk_coverage.py       # risk–coverage analysis (needs results/ JSONs + data pkls)
python analysis/export_mosi_records.py # per-sample records (needs trained weights in src/saved_models/)
python analysis/input_change.py        # input-change analysis (needs MOSI pkl only)
```

## Notes

- All reported numbers are **mean ± std over four training seeds**
  (42, 7, 2026, 3407) on the fixed test split.
- No attention mask is applied: padded positions participate in cross-attention
  and temporal average pooling; a padding-mask sensitivity experiment is included
  (`run_v3_paddingmask.py`).
- Trained model weights (`src/saved_models/`, ~1.9 GB) are not committed; they
  are regenerated by the training scripts.

## Citation

```bibtex
@article{cgf2026,
  title   = {Text Confidence-Guided Adaptive Gated Fusion for Multimodal Sentiment Analysis},
  author  = {},  % to fill after acceptance
  journal = {},  % to fill after acceptance
  year    = {2026}
}
```
