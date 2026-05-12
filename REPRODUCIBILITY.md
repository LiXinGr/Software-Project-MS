# Reproducibility

This document records the expected local setup and the main commands for reproducing the final thesis evaluation artifacts. Paths are relative to the repository root unless otherwise noted.

## Environment Assumptions

- Linux workstation or cluster node with CUDA-capable NVIDIA GPUs.
- Separate Python or Conda environments are used for model families: DINOv3/projection scripts, DIFT, LightGlue, RePoseD, RoMa, RoMaV2, and UniDepth.
- Launcher scripts accept environment variable overrides such as `DINOV3_PY`, `DIFT_PY`, `LIGHTGLUE_PY`, `REPOSED_PY`, `ROMA_PY`, `ROMAV2_PY`, and `UNIDEPTH_PY`.
- Recommended shell settings for long runs:

```bash
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
```

## Required External Data

- PhotoTourism scenes under `datasets/phototourism/<scene>/dense/images` and `datasets/phototourism/<scene>/dense/sparse`.
- Pair files under `output/pairs_<scene>.txt`.
- MegaDepth/SfM training data for projection-head training and LightGlue training assets, referenced by the training scripts and local config files.
- UniDepth depth outputs under `output_v2/depth_raw/<scene>/`, or the input images needed to regenerate them with `scripts/generate_depth_maps.py`.

Large datasets and generated data are intentionally ignored by Git.

## Required Checkpoints

- Projection checkpoint from the Chapter 5 projection experiment:
  `experiments/ch5_b16_train_proj_temp005_h1024_d256/best.pt`
- Projection checkpoint used by the final selected matcher/evaluation:
  `experiments/phase2_projection_wide/best.pt`
- Final selected LightGlue matcher checkpoint:
  `external/glue-factory/outputs/training/stage2_dinov3_lg_151scenes_v1/checkpoint_best.tar`
- Baseline model checkpoints required by DIFT, DINOv3, SuperPoint/LightGlue, RoMa, RoMaV2, and UniDepth environments.

Large checkpoint files are not stored in Git.

## Final Selected Configuration

The final selected method is recorded in `configs/final_selected_pipeline.json`.

Summary:

- SuperPoint keypoints.
- DINOv3 ViT-L/16 block 16, `feat_level=-8`.
- DIFT Stable Diffusion v1.5, `t=0`, `up_ft_index=2`, ensemble size 2.
- Equal DINOv3/DIFT branch weights.
- Projection architecture `1664 -> 1024 -> 256`.
- 256-dimensional projected descriptors.
- Expanded 151-scene LightGlue checkpoint.
- LightGlue filter threshold `0.02`.
- Correspondence cap `2048`.
- Raw-image coordinate protocol.

## Main Commands

Run commands from the repository root.

### Chapter 5 Projection Evaluation

```bash
scripts/run_ch5_ch6_overnight.sh --plan-only
scripts/run_ch5_ch6_overnight.sh --launch
python scripts/aggregate_ch5_ch6_overnight.py --write
```

Expected outputs include:

- `output_v2/csv/chapter5_projection_final_wide_temp005.csv`
- `output_v2/reports/chapter5_projection_final_wide_temp005_report.md`
- `output_v2/reports/ch5_ch6_overnight_final_report.md`

### Chapter 6 LightGlue Threshold Selection

```bash
scripts/run_ch6_missing_threshold_sweeps.sh --plan-only
scripts/run_ch6_missing_threshold_sweeps.sh --launch
python scripts/aggregate_ch6_threshold_selection.py --write
```

Expected outputs:

- `output_v2/csv/chapter6_all_threshold_sweeps.csv`
- `output_v2/csv/chapter6_selected_thresholds.csv`
- `output_v2/reports/chapter6_threshold_selection_report.md`

### Chapter 6 Selected All-Mode Evaluation

```bash
scripts/run_ch6_selected_all_modes.sh --plan-only
scripts/run_ch6_selected_all_modes.sh --launch
python scripts/aggregate_ch6_selected_all_modes.py --write
```

Expected outputs:

- `output_v2/csv/chapter6_selected_all_modes.csv`
- `output_v2/csv/chapter6_selected_per_scene.csv`
- `output_v2/reports/chapter6_selected_all_modes_report.md`

### Final Selected Test Evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,3 scripts/run_final_selected_lg_test.sh --launch --gpus 0,1,3
python scripts/aggregate_final_selected_lg_test.py --write
```

Expected outputs:

- `output_v2/csv/final_selected_expanded151_lg_test_summary.csv`
- `output_v2/csv/final_selected_expanded151_lg_test_per_scene.csv`
- `output_v2/reports/final_selected_expanded151_lg_test_report.md`
- `output_v2/reports/final_evaluation_readiness_report.md`

### Runtime Benchmark

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_final_runtime_benchmark.sh --launch --gpu 0
CUDA_VISIBLE_DEVICES=0 scripts/run_final_selected_runtime_breakdown_detailed.sh --launch --gpu 0
```

Expected outputs:

- `output_v2/csv/final_runtime_comparison.csv`
- `output_v2/reports/final_runtime_comparison_report.md`
- `output_v2/csv/final_selected_runtime_breakdown_detailed.csv`
- `output_v2/csv/final_selected_runtime_breakdown_table.csv`
- `output_v2/reports/final_selected_runtime_breakdown_detailed.md`

### Final Figure Generation

```bash
python scripts/make_final_evaluation_figures.py
```

Expected outputs:

- `output_v2/figures/fig_final_test_all_modes_accuracy.png`
- `output_v2/figures/fig_final_test_all_modes_accuracy.pdf`
- `output_v2/figures/fig_final_test_per_scene_calibrated.png`
- `output_v2/figures/fig_final_test_per_scene_calibrated.pdf`
- `output_v2/figures/fig_final_test_gain_vs_splg_per_scene.png`
- `output_v2/figures/fig_final_test_gain_vs_splg_per_scene.pdf`
- `output_v2/figures/fig_final_selected_runtime_breakdown.png`
- `output_v2/figures/fig_final_selected_runtime_breakdown.pdf`

### Qualitative Figure Generation

```bash
python scripts/make_qualitative_match_figures.py
```

Expected outputs:

- `output_v2/figures/fig_final_test_qualitative_temple_nara_comparison.png`
- `output_v2/figures/fig_final_test_qualitative_temple_nara_comparison.pdf`
- `output_v2/figures/fig_final_test_qualitative_san_marco_comparison.png`
- `output_v2/figures/fig_final_test_qualitative_san_marco_comparison.pdf`
- `output_v2/reports/qualitative_match_visualization_report.md`

## Notes

Generated artifacts under `output/`, `output_v2/`, `cache/`, `experiments/`, and local backup folders should remain outside Git. Keep sanitized documentation and small configuration files in `docs/` and `configs/`.
