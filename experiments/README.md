# Experiment Tracking

Each experiment run is logged as a JSON file in this directory by `log_experiment.py`.

## Naming convention

```
experiments/{run_id}.json
```

`run_id` is a human-readable label provided via `--run_id` when calling `run_thesis_benchmark.sh`.
Multiple scenes are accumulated into the same file (the script calls the logger once per scene).

## Directory structure

```
output/
  matches/
    {config_key}/
      {scene}/
        {img1}__{img2}.npz   ← raw match files per pair
  benchmarks/
    {config_key}_{scene}.h5  ← packed HDF5 for RePoseD
  results/
    {run_id}/
      {scene}/
        results_{matcher}_{scene}_{timestamp}.csv
        calibrated-benchmark_{matcher}_{scene}.json
        shared_focal-benchmark_{matcher}_{scene}.json
        varying_focal-benchmark_{matcher}_{scene}.json
cache/
  features/
    {config_key}/
      {scene}/
        {stem}_dinov3_{W}x{H}_l{level}.pt    ← DINOv3 features
        {stem}_dift_{W}x{H}_t{t}_up{u}.pt    ← DIFT features
        {stem}_sp_kpts_k{N}.pt               ← SuperPoint keypoints
```

## Config key format

| Matcher    | Example config key                      |
|------------|-----------------------------------------|
| dinov3     | `dinov3_l-1_sp_mnn_mp2000`              |
| dift       | `dift_t261_up1_ens8_sp_mnn_mp2000`      |
| superpoint | `superpoint_lg_mp2048`                  |
| roma       | `roma_outdoor_mp2000`                   |
| romav2     | `romav2_precise_mp2000`                 |

Get the config key for any configuration without loading models:
```bash
python3 scripts/dinov3_matches.py --print_config_key --feat_level -1 --max_points 2000 --use_sp_keypoints --use_mutual
```

## JSON schema

```json
{
  "run_id": "phase0_bugfix",
  "timestamp": "2026-03-04T14:30:00",
  "git_commit": "abc1234...",
  "git_branch": "phase0-bugfixes",
  "method": "dinov3",
  "config_key": "dinov3_l-1_sp_mnn_mp2000",
  "config": {"feat_level": -1, "img_size": 1120, "max_points": 2000},
  "scenes": {
    "sacre_coeur": {
      "calibrated":    {"mAA10": 52.1, "rot_err": 0.95, "trans_err": 4.0, "inlier_pct": 27.1},
      "shared_focal":  {"mAA10": 48.3, "rot_err": 1.1,  "trans_err": 4.5, "inlier_pct": 27.1},
      "varying_focal": {"mAA10": 45.0, "rot_err": 1.3,  "trans_err": 5.0, "inlier_pct": 27.1}
    }
  },
  "paths": {
    "matches":         "output/matches/dinov3_l-1_sp_mnn_mp2000/",
    "benchmark_files": ["output/benchmarks/dinov3_l-1_sp_mnn_mp2000_sacre_coeur.h5"],
    "results":         "output/results/phase0_bugfix/"
  }
}
```

## Usage

The logger is called automatically at the end of `run_thesis_benchmark.sh`.
To call manually:
```bash
python3 experiments/log_experiment.py \
    --run_id    phase0_bugfix \
    --method    dinov3 \
    --config_key dinov3_l-1_sp_mnn_mp2000 \
    --scene     sacre_coeur \
    --results_dir output/results/phase0_bugfix/sacre_coeur \
    --matches_dir output/matches/dinov3_l-1_sp_mnn_mp2000/sacre_coeur \
    --benchmark   output/benchmarks/dinov3_l-1_sp_mnn_mp2000_sacre_coeur.h5 \
    --config feat_level=-1 img_size=1120 max_points=2000
```
