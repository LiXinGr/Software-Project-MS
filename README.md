# Software-Project-MS

This repository contains the implementation and experiment orchestration for a master's thesis on visual correspondence and camera-pose evaluation. The code compares training-free feature matchers, supervised projection heads, and LightGlue-based learned matching on PhotoTourism/MegaDepth-style data.

## Pipeline Overview

The final pipeline uses raw input images and preserves original image coordinates through matching and evaluation:

1. Detect SuperPoint keypoints.
2. Extract DINOv3 ViT-L/16 and DIFT Stable Diffusion descriptors.
3. Fuse and project descriptors into the selected 256-dimensional descriptor space.
4. Match descriptors with either mutual nearest-neighbor matching or LightGlue.
5. Pack matches, depth, and sparse reconstruction metadata into RePoseD benchmark files.
6. Run calibrated, shared-focal, or varying-focal pose evaluation.
7. Aggregate CSV/report artifacts and generate thesis figures.

## Repository Layout

- `scripts/`: matchers, training/evaluation runners, aggregation scripts, runtime benchmarks, and figure generation.
- `configs/`: small reproducibility configuration files.
- `docs/`: final artifact manifest and curated thesis documentation.
- `envs/`: environment files used by model-specific Python environments.
- `external/`: local checkouts or patched copies of external dependencies used by the thesis pipeline.
- `data/`, `datasets/`, `output/`, `output_v2/`, `cache/`, and `experiments/`: local data, generated artifacts, caches, and checkpoints. These are not stored in Git.

## Data And Artifacts

Large datasets, feature caches, benchmark HDF5 files, match files, checkpoints, logs, and generated outputs are intentionally excluded from version control. Reproduce or restore those artifacts from the documented dataset and checkpoint locations before running final evaluations.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the expected environments, datasets, checkpoints, final selected configuration, and main thesis commands. The selected pipeline configuration is recorded in [configs/final_selected_pipeline.json](configs/final_selected_pipeline.json).
