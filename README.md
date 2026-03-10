# Software-Project-MS
The content of my software project as preparation for Master's thesis

## DINOv3 Layer Study (Reproducible Runbook)

### What this is for
Use this when you want to compare DINOv3 transformer blocks (`4 8 12 16 20 23`) on a scene and generate a summary table with calibrated solver metrics for each block.

The wrapper script:
- runs `run_thesis_benchmark.sh` for each block
- logs experiment JSONs in `experiments/layer_study_b*.json`
- writes a summary table to `experiments/layer_study_summary_<scene>_<timestamp>.txt`

### One-time setup
```bash
chmod +x scripts/run_layer_study.sh
```

### Full run (recompute as needed)
```bash
./scripts/run_layer_study.sh --scene sacre_coeur --device cuda:0 --layers "4 8 12 16 20 23"
```

### Summary-only rerun (no new matching/packing/depth)
Use this if experiments were already run and you only want to regenerate the table from existing outputs/JSON logs.
```bash
./scripts/run_layer_study.sh --scene sacre_coeur --device cuda:0 --layers "4 8 12 16 20 23" --skip-matches --skip-pack --skip-depth
```

### Read the newest summary table
```bash
cat "$(ls -t experiments/layer_study_summary_sacre_coeur_*.txt | head -1)"
```

### Quick dry-run sanity check (single block)
Use this to confirm argument wiring and config key generation before long runs.
```bash
./scripts/run_layer_study.sh --scene sacre_coeur --device cuda:0 --layers "12" --dry-run
```

### Optional: all-scenes run (slow)
```bash
./scripts/run_layer_study.sh --all-scenes --device cuda:0 --layers "4 8 12 16 20 23"
```

### Notes
- Block-to-feature mapping in this project is `feat_level = block - 24`:
  - block `23 -> feat_level -1`
  - block `20 -> feat_level -4`
  - block `16 -> feat_level -8`
  - block `12 -> feat_level -12`
  - block `8  -> feat_level -16`
  - block `4  -> feat_level -20`
- Calibrated results now store aggregate metrics and per-solver metrics in experiment JSONs:
  - `scenes.<scene>.calibrated.mAA10` (aggregate)
  - `scenes.<scene>.calibrated.solvers.<solver_name>` (per solver)
  - `scenes.<scene>.calibrated.primary_solver` (selected thesis solver)
