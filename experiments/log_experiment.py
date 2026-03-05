#!/usr/bin/env python3
"""
Experiment Logger

Reads evaluation results from a results directory and saves a structured
JSON log for the experiment run.

Usage:
    python3 experiments/log_experiment.py \
        --run_id        phase0_bugfix \
        --method        dinov3 \
        --config_key    dinov3_l-1_sp_mnn_mp2000 \
        --scene         sacre_coeur \
        --results_dir   output/results/phase0_bugfix/sacre_coeur \
        --matches_dir   output/matches/dinov3_l-1_sp_mnn_mp2000/sacre_coeur \
        --benchmark     output/benchmarks/dinov3_l-1_sp_mnn_mp2000_sacre_coeur.h5 \
        --config feat_level=-1 img_size=1120 max_points=2000 seed=42

Output: experiments/{run_id}.json  (created or updated in-place)

JSON schema:
{
  "run_id": "phase0_bugfix",
  "timestamp": "2026-03-04T14:30:00",
  "git_commit": "abc123...",
  "git_branch": "phase0-bugfixes",
  "method": "dinov3",
  "config_key": "dinov3_l-1_sp_mnn_mp2000",
  "config": {"feat_level": -1, "img_size": 1120, ...},
  "scenes": {
    "sacre_coeur": {
      "calibrated":    {"mAA10": 52.1, "rot_err": 0.95, "trans_err": 4.0, "inlier_pct": 27.1},
      "shared_focal":  {...},
      "varying_focal": {...}
    }
  },
  "paths": {
    "matches":         "output/matches/dinov3_l-1_sp_mnn_mp2000/",
    "benchmark_files": ["output/benchmarks/dinov3_l-1_sp_mnn_mp2000_sacre_coeur.h5"],
    "results":         "output/results/phase0_bugfix/"
  }
}
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent


def get_git_info():
    """Get current git commit hash and branch name."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        branch = "unknown"
    return commit, branch


def parse_config(config_list):
    """Parse key=value config pairs into a typed dict."""
    config = {}
    for item in config_list:
        if "=" not in item:
            config[item] = True
            continue
        k, v = item.split("=", 1)
        if v.lower() == "true":
            config[k] = True
        elif v.lower() == "false":
            config[k] = False
        elif v.lower() in ("none", "null", ""):
            config[k] = None
        else:
            try:
                config[k] = int(v)
            except ValueError:
                try:
                    config[k] = float(v)
                except ValueError:
                    config[k] = v
    return config


def parse_eval_json(json_path):
    """
    Parse a single eval output JSON (calibrated / shared_focal / varying_focal)
    and return aggregate metrics across all solvers and pairs.

    Eval JSON format: list of dicts with keys:
        R_err (°), t_err (°), experiment (solver name),
        info: {runtime (ms), inlier_ratio}

    Returns: {mAA10, rot_err, trans_err, inlier_pct}  or None if empty.
    """
    with open(json_path) as f:
        results = json.load(f)
    if not results:
        return None

    r_err    = np.array([r.get("R_err", float("nan")) for r in results], dtype=float)
    t_err    = np.array([r.get("t_err", float("nan")) for r in results], dtype=float)
    runtimes = np.array([r.get("info", {}).get("runtime",      float("nan")) for r in results], dtype=float)
    inliers  = np.array([r.get("info", {}).get("inlier_ratio", float("nan")) for r in results], dtype=float)

    pose_err = np.maximum(r_err, t_err)
    pose_err[np.isnan(pose_err)] = 180.0
    mAA_10 = float(np.mean([np.sum(pose_err < t) / len(pose_err) for t in range(1, 11)]) * 100)

    return {
        "mAA10":      round(mAA_10, 2),
        "rot_err":    round(float(np.nanmedian(r_err)), 2),
        "trans_err":  round(float(np.nanmedian(t_err)), 2),
        "inlier_pct": round(float(np.nanmean(inliers) * 100), 2),
    }


def load_or_create_record(output_path, run_id, method, config_key,
                           config, git_commit, git_branch, timestamp):
    """Load existing JSON (for multi-scene accumulation) or create fresh."""
    if output_path.exists():
        with open(output_path) as f:
            record = json.load(f)
        # Update top-level metadata in case this is a new scene being added
        record["timestamp"]  = timestamp
        record["git_commit"] = git_commit
        record["git_branch"] = git_branch
    else:
        record = {
            "run_id":     run_id,
            "timestamp":  timestamp,
            "git_commit": git_commit,
            "git_branch": git_branch,
            "method":     method,
            "config_key": config_key,
            "config":     config,
            "scenes":     {},
            "paths": {
                "matches":         None,
                "benchmark_files": [],
                "results":         None,
            },
        }
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Log experiment metadata and results to a JSON file."
    )
    parser.add_argument("--run_id",      required=True,
                        help="Human-readable experiment label (used as filename)")
    parser.add_argument("--method",      required=True,
                        help="Matcher name (e.g. dinov3, dift, superpoint)")
    parser.add_argument("--config_key",  required=True,
                        help="Config key string (e.g. dinov3_l-1_sp_mnn_mp2000)")
    parser.add_argument("--scene",       default=None,
                        help="Scene name (auto-detected from CSV filename if omitted)")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing results CSV files")
    parser.add_argument("--matches_dir", default=None,
                        help="Match output directory for this config+scene")
    parser.add_argument("--benchmark",   default=None,
                        help=".h5 benchmark file path")
    parser.add_argument("--config",      nargs="*", default=[],
                        help="key=value hyperparameter pairs")
    parser.add_argument("--output",      default=None,
                        help="Output JSON path (default: experiments/{run_id}.json)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Warning: results_dir does not exist: {results_dir}", file=sys.stderr)

    git_commit, git_branch = get_git_info()
    config    = parse_config(args.config)
    timestamp = datetime.now().isoformat(timespec="seconds")

    # Determine output path
    output_path = Path(args.output) if args.output else (EXPERIMENTS_DIR / f"{args.run_id}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load or create the experiment record (supports multi-scene accumulation)
    record = load_or_create_record(
        output_path, args.run_id, args.method, args.config_key,
        config, git_commit, git_branch, timestamp
    )

    # Parse eval JSONs written by eval.py / eval_shared_f.py / eval_varying_f.py.
    # Filename pattern: {exp_type}-{config_key}_{scene}.json
    KNOWN_SCENES = ["sacre_coeur", "reichstag", "st_peters_square"]

    # Determine which scenes to log
    if args.scene:
        scenes_to_log = [args.scene]
    else:
        # Auto-detect from files present in results_dir
        detected = set()
        for json_path in results_dir.glob("calibrated-*.json"):
            for s in KNOWN_SCENES:
                if s in json_path.stem:
                    detected.add(s)
        scenes_to_log = sorted(detected) or ["unknown"]

    for scene in scenes_to_log:
        basename = f"{args.config_key}_{scene}"
        eval_files = {
            "calibrated":    results_dir / f"calibrated-{basename}.json",
            "shared_focal":  results_dir / f"shared_focal-{basename}.json",
            "varying_focal": results_dir / f"varying_focal-{basename}.json",
        }

        scene_results = {}
        for exp_type, json_path in eval_files.items():
            if json_path.exists():
                try:
                    metrics = parse_eval_json(json_path)
                    if metrics:
                        scene_results[exp_type] = metrics
                except Exception as e:
                    print(f"Warning: failed to parse {json_path}: {e}", file=sys.stderr)
            else:
                print(f"  No {exp_type} results: {json_path.name}", file=sys.stderr)

        if scene_results:
            record["scenes"][scene] = scene_results
        else:
            print(f"Warning: no eval results found for scene={scene} config={args.config_key}", file=sys.stderr)

    # Update paths
    if args.matches_dir:
        # Store the config-level matches dir (strip scene suffix for readability)
        matches_root = str(Path(args.matches_dir).parent)
        record["paths"]["matches"] = matches_root

    results_root = str(Path(args.results_dir).parent)
    record["paths"]["results"] = results_root

    if args.benchmark:
        bm_path = Path(args.benchmark)
        try:
            bm = str(bm_path.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            bm = str(bm_path)
        if bm not in record["paths"]["benchmark_files"]:
            record["paths"]["benchmark_files"].append(bm)

    # Write
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"Experiment logged → {output_path}")
    print(f"  run_id:     {args.run_id}")
    print(f"  config_key: {args.config_key}")
    print(f"  git:        {git_branch} @ {git_commit[:8]}")
    print(f"  scenes:     {list(record['scenes'].keys())}")


if __name__ == "__main__":
    main()
