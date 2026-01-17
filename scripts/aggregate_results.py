#!/usr/bin/env python3
"""
Aggregate per-scene results into a combined table.

Takes multiple per-scene CSV files and averages the metrics across scenes
(as done in the original paper).

Usage:
    python3 aggregate_results.py --pattern "results_dinov3_*_TIMESTAMP.csv" --output combined.csv
    python3 aggregate_results.py --files file1.csv file2.csv file3.csv --output combined.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import glob


def aggregate_results(csv_files, output_path, matcher_name=None):
    """
    Aggregate multiple per-scene CSV files into one combined table.
    
    The paper averages metrics across scenes (simple arithmetic mean).
    """
    if not csv_files:
        print("No CSV files found!")
        return None
    
    print(f"Aggregating {len(csv_files)} scene results:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Load all CSVs
    dfs = []
    scenes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Extract scene name from filename (e.g., results_dinov3_sacre_coeur_*.csv)
        scene = Path(csv_file).stem.split('_')[2]  # Third part is scene name
        df['Scene'] = scene
        dfs.append(df)
        scenes.append(scene)
    
    # Combine all dataframes
    combined = pd.concat(dfs, ignore_index=True)
    
    # Get the matcher name from the first file if not provided
    if matcher_name is None:
        matcher_name = combined['Matches'].iloc[0] if 'Matches' in combined.columns else 'unknown'
    
    # Get columns to aggregate (numeric columns except Num_Pairs)
    # IMC-PT format: mAA@10 (AUC), mAA_f@10 (focal AUC if available)
    metric_cols = ['εr(°)', 'εt(°)', 'mAA@10', 'mAA_f@10', 'τ(ms)', 'Inliers']
    metric_cols = [c for c in metric_cols if c in combined.columns]
    
    
    # Determine grouping columns based on available columns
    group_cols = []
    
    # Include Matches if combining multiple matchers
    if 'Matches' in combined.columns:
        group_cols.append('Matches')
    
    group_cols.append('Solver')
    
    if 'Exp.Type' in combined.columns:
        group_cols.append('Exp.Type')
    if 'Opt.' in combined.columns:
        group_cols.append('Opt.')
    
    # Include hyperparameter columns in grouping (if they vary)
    hyperparam_cols = ['max_points', 'img_size', 'feat_level', 'up_ft_index', 'dift_t', 'ratio_thresh']
    hyperparam_cols_present = [c for c in hyperparam_cols if c in combined.columns]
    for hc in hyperparam_cols_present:
        if combined[hc].nunique() > 1:  # Only group by if there are different values
            group_cols.append(hc)
    
    # Group by Matches, Solver (and Exp.Type, Opt., hyperparams if exist) and average across scenes
    agg_dict = {
        **{col: 'mean' for col in metric_cols},  # Average metrics
        'Num_Pairs': 'sum',  # Sum the pairs
        'Scene': lambda x: '+'.join(sorted(set(x)))  # Combine scene names
    }
    
    # Keep hyperparameter columns that have consistent values (take first)
    for hc in hyperparam_cols_present:
        if hc not in group_cols:
            agg_dict[hc] = 'first'
    
    grouped = combined.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Only add Matches/Depth if they don't exist (single matcher mode)
    if 'Matches' not in grouped.columns:
        grouped['Matches'] = matcher_name
    if 'Depth' not in grouped.columns:
        grouped['Depth'] = 'UniDepth'
    
    # Reorder columns to match IMC-PT paper format + hyperparameters
    cols_order = ['Matches', 'Depth', 'Solver', 'Exp.Type', 'Opt.', 'εr(°)', 'εt(°)', 
                  'mAA@10', 'mAA_f@10', 'τ(ms)', 'Inliers', 'Num_Pairs', 'Scene'] + hyperparam_cols_present
    cols_order = [c for c in cols_order if c in grouped.columns]
    grouped = grouped[cols_order]
    
    # Round numeric columns
    for col in metric_cols:
        if col in grouped.columns:
            grouped[col] = grouped[col].round(2)
    
    # Save to CSV
    grouped.to_csv(output_path, index=False)
    print(f"\nCombined results saved to: {output_path}")
    print(f"Scenes combined: {', '.join(scenes)}")
    print(f"Total experiments: {len(grouped)}")
    
    # Display summary
    print("\n" + "="*80)
    print("COMBINED RESULTS (averaged across scenes)")
    print("="*80)
    print(grouped.to_string(index=False))
    
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-scene results")
    parser.add_argument("--files", nargs="+", help="List of CSV files to aggregate")
    parser.add_argument("--pattern", type=str, help="Glob pattern to find CSV files")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--matcher", type=str, help="Matcher name for the output")
    
    args = parser.parse_args()
    
    # Get list of files
    if args.files:
        csv_files = args.files
    elif args.pattern:
        csv_files = glob.glob(args.pattern)
    else:
        print("Error: Must specify --files or --pattern")
        return
    
    if not csv_files:
        print(f"No files found matching pattern: {args.pattern}")
        return
    
    aggregate_results(csv_files, args.output, args.matcher)


if __name__ == "__main__":
    main()
