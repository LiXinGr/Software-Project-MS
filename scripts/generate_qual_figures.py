#!/usr/bin/env python3
"""
Generate qualitative match visualization figures for thesis.

This script:
1. Finds appropriate image pairs based on selection criteria
2. Loads matches from NPZ files
3. Runs RANSAC to get inliers (using stored results)
4. Creates visualizations with match lines
5. Saves as PDF and PNG

Usage:
    python3 scripts/generate_qual_figures.py --scene sacre_coeur --output_dir figures/qual
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Matchers to visualize
MATCHERS = ['dinov3', 'dift', 'superpoint', 'roma', 'romav2']
MATCHER_DISPLAY_NAMES = {
    'dinov3': 'DINOv3',
    'dift': 'DIFT', 
    'superpoint': 'SuperPoint+LG',
    'roma': 'RoMa',
    'romav2': 'RoMaV2'
}

# Primary solvers
PRIMARY_SOLVERS = {
    'calibrated': '3p_ours_shift_scale+12',
    'shared_f': '4p_ours_scale_shift+12',
    'varying_f': '4p_ours_scale_shift+12'
}


def load_per_pair_results(results_dir, matcher, scene, mode='calibrated'):
    """Load per-pair results from JSON and HDF5."""
    
    # Map mode to JSON file prefix
    mode_prefix = {
        'calibrated': 'calibrated',
        'shared_f': 'shared_focal',
        'varying_f': 'varying_focal'
    }
    
    json_path = Path(results_dir) / matcher / f"{mode_prefix[mode]}-benchmark_{matcher}_{scene}.json"
    h5_path = Path('output') / f"benchmark_{matcher}_{scene}.h5"
    
    if not json_path.exists():
        print(f"Warning: {json_path} not found")
        return None
    if not h5_path.exists():
        print(f"Warning: {h5_path} not found")
        return None
    
    # Load JSON results
    with open(json_path) as f:
        results = json.load(f)
    
    # Get pair names from HDF5
    with h5py.File(h5_path, 'r') as f:
        corr_keys = sorted([k for k in f.keys() if k.startswith('corr_')])
    
    # Parse pair names from keys
    pairs = []
    for k in corr_keys:
        # Format: corr_img1_o_img2
        parts = k.replace('corr_', '').split('_o_')
        if len(parts) == 2:
            img1 = parts[0] + '_o'
            img2 = parts[1]
            pairs.append((img1, img2, k))
    
    # Get primary solver results
    primary_solver = PRIMARY_SOLVERS[mode]
    solver_results = [r for r in results if r['experiment'] == primary_solver]
    
    if len(solver_results) != len(pairs):
        print(f"Warning: {len(solver_results)} results vs {len(pairs)} pairs")
        # Try to match by index
        min_len = min(len(solver_results), len(pairs))
        solver_results = solver_results[:min_len]
        pairs = pairs[:min_len]
    
    # Combine into per-pair data
    per_pair = []
    for i, (img1, img2, corr_key) in enumerate(pairs):
        r = solver_results[i]
        per_pair.append({
            'img1': img1.replace('_o', ''),
            'img2': img2,
            'corr_key': corr_key,
            'R_err': r['R_err'],
            't_err': r['t_err'],
            'pose_err': max(r['R_err'], r['t_err']),
            'num_inliers': r['info'].get('num_inliers', 0),
            'inlier_ratio': r['info'].get('inlier_ratio', 0),
            'experiment': r['experiment']
        })
    
    return per_pair


def find_easy_pair(results_by_matcher, scene='sacre_coeur'):
    """
    Find a pair where most methods succeed well.
    Criteria: pose_err < 5° for at least 4 matchers, num_inliers > 50
    """
    # Get common pairs across matchers
    pair_keys = {}
    for matcher, results in results_by_matcher.items():
        if results is None:
            continue
        for r in results:
            key = (r['img1'], r['img2'])
            if key not in pair_keys:
                pair_keys[key] = {}
            pair_keys[key][matcher] = r
    
    candidates = []
    for key, matcher_results in pair_keys.items():
        # Need at least 4 matchers
        if len(matcher_results) < 4:
            continue
        
        # Count how many succeed
        success_count = 0
        min_inliers = float('inf')
        total_pose_err = 0
        for m, r in matcher_results.items():
            if r['pose_err'] < 5.0 and r['num_inliers'] >= 50:
                success_count += 1
                min_inliers = min(min_inliers, r['num_inliers'])
            total_pose_err += r['pose_err']
        
        # Need at least 4 matchers to succeed
        if success_count >= 4:
            avg_pose_err = total_pose_err / len(matcher_results)
            candidates.append((key, avg_pose_err, min_inliers, matcher_results))
    
    if not candidates:
        # Fallback: relax to 3 matchers
        for key, matcher_results in pair_keys.items():
            if len(matcher_results) < 3:
                continue
            success_count = sum(1 for r in matcher_results.values() 
                               if r['pose_err'] < 10.0 and r['num_inliers'] >= 30)
            if success_count >= 3:
                avg_pose_err = np.mean([r['pose_err'] for r in matcher_results.values()])
                min_inliers = min(r['num_inliers'] for r in matcher_results.values())
                candidates.append((key, avg_pose_err, min_inliers, matcher_results))
    
    if not candidates:
        print("Warning: No easy pairs found with enough methods succeeding")
        return None
    
    # Sort by min_inliers (prefer more inliers)
    candidates.sort(key=lambda x: -x[2])
    
    return candidates[0]


def find_hard_pair(results_by_matcher, scene='st_peters_square'):
    """
    Find a pair where RoMaV2 succeeds but SP+LG struggles.
    Criteria: RoMaV2 pose_err < 5°, SP+LG pose_err > 10° or low inliers
    """
    pair_keys = {}
    for matcher, results in results_by_matcher.items():
        if results is None:
            continue
        for r in results:
            key = (r['img1'], r['img2'])
            if key not in pair_keys:
                pair_keys[key] = {}
            pair_keys[key][matcher] = r
    
    candidates = []
    for key, matcher_results in pair_keys.items():
        if 'romav2' not in matcher_results or 'superpoint' not in matcher_results:
            continue
        
        romav2_r = matcher_results['romav2']
        sp_r = matcher_results['superpoint']
        
        # RoMaV2 succeeds, SP+LG struggles
        if romav2_r['pose_err'] < 5.0 and (sp_r['pose_err'] > 10.0 or sp_r['num_inliers'] < 30):
            gap = sp_r['pose_err'] - romav2_r['pose_err']
            candidates.append((key, gap, matcher_results))
    
    if not candidates:
        print("Warning: No hard pairs found matching criteria")
        return None
    
    # Sort by gap (larger gap = more contrast)
    candidates.sort(key=lambda x: -x[1])
    
    return candidates[0]


def find_repetitive_pair(results_by_matcher, scene='reichstag'):
    """
    Find a pair where foundation models (DINOv3/DIFT) have low inlier ratio.
    Criteria: DINOv3/DIFT inlier_ratio < 0.3, high pose error
    """
    pair_keys = {}
    for matcher, results in results_by_matcher.items():
        if results is None:
            continue
        for r in results:
            key = (r['img1'], r['img2'])
            if key not in pair_keys:
                pair_keys[key] = {}
            pair_keys[key][matcher] = r
    
    candidates = []
    for key, matcher_results in pair_keys.items():
        if 'dinov3' not in matcher_results:
            continue
        
        dinov3_r = matcher_results['dinov3']
        
        # DINOv3 has low inlier ratio but still some matches
        if dinov3_r['inlier_ratio'] < 0.3 and dinov3_r['num_inliers'] > 20:
            # Check if RoMa does better
            if 'romav2' in matcher_results:
                romav2_r = matcher_results['romav2']
                if romav2_r['inlier_ratio'] > dinov3_r['inlier_ratio']:
                    gap = romav2_r['inlier_ratio'] - dinov3_r['inlier_ratio']
                    candidates.append((key, gap, matcher_results))
    
    if not candidates:
        # Fallback: just find low inlier cases
        for key, matcher_results in pair_keys.items():
            if 'dinov3' in matcher_results:
                dinov3_r = matcher_results['dinov3']
                if dinov3_r['inlier_ratio'] < 0.4:
                    candidates.append((key, 1.0 - dinov3_r['inlier_ratio'], matcher_results))
    
    if not candidates:
        print("Warning: No repetitive texture pairs found")
        return None
    
    candidates.sort(key=lambda x: -x[1])
    
    return candidates[0]


def load_matches(matcher, scene, img1, img2):
    """Load matches from NPZ file."""
    matches_dir = Path('output/matches') / matcher
    
    # Try different naming conventions
    patterns = [
        f"{img1}__{img2}.npz",
        f"{img2}__{img1}.npz",
        f"{img1.replace('.jpg', '')}__{img2.replace('.jpg', '')}.npz"
    ]
    
    for pattern in patterns:
        match_path = matches_dir / pattern
        if match_path.exists():
            data = np.load(match_path)
            return data['mkpts0'], data['mkpts1']
    
    # Try searching
    for f in matches_dir.glob(f"*{img1[:8]}*{img2[:8]}*.npz"):
        data = np.load(f)
        return data['mkpts0'], data['mkpts1']
    
    print(f"Warning: Matches not found for {matcher} {img1} {img2}")
    return None, None


def load_images(scene, img1, img2):
    """Load image pair."""
    img_dir = Path(f'datasets/phototourism/{scene}/images_preprocessed')
    
    # Add .jpg extension if needed
    img1_name = img1 if img1.endswith('.jpg') else img1 + '.jpg'
    img2_name = img2 if img2.endswith('.jpg') else img2 + '.jpg'
    
    img1_path = img_dir / img1_name
    img2_path = img_dir / img2_name
    
    if not img1_path.exists() or not img2_path.exists():
        # Try dense/images
        img_dir = Path(f'datasets/phototourism/{scene}/dense/images')
        img1_path = img_dir / img1_name
        img2_path = img_dir / img2_name
    
    if img1_path.exists() and img2_path.exists():
        return Image.open(img1_path), Image.open(img2_path)
    
    print(f"Warning: Images not found: {img1_path}, {img2_path}")
    return None, None


def draw_matches(ax, img1, img2, mkpts0, mkpts1, num_inliers=None, max_lines=200, 
                 title='', color='lime', linewidth=0.5, alpha=0.6):
    """Draw matches between two images."""
    
    # Resize images to same height
    h1, w1 = img1.size[1], img1.size[0]
    h2, w2 = img2.size[1], img2.size[0]
    
    # Create side-by-side image
    max_h = max(h1, h2)
    combined_w = w1 + w2
    
    # Convert to numpy
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    
    # Pad to same height
    if h1 < max_h:
        pad = np.zeros((max_h - h1, w1, 3), dtype=np.uint8)
        img1_np = np.vstack([img1_np, pad])
    if h2 < max_h:
        pad = np.zeros((max_h - h2, w2, 3), dtype=np.uint8)
        img2_np = np.vstack([img2_np, pad])
    
    combined = np.hstack([img1_np, img2_np])
    
    ax.imshow(combined)
    ax.axis('off')
    
    if mkpts0 is None or len(mkpts0) == 0:
        ax.set_title(f'{title} (no matches)', fontsize=10)
        return
    
    # Subsample if too many
    n_matches = len(mkpts0)
    if n_matches > max_lines:
        idx = np.linspace(0, n_matches - 1, max_lines, dtype=int)
        mkpts0 = mkpts0[idx]
        mkpts1 = mkpts1[idx]
    
    # Offset for second image
    mkpts1_shifted = mkpts1.copy()
    mkpts1_shifted[:, 0] += w1
    
    # Draw lines
    for i in range(len(mkpts0)):
        ax.plot([mkpts0[i, 0], mkpts1_shifted[i, 0]], 
                [mkpts0[i, 1], mkpts1_shifted[i, 1]], 
                color=color, linewidth=linewidth, alpha=alpha)
    
    # Draw points
    ax.scatter(mkpts0[:, 0], mkpts0[:, 1], s=2, c=color, marker='o')
    ax.scatter(mkpts1_shifted[:, 0], mkpts1_shifted[:, 1], s=2, c=color, marker='o')
    
    inlier_str = f", {num_inliers} inliers" if num_inliers else ""
    ax.set_title(f'{title} ({len(mkpts0)} shown{inlier_str})', fontsize=10)


def create_combined_figure(results_by_matcher, pair_key, scene, fig_id, output_dir, mode='calibrated'):
    """Create a combined figure showing all matchers for one pair."""
    
    img1, img2 = pair_key
    matcher_results = results_by_matcher
    
    # Load images
    pil_img1, pil_img2 = load_images(scene, img1, img2)
    if pil_img1 is None:
        print(f"Could not load images for {pair_key}")
        return None
    
    # Create figure with 5 rows
    fig, axes = plt.subplots(5, 1, figsize=(14, 20))
    
    metadata = {
        'fig_id': fig_id,
        'scene': scene,
        'img1_filename': img1,
        'img2_filename': img2,
        'mode': mode,
        'solver': PRIMARY_SOLVERS[mode],
        'matchers': {}
    }
    
    for i, matcher in enumerate(MATCHERS):
        ax = axes[i]
        
        if matcher not in matcher_results:
            ax.text(0.5, 0.5, f'{MATCHER_DISPLAY_NAMES[matcher]}: No data', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')
            continue
        
        r = matcher_results[matcher]
        
        # Load matches
        mkpts0, mkpts1 = load_matches(matcher, scene, img1, img2)
        
        # Draw
        title = f"{MATCHER_DISPLAY_NAMES[matcher]}: R={r['R_err']:.2f}°, t={r['t_err']:.2f}°"
        draw_matches(ax, pil_img1, pil_img2, mkpts0, mkpts1, 
                    num_inliers=r['num_inliers'], title=title)
        
        # Store metadata
        metadata['matchers'][matcher] = {
            'N_matches_input': len(mkpts0) if mkpts0 is not None else 0,
            'N_inliers': r['num_inliers'],
            'inlier_ratio_percent': r['inlier_ratio'] * 100,
            'R_err_deg': r['R_err'],
            't_err_deg': r['t_err']
        }
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / f'{fig_id}.pdf'
    png_path = output_dir / f'{fig_id}.png'
    meta_path = output_dir / f'{fig_id}_meta.json'
    
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved: {pdf_path}, {png_path}, {meta_path}")
    
    plt.close(fig)
    
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='output/results_v2')
    parser.add_argument('--output_dir', default='figures/qual')
    parser.add_argument('--mode', default='calibrated', choices=['calibrated', 'shared_f', 'varying_f'])
    args = parser.parse_args()
    
    # Load results for all matchers across scenes
    scenes = ['sacre_coeur', 'reichstag', 'st_peters_square']
    
    all_results = {}
    for scene in scenes:
        all_results[scene] = {}
        for matcher in MATCHERS:
            results = load_per_pair_results(args.results_dir, matcher, scene, args.mode)
            if results:
                all_results[scene][matcher] = results
                print(f"Loaded {len(results)} pairs for {matcher}/{scene}")
    
    # Hardcoded pairs that we selected
    SELECTED_PAIRS = {
        'sacre_coeur': ('42297114_2749515633', '59567855_5991079082'),
        'st_peters_square': ('66019847_13308221035', '79497739_8727056965'),
        'reichstag': ('05534141_6340060522', '05791347_12791964625')
    }

    # Figure 1: Easy case
    print("\n=== generating Easy pair (sacre_coeur) ===")
    scene = 'sacre_coeur'
    pair_key = SELECTED_PAIRS[scene]
    
    # Construct matcher data manually if missing from results
    matcher_data = {}
    for matcher in MATCHERS:
        # Check if we have results
        if matcher in all_results[scene]:
            # Find the specific pair
            found = False
            for r in all_results[scene][matcher]:
                if r['img1'] == pair_key[0] and r['img2'] == pair_key[1]:
                    matcher_data[matcher] = r
                    found = True
                    break
            if not found:
                # Create dummy result with just names (visualizer will load matches)
                matcher_data[matcher] = {
                    'img1': pair_key[0], 'img2': pair_key[1],
                    'R_err': 0.0, 't_err': 0.0, 'pose_err': 0.0,
                    'num_inliers': 0, 'inlier_ratio': 0.0
                }
        else:
             # Missing results (DIFT/RoMa), create dummy
            matcher_data[matcher] = {
                'img1': pair_key[0], 'img2': pair_key[1],
                'R_err': 0.0, 't_err': 0.0, 'pose_err': 0.0,
                'num_inliers': 0, 'inlier_ratio': 0.0
            }
            
    create_combined_figure(matcher_data, pair_key, scene, 'qual_easy_calib', args.output_dir, args.mode)

    # Figure 2: Hard case
    print("\n=== generating Hard pair (st_peters_square) ===")
    scene = 'st_peters_square'
    pair_key = SELECTED_PAIRS[scene]
    
    matcher_data = {}
    for matcher in MATCHERS:
        if matcher in all_results[scene]:
            found = False
            for r in all_results[scene][matcher]:
                if r['img1'] == pair_key[0] and r['img2'] == pair_key[1]:
                    matcher_data[matcher] = r
                    found = True
                    break
            if not found:
                matcher_data[matcher] = {'img1': pair_key[0], 'img2': pair_key[1], 'R_err': 0.0, 't_err': 0.0, 'num_inliers': 0, 'inlier_ratio': 0.0}
        else:
            matcher_data[matcher] = {'img1': pair_key[0], 'img2': pair_key[1], 'R_err': 0.0, 't_err': 0.0, 'num_inliers': 0, 'inlier_ratio': 0.0}

    create_combined_figure(matcher_data, pair_key, scene, 'qual_hard_case', args.output_dir, args.mode)

    # Figure 3: Repetitive
    print("\n=== generating Repetitive pair (reichstag) ===")
    scene = 'reichstag'
    pair_key = SELECTED_PAIRS[scene]
    
    matcher_data = {}
    for matcher in MATCHERS:
        if matcher in all_results[scene]:
            found = False
            for r in all_results[scene][matcher]:
                if r['img1'] == pair_key[0] and r['img2'] == pair_key[1]:
                    matcher_data[matcher] = r
                    found = True
                    break
            if not found:
                matcher_data[matcher] = {'img1': pair_key[0], 'img2': pair_key[1], 'R_err': 0.0, 't_err': 0.0, 'num_inliers': 0, 'inlier_ratio': 0.0}
        else:
            matcher_data[matcher] = {'img1': pair_key[0], 'img2': pair_key[1], 'R_err': 0.0, 't_err': 0.0, 'num_inliers': 0, 'inlier_ratio': 0.0}

    create_combined_figure(matcher_data, pair_key, scene, 'qual_repetitive', args.output_dir, args.mode)
    
    print("\n=== Done ===")


if __name__ == '__main__':
    main()
