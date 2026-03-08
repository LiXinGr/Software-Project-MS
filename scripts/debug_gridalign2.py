"""
Debug part 2: verify the coordinate mismatch hypothesis.
For each GA match, compare:
  (a) the reported output coord (should be SP coord)
  (b) the snapped grid center (where the descriptor was actually sampled)
  (c) the offset between them
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from util import get_superpoint_keypoints

PATCH_SIZE = 16
SCENE = "sacre_coeur"
BASE = Path("/mnt/datagrid/personal/gorbuden/Software-Project-MS")
IMAGES_DIR = BASE / "datasets/phototourism" / SCENE / "images_preprocessed"
SP_NPZ_DIR  = BASE / "output/matches/dinov3_l-1_sp_mnn_mp2000" / SCENE
GA_NPZ_DIR  = BASE / "output/matches/dinov3_l-1_gridalign_mnn_mp2000" / SCENE
SP_KPT_CACHE = BASE / "cache/features/dinov3_l-1_sp_mnn_mp2000" / SCENE / "superpoint_kpts"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pick first common pair
sp_files = {p.name for p in SP_NPZ_DIR.glob("*.npz")}
ga_files = {p.name for p in GA_NPZ_DIR.glob("*.npz")}
common = sorted(sp_files & ga_files)
pair_name = common[0]
print(f"Pair: {pair_name}\n")

img1_stem, img2_stem = pair_name.replace(".npz","").split("__")
img1_path = IMAGES_DIR / f"{img1_stem}.jpg"
img2_path = IMAGES_DIR / f"{img2_stem}.jpg"
if not img1_path.exists(): img1_path = IMAGES_DIR / f"{img1_stem}.png"
if not img2_path.exists(): img2_path = IMAGES_DIR / f"{img2_stem}.png"

from PIL import Image
W1, H1 = Image.open(img1_path).size
W2, H2 = Image.open(img2_path).size
H_feat1, W_feat1 = H1 // PATCH_SIZE, W1 // PATCH_SIZE
H_feat2, W_feat2 = H2 // PATCH_SIZE, W2 // PATCH_SIZE

# Load SP keypoints for both images
kpts1 = get_superpoint_keypoints(img1_path, device='cpu', max_keypoints=2000, cache_dir=SP_KPT_CACHE)
kpts2 = get_superpoint_keypoints(img2_path, device='cpu', max_keypoints=2000, cache_dir=SP_KPT_CACHE)

def snap(kpts, H_feat, W_feat):
    x = kpts[:, 0].float()
    y = kpts[:, 1].float()
    wi = ((x - 8) / PATCH_SIZE).round().long().clamp(0, W_feat - 1)
    hi = ((y - 8) / PATCH_SIZE).round().long().clamp(0, H_feat - 1)
    px_snap = (8 + PATCH_SIZE * wi).float()
    py_snap = (8 + PATCH_SIZE * hi).float()
    return wi, hi, px_snap, py_snap

wi1, hi1, px1_snap, py1_snap = snap(kpts1, H_feat1, W_feat1)
wi2, hi2, px2_snap, py2_snap = snap(kpts2, H_feat2, W_feat2)

# Build a lookup: (wi, hi) -> (original_sp_x, original_sp_y) for first occurrence
flat1 = (hi1 * W_feat1 + wi1).numpy()
_, first1 = np.unique(flat1, return_index=True)
cell_to_sp1 = {}
for idx in first1:
    key = (wi1[idx].item(), hi1[idx].item())
    cell_to_sp1[key] = (kpts1[idx, 0].item(), kpts1[idx, 1].item())

flat2 = (hi2 * W_feat2 + wi2).numpy()
_, first2 = np.unique(flat2, return_index=True)
cell_to_sp2 = {}
for idx in first2:
    key = (wi2[idx].item(), hi2[idx].item())
    cell_to_sp2[key] = (kpts2[idx, 0].item(), kpts2[idx, 1].item())

# Load GA matches
ga_data = np.load(GA_NPZ_DIR / pair_name)
ga_mkpts0 = ga_data['mkpts0']  # reported coords for img1
ga_mkpts1 = ga_data['mkpts1']  # reported coords for img2

print("=" * 70)
print("Checking: for each GA match, is output = orig SP coord OR grid center?")
print("=" * 70)

# For each reported GA match coord, find the nearest snapped grid center
def find_snap_center(px, py, H_feat, W_feat):
    wi = round((px - 8) / PATCH_SIZE)
    hi = round((py - 8) / PATCH_SIZE)
    wi = max(0, min(W_feat-1, wi))
    hi = max(0, min(H_feat-1, hi))
    snap_x = 8 + PATCH_SIZE * wi
    snap_y = 8 + PATCH_SIZE * hi
    return snap_x, snap_y, wi, hi

print(f"\n{'idx':>3}  {'GA_x':>9} {'GA_y':>9}  {'snap_x':>7} {'snap_y':>7}  {'offset_x':>9} {'offset_y':>9}  {'is_on_grid':>10}")
print("-" * 90)
offsets = []
for i, (pt0) in enumerate(ga_mkpts0[:20]):
    px, py = pt0
    snap_x, snap_y, wi, hi = find_snap_center(px, py, H_feat1, W_feat1)
    off_x = px - snap_x
    off_y = py - snap_y
    on_grid = (abs(off_x) < 0.001 and abs(off_y) < 0.001)
    offsets.append((off_x, off_y))
    print(f"  {i:>3}  {px:>9.4f} {py:>9.4f}  {snap_x:>7} {snap_y:>7}  {off_x:>9.4f} {off_y:>9.4f}  {str(on_grid):>10}")

offsets = np.array(offsets)
print(f"\nOffset stats (GA output coord - nearest grid center):")
print(f"  Mean absolute offset x: {np.mean(np.abs(offsets[:,0])):.3f} px")
print(f"  Mean absolute offset y: {np.mean(np.abs(offsets[:,1])):.3f} px")
print(f"  Max absolute offset x: {np.max(np.abs(offsets[:,0])):.3f} px")
print(f"  Max absolute offset y: {np.max(np.abs(offsets[:,1])):.3f} px")

# Full distribution over ALL GA matches
all_offsets = []
for pt0 in ga_mkpts0:
    px, py = pt0
    snap_x, snap_y, _, _ = find_snap_center(px, py, H_feat1, W_feat1)
    all_offsets.append((px - snap_x, py - snap_y))
all_offsets = np.array(all_offsets)
print(f"\nFull distribution over {len(ga_mkpts0)} GA matches:")
print(f"  Mean |offset| x: {np.mean(np.abs(all_offsets[:,0])):.3f} px  y: {np.mean(np.abs(all_offsets[:,1])):.3f} px")
print(f"  Max  |offset| x: {np.max(np.abs(all_offsets[:,0])):.3f} px  y: {np.max(np.abs(all_offsets[:,1])):.3f} px")
print(f"  % on grid exactly: {100*np.mean((np.abs(all_offsets[:,0])<0.001)&(np.abs(all_offsets[:,1])<0.001)):.1f}%")

# Compare: SP output coords vs what we'd expect from matching
print("\n" + "=" * 70)
print("SP mode: are output coords exactly the SP keypoint positions?")
print("=" * 70)
sp_data = np.load(SP_NPZ_DIR / pair_name)
sp_mkpts0 = sp_data['mkpts0']

# Check if SP output coords are in the SP keypoint set
sp_set = set(map(tuple, kpts1.numpy().round(4)))
n_in_sp = sum(1 for pt in sp_mkpts0 if tuple(np.round(pt, 4)) in sp_set)
print(f"SP matches: {len(sp_mkpts0)} total, {n_in_sp} coords found exactly in SP kpts set")

ga_set = set(map(tuple, kpts1.numpy().round(4)))
n_ga_in_sp = sum(1 for pt in ga_mkpts0 if tuple(np.round(pt, 4)) in ga_set)
print(f"GA matches: {len(ga_mkpts0)} total, {n_ga_in_sp} coords found exactly in SP kpts set")

print("\nConclusion:")
if n_ga_in_sp < len(ga_mkpts0):
    print("  GA coords are NOT all from the SP kpts set — something is wrong!")
else:
    print("  GA coords are all original SP positions — coordinate output is correct.")
print("  BUT even if correct, the offset between SP coord and grid center")
print("  means the DESCRIPTOR was sampled at a different location than reported.")
