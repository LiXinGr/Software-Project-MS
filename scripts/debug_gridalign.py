"""
Debug script: compare original SP mode vs gridalign mode.
Run in dinov3 conda environment.
"""
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import timm
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent))
from util import get_superpoint_keypoints

PATCH_SIZE = 16
SCENE = "sacre_coeur"
BASE = Path("/mnt/datagrid/personal/gorbuden/Software-Project-MS")
IMAGES_DIR = BASE / "datasets/phototourism" / SCENE / "images_preprocessed"
SP_NPZ_DIR  = BASE / "output/matches/dinov3_l-1_sp_mnn_mp2000" / SCENE
GA_NPZ_DIR  = BASE / "output/matches/dinov3_l-1_gridalign_mnn_mp2000" / SCENE
SP_KPT_CACHE = BASE / "cache/features/dinov3_l-1_sp_mnn_mp2000" / SCENE / "superpoint_kpts"
FEAT_CACHE   = BASE / "cache/features/dinov3_l-1_sp_mnn_mp2000" / SCENE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── pick a test pair that exists in BOTH dirs ──────────────────────────────
sp_files  = {p.name for p in SP_NPZ_DIR.glob("*.npz")}
ga_files  = {p.name for p in GA_NPZ_DIR.glob("*.npz")}
common    = sorted(sp_files & ga_files)
if not common:
    print("ERROR: no common .npz files between sp and gridalign dirs!")
    sys.exit(1)
pair_name = common[0]
print(f"Test pair: {pair_name}\n")

img1_stem, img2_stem = pair_name.replace(".npz","").split("__")
img1_path = IMAGES_DIR / f"{img1_stem}.jpg"
img2_path = IMAGES_DIR / f"{img2_stem}.jpg"
if not img1_path.exists():
    img1_path = IMAGES_DIR / f"{img1_stem}.png"
if not img2_path.exists():
    img2_path = IMAGES_DIR / f"{img2_stem}.png"

print(f"  img1: {img1_path.exists()} → {img1_path.name}")
print(f"  img2: {img2_path.exists()} → {img2_path.name}\n")

# ── 1. Load NPZ files and check coordinate precision ──────────────────────
sp_data = np.load(SP_NPZ_DIR / pair_name)
ga_data = np.load(GA_NPZ_DIR / pair_name)

sp_mkpts0 = sp_data['mkpts0']
sp_mkpts1 = sp_data['mkpts1']
ga_mkpts0 = ga_data['mkpts0']
ga_mkpts1 = ga_data['mkpts1']

print("=" * 60)
print("1. NPZ COORDINATE CHECK")
print("=" * 60)
print(f"SP mode   — {len(sp_mkpts0)} matches")
print(f"  mkpts0 dtype: {sp_mkpts0.dtype}")
print(f"  First 5 mkpts0 (x, y):")
for pt in sp_mkpts0[:5]:
    print(f"    ({pt[0]:.4f}, {pt[1]:.4f})  {'sub-pixel' if pt[0] != round(pt[0]) else 'INTEGER'}")
print(f"  Are coords sub-pixel? {np.any(sp_mkpts0 != np.round(sp_mkpts0))}")

print()
print(f"Gridalign — {len(ga_mkpts0)} matches")
print(f"  mkpts0 dtype: {ga_mkpts0.dtype}")
print(f"  First 5 mkpts0 (x, y):")
for pt in ga_mkpts0[:5]:
    print(f"    ({pt[0]:.4f}, {pt[1]:.4f})  {'sub-pixel' if pt[0] != round(pt[0]) else 'INTEGER'}")
print(f"  Are coords sub-pixel? {np.any(ga_mkpts0 != np.round(ga_mkpts0))}")

# Check if gridalign coords are on the patch-center grid (8, 24, 40, ...)
if len(ga_mkpts0) > 0:
    x_vals = ga_mkpts0[:, 0]
    is_grid_x = np.all((x_vals - 8) % PATCH_SIZE == 0)
    y_vals = ga_mkpts0[:, 1]
    is_grid_y = np.all((y_vals - 8) % PATCH_SIZE == 0)
    print(f"  Gridalign coords on patch-center grid (8+16k)? x={is_grid_x}, y={is_grid_y}")

# ── 2. Load SP keypoints and check snapping ───────────────────────────────
print()
print("=" * 60)
print("2. KEYPOINT SNAPPING ANALYSIS")
print("=" * 60)

kpts1 = get_superpoint_keypoints(img1_path, device=device, max_keypoints=2000, cache_dir=SP_KPT_CACHE)
print(f"SP keypoints for img1: {len(kpts1)}")

img = Image.open(img1_path)
W, H = img.size
H_feat, W_feat = H // PATCH_SIZE, W // PATCH_SIZE
print(f"Image size: {W}x{H}, Feature map: {W_feat}x{H_feat} = {W_feat*H_feat} cells")

# Compute snapped indices
x = kpts1[:, 0].float().cpu()
y = kpts1[:, 1].float().cpu()
wi = ((x - 8) / PATCH_SIZE).round().long().clamp(0, W_feat - 1)
hi = ((y - 8) / PATCH_SIZE).round().long().clamp(0, H_feat - 1)
flat = hi * W_feat + wi
unique_flat, first_idx = np.unique(flat.numpy(), return_index=True)
n_unique = len(unique_flat)
print(f"SP kpts → {len(kpts1)} total, {n_unique} unique grid cells after snapping")
print(f"Deduplication removes {len(kpts1) - n_unique} keypoints ({100*(len(kpts1)-n_unique)/len(kpts1):.1f}%)")

print(f"\nFirst 5 SP keypoints + their snapped grid positions:")
for i in range(min(5, len(kpts1))):
    ox, oy = kpts1[i, 0].item(), kpts1[i, 1].item()
    sx = (8 + PATCH_SIZE * wi[i].item())
    sy = (8 + PATCH_SIZE * hi[i].item())
    print(f"  SP ({ox:.2f}, {oy:.2f})  →  snapped ({sx}, {sy})  [grid cell ({wi[i].item()}, {hi[i].item()})]")

# ── 3. Load feature map and compare descriptors ───────────────────────────
print()
print("=" * 60)
print("3. DESCRIPTOR COMPARISON (first 5 keypoints)")
print("=" * 60)

num_blocks = 24
feat_level = -1
out_idx = num_blocks + feat_level  # 23

model = timm.create_model(
    "vit_large_patch16_dinov3.lvd1689m",
    pretrained=True,
    features_only=True,
    out_indices=[out_idx],
    dynamic_img_size=True,
)
model.to(device).eval()
data_config = timm.data.resolve_model_data_config(model)
transform = T.Compose([T.ToTensor(), T.Normalize(mean=data_config["mean"], std=data_config["std"])])

# Try to load from cache first
cache_path = Path(FEAT_CACHE) / f"{img1_stem}_dinov3_{W}x{H}_l{feat_level}.pt"
if cache_path.exists():
    print(f"Loading feature map from cache: {cache_path.name}")
    ft1 = torch.load(cache_path).to(device)
else:
    print("Extracting features (no cache found)...")
    xt = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model(xt)
    ft1 = feats[0].squeeze(0)
    print(f"Feature map shape: {ft1.shape}")

print(f"Feature map shape: {ft1.shape}  [C, H_feat, W_feat]")

# --- Original SP mode: bilinear interpolation ---
import torch.nn.functional as F

kpts1_feat_x = (kpts1[:5, 0].float().cpu() / W) * 2 - 1
kpts1_feat_y = (kpts1[:5, 1].float().cpu() / H) * 2 - 1
# Actually match_at_keypoints scales to feat space first then normalizes
scale_x = W_feat / W
scale_y = H_feat / H
kx = (kpts1[:5, 0].float().cpu() * scale_x).clamp(0, W_feat - 1)
ky = (kpts1[:5, 1].float().cpu() * scale_y).clamp(0, H_feat - 1)
gx = (kx / (W_feat - 1)) * 2 - 1
gy = (ky / (H_feat - 1)) * 2 - 1
grid = torch.stack([gx, gy], dim=1).view(1, 5, 1, 2).to(device)
desc_bilinear = F.grid_sample(ft1.unsqueeze(0).cpu().to(device), grid, mode='bilinear', align_corners=True)
desc_bilinear = desc_bilinear.squeeze(0).squeeze(-1).t()  # [5, C]
desc_bilinear = desc_bilinear / (desc_bilinear.norm(dim=1, keepdim=True) + 1e-8)

# --- Gridalign mode: exact grid lookup ---
wi5 = wi[:5].to(device)
hi5 = hi[:5].to(device)
desc_grid = ft1[:, hi5, wi5].t()  # [5, C]
desc_grid = desc_grid / (desc_grid.norm(dim=1, keepdim=True) + 1e-8)

print("\n  SP (bilinear) vs Gridalign (exact) descriptors, first 5 values each:")
print(f"  {'idx':>3}  {'SP x':>8} {'SP y':>8}  {'Bilinear d[:5]':<40}  {'Grid d[:5]':<40}  {'cos_sim':>8}")
for i in range(5):
    ox, oy = kpts1[i, 0].item(), kpts1[i, 1].item()
    b = desc_bilinear[i, :5].cpu().numpy()
    g = desc_grid[i, :5].cpu().numpy()
    cos = (desc_bilinear[i] @ desc_grid[i]).item()
    print(f"  {i:>3}  {ox:>8.2f} {oy:>8.2f}  {np.array2string(b, precision=4, suppress_small=True):<40}  {np.array2string(g, precision=4, suppress_small=True):<40}  {cos:>8.4f}")

# ── 4. Match count comparison over 10 common pairs ────────────────────────
print()
print("=" * 60)
print("4. MATCH COUNT COMPARISON (first 10 common pairs)")
print("=" * 60)
print(f"{'Pair':<55} {'SP':>6} {'GA':>6}")
print("-" * 70)
for pname in common[:10]:
    sp = np.load(SP_NPZ_DIR / pname)
    ga = np.load(GA_NPZ_DIR / pname)
    n_sp = len(sp['mkpts0'])
    n_ga = len(ga['mkpts0'])
    print(f"  {pname:<53} {n_sp:>6} {n_ga:>6}")

# Summary stats
sp_counts = [len(np.load(SP_NPZ_DIR / p)['mkpts0']) for p in common[:10]]
ga_counts = [len(np.load(GA_NPZ_DIR / p)['mkpts0']) for p in common[:10]]
print(f"\n  Average SP matches: {np.mean(sp_counts):.1f}")
print(f"  Average GA matches: {np.mean(ga_counts):.1f}")
print(f"  Ratio GA/SP: {np.mean(ga_counts)/np.mean(sp_counts):.3f}")

print("\nDone.")
