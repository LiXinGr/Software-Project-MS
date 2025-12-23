import torch
from PIL import Image
import torchvision.transforms as T
import timm
from util import compute_matches, visualize_matches
import argparse
from pathlib import Path
import numpy as np

def run_dinov3_extractor(
    img_path,
    ft_path,
    model,
    transform,
    img_size,
    feat_level,
):
    """
    Extract a single DINOv3 feature map [C, H, W] for one image.
    """

    if ft_path.exists():
        print(f"[DINOv3] Using cached features: {ft_path}")
        return torch.load(ft_path)

    img = Image.open(img_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)

    x = transform(img).unsqueeze(0).to(next(model.parameters()).device)

    model.eval()
    with torch.no_grad():
        feats = model(x)

    # if batch size were >1: squeeze(0) would be dangerous
    ft = feats[feat_level].squeeze(0).cpu()  # [C, H, W]

    torch.save(ft, ft_path)
    return ft


def main():
    parser = argparse.ArgumentParser(
        description="Two-image correspondence using DINOv3 features"
    )

    parser.add_argument("--img1", type=str, required=True)
    parser.add_argument("--img2", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--feat_level", type=int, default=-1)
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--use_mutual", action="store_true")
    parser.add_argument("--ratio_thresh", type=float, default=None)
    parser.add_argument("--max_lines", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    img1_path = Path(args.img1)
    img2_path = Path(args.img2)

    ft1_path = img1_path.with_suffix(".dinov3.pt")
    ft2_path = img2_path.with_suffix(".dinov3.pt")

    transform = T.Compose([T.ToTensor()])

    model = timm.create_model(
        "vit_large_patch16_dinov3.lvd1689m",
        pretrained=True,
        features_only=True,
    )
    model.to(device)

    # ------------------------------
    # OPTIONAL: extract features from an exact transformer block
    # (useful if you want block-level control instead of timm feature stages)
    #
    # saved_block_output = None
    #
    # def hook_fn(module, input, output):
    #     """
    #     This function is called automatically during the forward pass
    #     of the selected transformer block.
    #
    #     module: the block itself
    #     input:  input to the block (tuple)
    #     output: output of the block (tensor)
    #     """
    #     nonlocal saved_block_output
    #     saved_block_output = output
    #
    # # Example: hook into block 10 (0-based index)
    # # model.model.blocks[10].register_forward_hook(hook_fn)
    #
    # # After running: model(x)
    # # saved_block_output will contain features from block 10
    # ------------------------------

    ft1 = run_dinov3_extractor(
        img1_path,
        ft1_path,
        model,
        transform,
        img_size=args.img_size,
        feat_level=args.feat_level,
    )

    ft2 = run_dinov3_extractor(
        img2_path,
        ft2_path,
        model,
        transform,
        img_size=args.img_size,
        feat_level=args.feat_level,
    )

    x1, y1, x2, y2, feat_hw1, feat_hw2 = compute_matches(
        ft1,
        ft2,
        max_points=args.max_points,
        use_mutual=args.use_mutual,
        ratio_thresh=args.ratio_thresh,
    )

    img1_np = np.array(Image.open(img1_path).convert("RGB"))
    img2_np = np.array(Image.open(img2_path).convert("RGB"))

    vis_path = "datasets/" + img1_path.stem + "_dinov3_matches.png"

    visualize_matches(
        img1_np,
        img2_np,
        x1, y1, x2, y2,
        feat_hw1, feat_hw2,
        out_path=vis_path,
        max_lines=args.max_lines,
    )

if __name__ == "__main__":
    main()


# python3 scripts/3.py \
#   --img1 datasets/bran1.jpg \
#   --img2 datasets/bran2.jpg \
#   --img_size 512 \
#   --feat_level -1 \
#   --use_mutual \
#   --ratio_thresh 1.1