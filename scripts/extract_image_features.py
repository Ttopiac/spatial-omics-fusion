"""
Extract histology image patch features from H&E stained Visium images.

For each spot, crops a patch around its spatial coordinates from the H&E image,
then encodes it using a pretrained ResNet50 (ImageNet weights) to produce a
2048-dim feature vector per spot.

Usage:
    python scripts/extract_image_features.py --all
    python scripts/extract_image_features.py --sample_id 151673
"""
import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scanpy as sc
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

PATCH_SIZE = 64  # pixels in hires image space


def extract_patches(adata, sample_id, patch_size=PATCH_SIZE):
    """
    Extract image patches around each spot from the H&E image.

    Returns:
        patches: (N, 3, patch_size, patch_size) tensor
    """
    # Get hires image and scale factor
    spatial_data = adata.uns["spatial"][sample_id]
    img = spatial_data["images"]["hires"]  # (H, W, 3), float32 in [0, 1]
    scale = spatial_data["scalefactors"]["tissue_hires_scalef"]

    # Spot coordinates are in full-resolution space, scale to hires
    coords = adata.obsm["spatial"] * scale  # (N, 2) in hires pixel space

    h, w = img.shape[:2]
    half = patch_size // 2
    patches = []

    for i in range(len(coords)):
        cx, cy = int(coords[i, 0]), int(coords[i, 1])

        # Crop with boundary padding
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)

        patch = img[y1:y2, x1:x2]  # (patch_h, patch_w, 3)

        # Pad if at boundary
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            padded = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded

        patches.append(patch)

    patches = np.stack(patches)  # (N, patch_size, patch_size, 3)
    # Convert to (N, 3, H, W) torch tensor
    patches = torch.tensor(patches, dtype=torch.float32).permute(0, 3, 1, 2)
    return patches


def encode_patches(patches, batch_size=64):
    """
    Encode image patches using pretrained ResNet50.

    Returns:
        features: (N, 2048) tensor
    """
    # Load pretrained ResNet50, remove classification head
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # remove classifier → output is 2048-dim
    model.eval()

    # ImageNet normalization
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = T.Resize((224, 224), antialias=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    features = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            batch = resize(batch)
            batch = normalize(batch)
            batch = batch.to(device)
            feat = model(batch).cpu()
            features.append(feat)

    return torch.cat(features, dim=0)  # (N, 2048)


def process_slice(sample_id, raw_dir, output_dir):
    """Extract and save image features for one slice."""
    raw_path = os.path.join(raw_dir, f"{sample_id}.h5ad")
    print(f"  [{sample_id}] Loading {raw_path}")
    adata = sc.read_h5ad(raw_path)

    # Drop unlabeled spots (same filtering as preprocess.py)
    adata = adata[~adata.obs["sce.layer_guess"].isna()].copy()
    print(f"  [{sample_id}] {adata.n_obs} spots with labels")

    # Extract patches
    print(f"  [{sample_id}] Extracting {PATCH_SIZE}x{PATCH_SIZE} patches...")
    patches = extract_patches(adata, sample_id)
    print(f"  [{sample_id}] Patches: {patches.shape}")

    # Encode with ResNet50
    print(f"  [{sample_id}] Encoding with ResNet50...")
    features = encode_patches(patches)
    print(f"  [{sample_id}] Features: {features.shape}")

    # Save
    out_dir = os.path.join(output_dir, sample_id)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "image_features.pt")
    torch.save(features, save_path)
    print(f"  [{sample_id}] Saved: {save_path}")

    return features.shape


def main():
    parser = argparse.ArgumentParser(description="Extract H&E image features")
    parser.add_argument("--sample_id", type=str, default="151673")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    ALL_SAMPLE_IDS = [
        "151507", "151508", "151509", "151510",
        "151669", "151670", "151671", "151672",
        "151673", "151674", "151675", "151676",
    ]

    if args.all:
        for sid in ALL_SAMPLE_IDS:
            process_slice(sid, args.raw_dir, args.output_dir)
    else:
        process_slice(args.sample_id, args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
