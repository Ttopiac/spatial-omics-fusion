"""
Image Encoder: projects pretrained ResNet50 features into the shared embedding space.

The heavy lifting (CNN feature extraction) is done offline by
scripts/extract_image_features.py, which produces a 2048-dim vector per spot.
This module simply projects that to our 128-dim embedding space.

Input:  image features (N, 2048) — precomputed ResNet50 features
Output: image embedding (N, embed_dim)
"""
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, input_dim: int = 2048, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x: image features, shape (N, 2048)
        Returns:
            image embedding, shape (N, embed_dim)
        """
        return self.projection(x)
