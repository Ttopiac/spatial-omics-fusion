"""
Full model: SpatialOmicsFusion

Combines all components:
1. Expression Encoder (MLP): raw gene expression → expression embedding
2. Spatial Encoder (GAT): expression embedding + graph → spatial embedding
3. Fusion (gated or cross-attention): expression + spatial → fused embedding
4. Classifier (linear): fused embedding → class logits

The model supports different "modes" for ablation studies:
- "full":       MLP + GAT + Fusion + Classifier  (our main model)
- "expr_only":  MLP + Classifier                  (no spatial info)
- "gat_only":   GAT + Classifier                  (spatial but no separate expression branch)
- "concat":     MLP + GAT + Concatenate + Classifier  (simple baseline)
- "scgpt":      Frozen scGPT projection + GAT + Fusion + Classifier
- "scgpt_only": Frozen scGPT projection + Classifier (no spatial info)
"""
import torch
import torch.nn as nn

from src.models.expression_encoder import ExpressionEncoder
from src.models.spatial_encoder import SpatialEncoder
from src.models.fusion import get_fusion

SCGPT_EMBED_DIM = 512


class SpatialOmicsFusion(nn.Module):
    def __init__(self, n_genes: int, n_classes: int, embed_dim: int = 128,
                 hidden_dim: int = 256, expr_layers: int = 2,
                 gat_heads: int = 4, gat_layers: int = 2,
                 fusion_type: str = "gated", fusion_heads: int = 4,
                 fusion_layers: int = 2, dropout: float = 0.1,
                 mode: str = "full"):
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim

        # scGPT modes use a linear projection instead of MLP
        if mode in ("scgpt", "scgpt_only"):
            self.scgpt_projection = nn.Sequential(
                nn.LayerNorm(SCGPT_EMBED_DIM),
                nn.Linear(SCGPT_EMBED_DIM, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Expression encoder (used in non-scgpt modes except gat_only)
        if mode not in ("gat_only", "scgpt", "scgpt_only"):
            self.expr_encoder = ExpressionEncoder(
                n_genes=n_genes, embed_dim=embed_dim,
                hidden_dim=hidden_dim, n_layers=expr_layers, dropout=dropout,
            )

        # Spatial encoder (used in all modes except expr_only and scgpt_only)
        if mode not in ("expr_only", "scgpt_only"):
            gat_input_dim = embed_dim if mode != "gat_only" else n_genes
            self.spatial_encoder = SpatialEncoder(
                input_dim=gat_input_dim, embed_dim=embed_dim,
                n_heads=gat_heads, n_layers=gat_layers, dropout=dropout,
            )

        # Fusion (used in "full" and "scgpt" modes)
        if mode in ("full", "scgpt"):
            self.fusion = get_fusion(
                fusion_type, embed_dim=embed_dim, n_heads=fusion_heads,
                n_layers=fusion_layers, dropout=dropout,
            )

        # Classifier
        if mode == "concat":
            self.classifier = nn.Linear(embed_dim * 2, n_classes)
        else:
            self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                scgpt_embeddings: torch.Tensor = None):
        """
        Args:
            x: gene expression features, shape (N, n_genes)
            edge_index: spatial graph edges, shape (2, E)
            scgpt_embeddings: optional scGPT embeddings, shape (N, 512)
        Returns:
            logits: (N, n_classes)
            embeddings: (N, embed_dim) — the representation before classifier
        """
        if self.mode == "expr_only":
            embed = self.expr_encoder(x)
            return self.classifier(embed), embed

        if self.mode == "scgpt_only":
            embed = self.scgpt_projection(scgpt_embeddings)
            return self.classifier(embed), embed

        if self.mode == "gat_only":
            embed = self.spatial_encoder(x, edge_index)
            return self.classifier(embed), embed

        # Get expression embedding (MLP or scGPT projection)
        if self.mode == "scgpt":
            expr_embed = self.scgpt_projection(scgpt_embeddings)
        else:
            expr_embed = self.expr_encoder(x)

        spatial_embed = self.spatial_encoder(expr_embed, edge_index)

        if self.mode == "concat":
            embed = torch.cat([expr_embed, spatial_embed], dim=-1)
            return self.classifier(embed), embed

        # mode == "full" or "scgpt": use fusion
        embed = self.fusion(expr_embed, spatial_embed, edge_index)
        return self.classifier(embed), embed
