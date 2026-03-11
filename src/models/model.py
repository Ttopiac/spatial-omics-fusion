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
"""
import torch
import torch.nn as nn

from src.models.expression_encoder import ExpressionEncoder
from src.models.spatial_encoder import SpatialEncoder
from src.models.fusion import get_fusion


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

        # Expression encoder (used in all modes except gat_only)
        if mode != "gat_only":
            self.expr_encoder = ExpressionEncoder(
                n_genes=n_genes, embed_dim=embed_dim,
                hidden_dim=hidden_dim, n_layers=expr_layers, dropout=dropout,
            )

        # Spatial encoder (used in all modes except expr_only)
        if mode != "expr_only":
            gat_input_dim = embed_dim if mode != "gat_only" else n_genes
            self.spatial_encoder = SpatialEncoder(
                input_dim=gat_input_dim, embed_dim=embed_dim,
                n_heads=gat_heads, n_layers=gat_layers, dropout=dropout,
            )

        # Fusion (used in "full" mode)
        if mode == "full":
            self.fusion = get_fusion(
                fusion_type, embed_dim=embed_dim, n_heads=fusion_heads,
                n_layers=fusion_layers, dropout=dropout,
            )

        # Classifier
        if mode == "concat":
            self.classifier = nn.Linear(embed_dim * 2, n_classes)
        else:
            self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Args:
            x: gene expression features, shape (N, n_genes)
            edge_index: spatial graph edges, shape (2, E)
        Returns:
            logits: (N, n_classes)
            embeddings: (N, embed_dim) — the representation before classifier
        """
        if self.mode == "expr_only":
            embed = self.expr_encoder(x)
            return self.classifier(embed), embed

        if self.mode == "gat_only":
            embed = self.spatial_encoder(x, edge_index)
            return self.classifier(embed), embed

        # Modes that use both encoders: "full" and "concat"
        expr_embed = self.expr_encoder(x)
        spatial_embed = self.spatial_encoder(expr_embed, edge_index)

        if self.mode == "concat":
            embed = torch.cat([expr_embed, spatial_embed], dim=-1)
            return self.classifier(embed), embed

        # mode == "full": use fusion
        embed = self.fusion(expr_embed, spatial_embed, edge_index)
        return self.classifier(embed), embed
