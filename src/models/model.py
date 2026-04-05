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
- "geneformer":      Frozen Geneformer projection + GAT + Fusion + Classifier
- "geneformer_only": Frozen Geneformer projection + Classifier (no spatial info)
- "scgpt_brain":      Same as scgpt but with brain-specific scGPT embeddings
- "scgpt_brain_only": Same as scgpt_only but with brain-specific scGPT embeddings
- "multimodal": MLP + GAT + Image Encoder + Three-way Fusion + Classifier
- "img_expr":   MLP + Image Encoder + Cross-Attention Fusion + Classifier (no GAT)
- "gcn_only":   GCN + Classifier (GCN baseline, no attention)
- "gcn_full":   MLP + GCN + Fusion + Classifier (GCN replaces GAT)
"""
import torch
import torch.nn as nn

from src.models.expression_encoder import ExpressionEncoder
from src.models.spatial_encoder import SpatialEncoder
from src.models.fusion import get_fusion

SCGPT_EMBED_DIM = 512
GENEFORMER_EMBED_DIM = 768
RESNET_EMBED_DIM = 2048


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
        if mode in ("scgpt", "scgpt_only", "scgpt_brain", "scgpt_brain_only"):
            self.scgpt_projection = nn.Sequential(
                nn.LayerNorm(SCGPT_EMBED_DIM),
                nn.Linear(SCGPT_EMBED_DIM, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Geneformer modes use a linear projection instead of MLP
        if mode in ("geneformer", "geneformer_only"):
            self.geneformer_projection = nn.Sequential(
                nn.LayerNorm(GENEFORMER_EMBED_DIM),
                nn.Linear(GENEFORMER_EMBED_DIM, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Image encoder (used in multimodal and img_expr modes)
        if mode in ("multimodal", "img_expr"):
            from src.models.image_encoder import ImageEncoder
            self.image_encoder = ImageEncoder(
                input_dim=RESNET_EMBED_DIM, embed_dim=embed_dim, dropout=dropout,
            )

        # Expression encoder (used in most modes)
        if mode not in ("gat_only", "gcn_only", "scgpt", "scgpt_only", "scgpt_brain", "scgpt_brain_only", "geneformer", "geneformer_only"):
            self.expr_encoder = ExpressionEncoder(
                n_genes=n_genes, embed_dim=embed_dim,
                hidden_dim=hidden_dim, n_layers=expr_layers, dropout=dropout,
            )

        # Spatial encoder (GAT or GCN depending on mode)
        if mode in ("gcn_only", "gcn_full"):
            from src.models.gcn_encoder import GCNEncoder
            gcn_input_dim = n_genes if mode == "gcn_only" else embed_dim
            self.spatial_encoder = GCNEncoder(
                input_dim=gcn_input_dim, embed_dim=embed_dim,
                n_layers=gat_layers, dropout=dropout,
            )
        elif mode not in ("expr_only", "scgpt_only", "scgpt_brain_only", "geneformer_only", "img_expr"):
            gat_input_dim = embed_dim if mode != "gat_only" else n_genes
            self.spatial_encoder = SpatialEncoder(
                input_dim=gat_input_dim, embed_dim=embed_dim,
                n_heads=gat_heads, n_layers=gat_layers, dropout=dropout,
            )

        # img_expr uses gated fusion (not cross-attention) to avoid leaking spatial graph
        if mode == "img_expr":
            from src.models.fusion import GatedFusion
            self.fusion = GatedFusion(embed_dim=embed_dim, dropout=dropout)

        # Fusion for other modes
        elif mode in ("full", "scgpt", "scgpt_brain", "geneformer", "multimodal", "gcn_full"):
            self.fusion = get_fusion(
                fusion_type, embed_dim=embed_dim, n_heads=fusion_heads,
                n_layers=fusion_layers, dropout=dropout,
            )

        # Multimodal mode: additional projection to merge image with fused embed
        if mode == "multimodal":
            self.multimodal_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )
            self.multimodal_norm = nn.LayerNorm(embed_dim)

        # Classifier
        if mode == "concat":
            self.classifier = nn.Linear(embed_dim * 2, n_classes)
        else:
            self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                scgpt_embeddings: torch.Tensor = None,
                image_features: torch.Tensor = None,
                geneformer_embeddings: torch.Tensor = None):
        """
        Args:
            x: gene expression features, shape (N, n_genes)
            edge_index: spatial graph edges, shape (2, E)
            scgpt_embeddings: optional scGPT embeddings, shape (N, 512)
            image_features: optional image features, shape (N, 2048)
            geneformer_embeddings: optional Geneformer embeddings, shape (N, 768)
        Returns:
            logits: (N, n_classes)
            embeddings: (N, embed_dim) — the representation before classifier
        """
        if self.mode == "expr_only":
            embed = self.expr_encoder(x)
            return self.classifier(embed), embed

        if self.mode in ("scgpt_only", "scgpt_brain_only"):
            embed = self.scgpt_projection(scgpt_embeddings)
            return self.classifier(embed), embed

        if self.mode == "geneformer_only":
            embed = self.geneformer_projection(geneformer_embeddings)
            return self.classifier(embed), embed

        if self.mode in ("gat_only", "gcn_only"):
            embed = self.spatial_encoder(x, edge_index)
            return self.classifier(embed), embed

        if self.mode == "img_expr" and image_features is not None:
            # Image + Expression fusion (no GAT)
            # Use image embedding as the "spatial" stream in cross-attention
            expr_embed = self.expr_encoder(x)
            img_embed = self.image_encoder(image_features)
            embed = self.fusion(expr_embed, img_embed, edge_index)
            return self.classifier(embed), embed

        # Get expression embedding (MLP, scGPT projection, or Geneformer projection)
        if self.mode in ("scgpt", "scgpt_brain"):
            expr_embed = self.scgpt_projection(scgpt_embeddings)
        elif self.mode == "geneformer":
            expr_embed = self.geneformer_projection(geneformer_embeddings)
        else:
            expr_embed = self.expr_encoder(x)

        spatial_embed = self.spatial_encoder(expr_embed, edge_index)

        if self.mode == "concat":
            embed = torch.cat([expr_embed, spatial_embed], dim=-1)
            return self.classifier(embed), embed

        # Fuse expression + spatial
        fused = self.fusion(expr_embed, spatial_embed, edge_index)

        if self.mode == "multimodal" and image_features is not None:
            # Third modality: merge image embedding with fused expr+spatial
            img_embed = self.image_encoder(image_features)
            gate = self.multimodal_gate(torch.cat([fused, img_embed], dim=-1))
            embed = self.multimodal_norm(gate * fused + (1 - gate) * img_embed)
            return self.classifier(embed), embed

        # mode == "full", "scgpt", "scgpt_brain", or "geneformer"
        return self.classifier(fused), fused
