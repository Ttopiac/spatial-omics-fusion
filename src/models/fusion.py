"""
Fusion modules: combine expression embeddings and spatial embeddings.

We implement two strategies:

1. Gated Fusion (simple, fast):
   - Learn a gate that decides how much to weight expression vs spatial info
   - gate = sigmoid(W * [expr; spatial])
   - output = gate * expr + (1-gate) * spatial
   - Think of it like: "for this spot, should I trust the gene expression more,
     or the spatial neighborhood more?"

2. Cross-Attention Fusion (more powerful):
   - For each spot, attend over its spatial neighbors
   - Query = expression embedding, Key/Value = spatial embeddings of neighbors
   - This lets each spot selectively look at relevant neighbors
   - More expressive but slightly more complex
"""
import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Simple gated fusion: learns a per-spot gate between two modalities."""

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, expr_embed: torch.Tensor, spatial_embed: torch.Tensor,
                edge_index: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            expr_embed: (N, d) from expression encoder
            spatial_embed: (N, d) from spatial encoder
            edge_index: unused, accepted for API compatibility
        Returns:
            fused: (N, d)
        """
        g = self.gate(torch.cat([expr_embed, spatial_embed], dim=-1))
        fused = g * expr_embed + (1 - g) * spatial_embed
        fused = self.norm(fused + self.ffn(fused))
        return fused


class CrossAttentionFusion(nn.Module):
    """
    Neighbor-aware cross-attention fusion.

    For each spot, the query comes from its expression embedding,
    and keys/values come from spatial embeddings of its graph neighbors.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers

        self.cross_attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.norm1s = nn.ModuleList()
        self.norm2s = nn.ModuleList()

        for _ in range(n_layers):
            self.cross_attns.append(
                nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.ffns.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
            ))
            self.norm1s.append(nn.LayerNorm(embed_dim))
            self.norm2s.append(nn.LayerNorm(embed_dim))

        self.dropout = nn.Dropout(dropout)

    def _build_neighbor_index(self, edge_index, N, device):
        """Build padded neighbor indices and attention mask from edge_index."""
        # Count neighbors per node
        neighbor_counts = torch.zeros(N, dtype=torch.long, device=device)
        neighbor_counts.scatter_add_(
            0, edge_index[1],
            torch.ones(edge_index.shape[1], dtype=torch.long, device=device)
        )

        # For nodes with 0 neighbors, we'll add self-loops (attend to self)
        isolated = neighbor_counts == 0
        max_neighbors = max(neighbor_counts.max().item(), 1)

        # Vectorized neighbor list construction using sorting
        # Sort edges by target node
        sorted_idx = edge_index[1].argsort()
        sorted_src = edge_index[0][sorted_idx]
        sorted_tgt = edge_index[1][sorted_idx]

        # Build padded tensor
        neighbor_idx = torch.zeros(N, max_neighbors, dtype=torch.long, device=device)
        attn_mask = torch.ones(N, max_neighbors, dtype=torch.bool, device=device)

        # Fill using cumulative count within each target group
        offsets = torch.zeros(N, dtype=torch.long, device=device)
        offsets[1:] = neighbor_counts[:-1].cumsum(0)

        # For each target node, compute position within its neighbor list
        pos_in_group = torch.arange(len(sorted_src), device=device) - offsets[sorted_tgt]

        # Only keep positions within max_neighbors
        valid = pos_in_group < max_neighbors
        neighbor_idx[sorted_tgt[valid], pos_in_group[valid]] = sorted_src[valid]
        attn_mask[sorted_tgt[valid], pos_in_group[valid]] = False

        # Handle isolated nodes: self-attend
        if isolated.any():
            iso_nodes = isolated.nonzero(as_tuple=True)[0]
            neighbor_idx[iso_nodes, 0] = iso_nodes
            attn_mask[iso_nodes, 0] = False

        return neighbor_idx, attn_mask

    def forward(self, expr_embed: torch.Tensor, spatial_embed: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expr_embed: (N, d) expression embeddings
            spatial_embed: (N, d) spatial embeddings
            edge_index: (2, E) graph edges — defines neighborhoods
        Returns:
            fused: (N, d)
        """
        N = expr_embed.shape[0]
        device = expr_embed.device

        neighbor_idx, attn_mask = self._build_neighbor_index(edge_index, N, device)

        x = expr_embed
        for i in range(self.n_layers):
            # Gather neighbor spatial embeddings: (N, max_neighbors, d)
            kv = spatial_embed[neighbor_idx]

            # Cross-attention: Q=expression, K=V=spatial neighbors
            q = x.unsqueeze(1)  # (N, 1, d)
            attn_out, _ = self.cross_attns[i](q, kv, kv, key_padding_mask=attn_mask)
            attn_out = attn_out.squeeze(1)  # (N, d)

            # Residual + norm
            x = self.norm1s[i](x + self.dropout(attn_out))

            # FFN + residual + norm
            x = self.norm2s[i](x + self.dropout(self.ffns[i](x)))

        return x


def get_fusion(fusion_type: str, **kwargs):
    """Factory function to create fusion module."""
    if fusion_type == "gated":
        return GatedFusion(embed_dim=kwargs["embed_dim"], dropout=kwargs.get("dropout", 0.1))
    elif fusion_type == "cross_attention":
        return CrossAttentionFusion(
            embed_dim=kwargs["embed_dim"],
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("n_layers", 2),
            dropout=kwargs.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
