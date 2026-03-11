"""
Spatial Encoder: Graph Attention Network (GAT) that learns from spatial neighborhoods.

Why GAT?
- Each spot on the tissue has spatial neighbors (nearby spots).
- A GAT aggregates information from neighbors using attention — it learns which
  neighbors are most relevant.
- This is how the model learns spatial context: "what are my neighboring spots like?"

Input:  node features (N, input_dim) + graph edges (2, E)
Output: spatial embeddings (N, embed_dim)

The graph structure comes from KNN on (x,y) coordinates — spots that are
physically close on the tissue are connected.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class SpatialEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_dim = input_dim if i == 0 else embed_dim
            # GAT outputs n_heads * (embed_dim // n_heads) = embed_dim
            self.convs.append(GATConv(
                in_channels=in_dim,
                out_channels=embed_dim // n_heads,
                heads=n_heads,
                dropout=dropout,
                concat=True,  # concatenate head outputs → embed_dim
            ))
            self.norms.append(nn.LayerNorm(embed_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: node features, shape (N, input_dim)
            edge_index: graph edges, shape (2, E)
        Returns:
            spatial embeddings, shape (N, embed_dim)
        """
        for i in range(self.n_layers):
            residual = x if x.shape[-1] == self.norms[i].normalized_shape[0] else None
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = torch.nn.functional.elu(x)
            x = self.dropout(x)
            if residual is not None:
                x = x + residual  # residual connection
        return x
