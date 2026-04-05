"""
GCN Spatial Encoder: Graph Convolutional Network baseline.

Unlike GAT which learns attention weights per edge, GCN treats all
neighbors equally — it averages neighbor features with fixed weights
from the normalized adjacency matrix.

Input:  node features (N, input_dim) + graph edges (2, E)
Output: spatial embeddings (N, embed_dim)
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_dim = input_dim if i == 0 else embed_dim
            self.convs.append(GCNConv(in_dim, embed_dim))
            self.norms.append(nn.LayerNorm(embed_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_layers):
            residual = x if x.shape[-1] == self.norms[i].normalized_shape[0] else None
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = torch.nn.functional.elu(x)
            x = self.dropout(x)
            if residual is not None:
                x = x + residual
        return x
