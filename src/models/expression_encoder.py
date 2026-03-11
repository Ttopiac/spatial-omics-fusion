"""
Expression Encoder: MLP that maps gene expression vectors to embeddings.

This is the simplest component — a standard feedforward network.
Input:  gene expression vector (n_genes=3000,)
Output: embedding vector (embed_dim=128,)

Think of it as: compressing a 3000-dim feature vector into 128 dims,
keeping the most important information for classification.
"""
import torch
import torch.nn as nn


class ExpressionEncoder(nn.Module):
    def __init__(self, n_genes: int, embed_dim: int, hidden_dim: int,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(n_genes)

        layers = []
        in_dim = n_genes
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else embed_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:  # no activation/dropout on last layer
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: gene expression, shape (N, n_genes)
        Returns:
            embedding, shape (N, embed_dim)
        """
        return self.mlp(self.norm(x))
