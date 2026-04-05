"""
Fine-tuning wrapper for foundation models (scGPT, Geneformer).

Instead of using fully frozen pre-extracted embeddings, this model includes
the foundation model with its last 2 transformer layers trainable (the rest
frozen). This is standard "partial fine-tuning" — it keeps most pretrained
representations intact while adapting the top layers to the downstream task.
"""
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from src.models.model import SpatialOmicsFusion

SCGPT_EMBED_DIM = 512
GENEFORMER_EMBED_DIM = 768


def _freeze_except_last_n(module, n_trainable=2):
    """Freeze all parameters, then unfreeze the last n transformer layers.

    Finds the first ModuleList of transformer layers and unfreezes the last n.
    """
    for p in module.parameters():
        p.requires_grad = False

    # Find the transformer layers (ModuleList)
    for name, child in module.named_modules():
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            for layer in child[-n_trainable:]:
                for p in layer.parameters():
                    p.requires_grad = True
            n_total = len(child)
            trainable = sum(p.requires_grad for p in module.parameters())
            total = sum(1 for _ in module.parameters())
            print(f"  Fine-tuning: {n_trainable}/{n_total} layers ({trainable}/{total} params trainable)")
            return


class ScGPTFinetune(nn.Module):
    """scGPT with last 2 transformer layers trainable + downstream model."""

    def __init__(self, model_dir: str, n_genes: int, n_classes: int,
                 embed_dim: int = 128, hidden_dim: int = 256,
                 expr_layers: int = 2, gat_heads: int = 4,
                 gat_layers: int = 2, fusion_type: str = "cross_attention",
                 fusion_heads: int = 4, fusion_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        self.scgpt_model = self._load_scgpt(model_dir)
        self.pad_value = -2

        # Freeze all but last 2 transformer layers
        _freeze_except_last_n(self.scgpt_model, n_trainable=2)

        self.downstream = SpatialOmicsFusion(
            n_genes=n_genes, n_classes=n_classes, embed_dim=embed_dim,
            hidden_dim=hidden_dim, expr_layers=expr_layers,
            gat_heads=gat_heads, gat_layers=gat_layers,
            fusion_type=fusion_type, fusion_heads=fusion_heads,
            fusion_layers=fusion_layers, dropout=dropout,
            mode="scgpt",
        )

    def _load_scgpt(self, model_dir):
        """Load the scGPT TransformerModel."""
        # torchtext shim
        class _Vocab:
            def __init__(self, vocab_obj=None):
                self._stoi = {} if vocab_obj is None else dict(getattr(vocab_obj, '_stoi', {}))
                self._itos = [] if vocab_obj is None else list(getattr(vocab_obj, '_itos', []))
                self._default_index = 0
            def __getitem__(self, token): return self._stoi.get(token, self._default_index)
            def __contains__(self, token): return token in self._stoi
            def __len__(self): return len(self._itos)
            def set_default_index(self, idx): self._default_index = idx
            def get_stoi(self): return self._stoi
            def get_itos(self): return self._itos
            def insert_token(self, token, index):
                if token not in self._stoi:
                    for t, i in self._stoi.items():
                        if i >= index: self._stoi[t] = i + 1
                    self._stoi[token] = index
                    self._itos.insert(index, token)
            def append_token(self, token):
                if token not in self._stoi:
                    self._stoi[token] = len(self._itos)
                    self._itos.append(token)
            def lookup_token(self, index): return self._itos[index]
            def lookup_indices(self, tokens): return [self[t] for t in tokens]
            @property
            def vocab(self): return self

        def _vocab_factory(ordered_dict, min_freq=1):
            v = _Vocab()
            v._stoi = {k: i for i, k in enumerate(ordered_dict.keys())}
            v._itos = list(ordered_dict.keys())
            return v

        if "torchtext" not in sys.modules:
            _torchtext = types.ModuleType("torchtext")
            _torchtext_vocab = types.ModuleType("torchtext.vocab")
            _torchtext_vocab.Vocab = _Vocab
            _torchtext_vocab.vocab = _vocab_factory
            _torchtext.vocab = _torchtext_vocab
            sys.modules["torchtext"] = _torchtext
            sys.modules["torchtext.vocab"] = _torchtext_vocab

        from scgpt.model.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab

        model_dir = Path(model_dir)
        with open(model_dir / "args.json") as f:
            config = json.load(f)

        vocab = GeneVocab.from_file(model_dir / "vocab.json")
        pad_token = config.get("pad_token", "<pad>")
        for t in [pad_token, "<cls>", "<eoc>"]:
            if t not in vocab:
                vocab.append_token(t)
        vocab.set_default_index(vocab[pad_token])
        self.pad_value = config.get("pad_value", -2)

        model = TransformerModel(
            ntoken=len(vocab),
            d_model=config["embsize"],
            nhead=config["nheads"],
            d_hid=config["d_hid"],
            nlayers=config["nlayers"],
            vocab=vocab,
            dropout=config["dropout"],
            pad_token=pad_token,
            pad_value=self.pad_value,
            do_mvc=config.get("MVC", False),
            do_dab=False,
            use_batch_labels=False,
            input_emb_style=config.get("input_emb_style", "continuous"),
            n_input_bins=config.get("n_bins", 51),
            use_fast_transformer=False,
        )

        state_dict = torch.load(model_dir / "best_model.pt", map_location="cpu")
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        model.load_state_dict(filtered, strict=False)
        return model

    def _encode_scgpt_batch(self, gene_ids, values):
        src_key_padding_mask = values.eq(self.pad_value)
        output = self.scgpt_model._encode(
            gene_ids, values, src_key_padding_mask=src_key_padding_mask,
        )
        mask = ~src_key_padding_mask
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return pooled

    def _encode_scgpt(self, gene_ids, values, chunk_size=64):
        n = gene_ids.shape[0]
        if n <= chunk_size:
            return self._encode_scgpt_batch(gene_ids, values)
        chunks = []
        for i in range(0, n, chunk_size):
            gids = gene_ids[i:i+chunk_size]
            vals = values[i:i+chunk_size]
            chunk = grad_checkpoint(
                self._encode_scgpt_batch, gids, vals, use_reentrant=False,
            )
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)

    def forward(self, x, edge_index, scgpt_tokens=None, **kwargs):
        gene_ids, values = scgpt_tokens["gene_ids"], scgpt_tokens["values"]
        with torch.autocast(device_type=x.device.type, enabled=x.is_cuda):
            embeddings = self._encode_scgpt(gene_ids, values)
        return self.downstream(x, edge_index, scgpt_embeddings=embeddings.float())

    def foundation_parameters(self):
        return [p for p in self.scgpt_model.parameters() if p.requires_grad]

    def downstream_parameters(self):
        return self.downstream.parameters()


class GeneformerFinetune(nn.Module):
    """Geneformer with last 2 transformer layers trainable + downstream model."""

    def __init__(self, n_genes: int, n_classes: int,
                 embed_dim: int = 128, hidden_dim: int = 256,
                 expr_layers: int = 2, gat_heads: int = 4,
                 gat_layers: int = 2, fusion_type: str = "cross_attention",
                 fusion_heads: int = 4, fusion_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        from transformers import BertForMaskedLM
        self.geneformer_model = BertForMaskedLM.from_pretrained(
            "ctheodoris/Geneformer",
            subfolder="Geneformer-V2-104M",
            output_hidden_states=True,
        )

        # Freeze all but last 2 encoder layers
        _freeze_except_last_n(self.geneformer_model, n_trainable=2)

        self.downstream = SpatialOmicsFusion(
            n_genes=n_genes, n_classes=n_classes, embed_dim=embed_dim,
            hidden_dim=hidden_dim, expr_layers=expr_layers,
            gat_heads=gat_heads, gat_layers=gat_layers,
            fusion_type=fusion_type, fusion_heads=fusion_heads,
            fusion_layers=fusion_layers, dropout=dropout,
            mode="geneformer",
        )

    def _encode_geneformer_batch(self, input_ids, attention_mask):
        outputs = self.geneformer_model(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        return outputs.hidden_states[-2][:, 0, :]

    def _encode_geneformer(self, input_ids, attention_mask, chunk_size=8):
        n = input_ids.shape[0]
        if n <= chunk_size:
            return self._encode_geneformer_batch(input_ids, attention_mask)
        chunks = []
        for i in range(0, n, chunk_size):
            ids = input_ids[i:i+chunk_size]
            mask = attention_mask[i:i+chunk_size]
            chunk = grad_checkpoint(
                self._encode_geneformer_batch, ids, mask, use_reentrant=False,
            )
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)

    def forward(self, x, edge_index, geneformer_tokens=None, **kwargs):
        input_ids = geneformer_tokens["input_ids"]
        attention_mask = geneformer_tokens["attention_mask"]
        with torch.autocast(device_type=x.device.type, enabled=x.is_cuda):
            embeddings = self._encode_geneformer(input_ids, attention_mask)
        return self.downstream(x, edge_index, geneformer_embeddings=embeddings.float())

    def foundation_parameters(self):
        return [p for p in self.geneformer_model.parameters() if p.requires_grad]

    def downstream_parameters(self):
        return self.downstream.parameters()
