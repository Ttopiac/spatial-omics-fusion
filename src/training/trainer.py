"""
Training loop for SpatialOmicsFusion.

This is a standard PyTorch training loop adapted for graph data:
- Full-batch training (the entire graph fits in memory)
- CrossEntropy loss with class weights (because some brain layers have fewer spots)
- Early stopping based on validation ARI
"""
import time

import torch
import torch.nn as nn
import numpy as np

from src.utils.metrics import compute_metrics, compute_extended_metrics


class Trainer:
    def __init__(self, model, data, device, lr=1e-3, weight_decay=1e-4,
                 epochs=200, patience=20, foundation_lr=None):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.is_finetune = hasattr(model, "foundation_parameters")

        # Move data to device, handling nested dicts (tokenized data)
        self.data = data.to(device)
        if hasattr(data, "scgpt_tokens"):
            self.data.scgpt_tokens = {k: v.to(device) for k, v in data.scgpt_tokens.items()}
        if hasattr(data, "geneformer_tokens"):
            self.data.geneformer_tokens = {k: v.to(device) for k, v in data.geneformer_tokens.items()}

        # Compute class weights (inverse frequency) for imbalanced classes
        labels = data.y[data.train_mask].cpu()
        class_counts = torch.bincount(labels, minlength=data.n_classes).float()
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * data.n_classes
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        # Use parameter groups for fine-tuning (lower LR for foundation model)
        if self.is_finetune and foundation_lr is not None:
            self.optimizer = torch.optim.AdamW([
                {"params": model.foundation_parameters(), "lr": foundation_lr},
                {"params": model.downstream_parameters(), "lr": lr},
            ], weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10
        )

        # Tracking
        self.best_val_ari = -1
        self.best_state = None
        self.patience_counter = 0
        self.history = []

    def _forward(self):
        """Forward pass with optional scGPT/Geneformer/image embeddings or tokens."""
        scgpt_embed = getattr(self.data, "scgpt_embeddings", None)
        image_feat = getattr(self.data, "image_features", None)
        geneformer_embed = getattr(self.data, "geneformer_embeddings", None)
        kwargs = dict(
            scgpt_embeddings=scgpt_embed,
            image_features=image_feat,
            geneformer_embeddings=geneformer_embed,
        )
        # Fine-tune models accept token inputs
        scgpt_tokens = getattr(self.data, "scgpt_tokens", None)
        geneformer_tokens = getattr(self.data, "geneformer_tokens", None)
        if scgpt_tokens is not None:
            kwargs["scgpt_tokens"] = scgpt_tokens
        if geneformer_tokens is not None:
            kwargs["geneformer_tokens"] = geneformer_tokens
        return self.model(self.data.x, self.data.edge_index, **kwargs)

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        logits, _ = self._forward()
        loss = self.criterion(
            logits[self.data.train_mask], self.data.y[self.data.train_mask]
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, mask):
        self.model.eval()
        logits, embeddings = self._forward()
        preds = logits[mask].argmax(dim=-1).cpu().numpy()
        true = self.data.y[mask].cpu().numpy()
        loss = self.criterion(logits[mask], self.data.y[mask]).item()

        metrics = compute_metrics(true, preds)
        metrics["loss"] = loss
        return metrics

    def fit(self):
        """Run full training loop with early stopping."""
        start_time = time.time()
        print(f"Training on {self.device} | {self.epochs} max epochs | patience={self.patience}")
        print("-" * 75)

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.data.val_mask)

            self.scheduler.step(val_metrics["ari"])

            # Track history
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            # Early stopping
            if val_metrics["ari"] > self.best_val_ari:
                self.best_val_ari = val_metrics["ari"]
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if epoch % 20 == 0 or self.patience_counter == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                      f"val_loss={val_metrics['loss']:.4f} | "
                      f"val_ari={val_metrics['ari']:.4f} | "
                      f"val_acc={val_metrics['accuracy']:.4f} | "
                      f"lr={lr:.1e}"
                      + (" *" if self.patience_counter == 0 else ""))

            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.model.to(self.device)

        elapsed = time.time() - start_time
        print("-" * 75)
        print(f"Done in {elapsed:.1f}s | Best val ARI: {self.best_val_ari:.4f}")
        return self.history

    def test(self, extended=False):
        """Evaluate on test set using best model."""
        if not extended:
            test_metrics = self.evaluate(self.data.test_mask)
            print(f"Test | ARI={test_metrics['ari']:.4f} | "
                  f"NMI={test_metrics['nmi']:.4f} | "
                  f"Acc={test_metrics['accuracy']:.4f}")
            return test_metrics

        # Extended evaluation with probabilities and boundary analysis
        self.model.eval()
        with torch.no_grad():
            logits, _ = self._forward()
            mask = self.data.test_mask
            probs = torch.softmax(logits[mask], dim=-1).cpu().numpy()
            preds = logits[mask].argmax(dim=-1).cpu().numpy()
            true = self.data.y[mask].cpu().numpy()
            edge_index = self.data.edge_index.cpu().numpy()

            # Remap edge_index to test-only indices
            test_indices = mask.nonzero(as_tuple=True)[0].cpu().numpy()
            idx_map = {orig: new for new, orig in enumerate(test_indices)}
            test_edges_src, test_edges_tgt = [], []
            for i in range(edge_index.shape[1]):
                s, t = edge_index[0, i], edge_index[1, i]
                if s in idx_map and t in idx_map:
                    test_edges_src.append(idx_map[s])
                    test_edges_tgt.append(idx_map[t])
            test_edge_index = np.array([test_edges_src, test_edges_tgt])

        test_metrics = compute_extended_metrics(true, preds, probs, test_edge_index)
        print(f"Test | ARI={test_metrics['ari']:.4f} | "
              f"NMI={test_metrics['nmi']:.4f} | "
              f"Acc={test_metrics['accuracy']:.4f} | "
              f"Top2={test_metrics['top2_accuracy']:.4f} | "
              f"Interior={test_metrics['interior_accuracy']:.4f} | "
              f"Boundary={test_metrics['boundary_accuracy']:.4f}")
        return test_metrics
