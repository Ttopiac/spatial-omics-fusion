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

from src.utils.metrics import compute_metrics


class Trainer:
    def __init__(self, model, data, device, lr=1e-3, weight_decay=1e-4,
                 epochs=200, patience=20):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.epochs = epochs
        self.patience = patience

        # Compute class weights (inverse frequency) for imbalanced classes
        labels = data.y[data.train_mask].cpu()
        class_counts = torch.bincount(labels, minlength=data.n_classes).float()
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * data.n_classes
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

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

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        logits, _ = self.model(self.data.x, self.data.edge_index)
        loss = self.criterion(
            logits[self.data.train_mask], self.data.y[self.data.train_mask]
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, mask):
        self.model.eval()
        logits, embeddings = self.model(self.data.x, self.data.edge_index)
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

    def test(self):
        """Evaluate on test set using best model."""
        test_metrics = self.evaluate(self.data.test_mask)
        print(f"Test | ARI={test_metrics['ari']:.4f} | "
              f"NMI={test_metrics['nmi']:.4f} | "
              f"Acc={test_metrics['accuracy']:.4f}")
        return test_metrics
