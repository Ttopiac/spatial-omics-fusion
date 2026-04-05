"""
PyTorch Geometric dataset class for DLPFC data.

This wraps our preprocessed tensors into a PyG Data object, which is basically
a container that holds: node features, graph edges, labels, and train/val/test masks.

Think of it like a PyTorch Dataset, but for graph-structured data.
"""
import json
import os

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


def load_dlpfc_data(sample_id: str, processed_dir: str = "data/processed",
                    train_ratio: float = 0.6, val_ratio: float = 0.2,
                    seed: int = 42, load_scgpt: bool = False,
                    load_image: bool = False) -> Data:
    """
    Load preprocessed DLPFC data as a PyG Data object.

    Returns:
        Data object with:
        - x: node features (expression), shape [n_spots, n_genes]
        - edge_index: graph connectivity, shape [2, n_edges]
        - y: labels, shape [n_spots]
        - pos: spatial coordinates, shape [n_spots, 2]
        - train_mask, val_mask, test_mask: boolean masks
        - n_classes: number of classes
    """
    data_dir = os.path.join(processed_dir, sample_id)

    expression = torch.load(os.path.join(data_dir, "expression.pt"), weights_only=True)
    edge_index = torch.load(os.path.join(data_dir, "edge_index.pt"), weights_only=True)
    labels = torch.load(os.path.join(data_dir, "labels.pt"), weights_only=True)
    coordinates = torch.load(os.path.join(data_dir, "coordinates.pt"), weights_only=True)

    with open(os.path.join(data_dir, "metadata.json")) as f:
        metadata = json.load(f)

    # Create train/val/test split (stratified by label to keep class balance)
    n = expression.shape[0]
    indices = list(range(n))
    labels_np = labels.numpy()

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, random_state=seed, stratify=labels_np
    )
    # Second split: val vs test (split the remaining equally)
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_ratio_adjusted, random_state=seed,
        stratify=labels_np[temp_idx]
    )

    # Create boolean masks
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=expression,
        edge_index=edge_index,
        y=labels,
        pos=coordinates,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.n_classes = metadata["n_classes"]
    data.label_map = metadata["label_map"]
    data.sample_id = sample_id

    # Optionally load scGPT embeddings
    if load_scgpt:
        scgpt_path = os.path.join(data_dir, "scgpt_embeddings.pt")
        if os.path.exists(scgpt_path):
            data.scgpt_embeddings = torch.load(scgpt_path, weights_only=True)
        else:
            raise FileNotFoundError(
                f"scGPT embeddings not found at {scgpt_path}. "
                "Run scripts/extract_scgpt_embeddings.py first."
            )

    # Optionally load image features
    if load_image:
        img_path = os.path.join(data_dir, "image_features.pt")
        if os.path.exists(img_path):
            data.image_features = torch.load(img_path, weights_only=True)
        else:
            raise FileNotFoundError(
                f"Image features not found at {img_path}. "
                "Run scripts/extract_image_features.py first."
            )

    return data
