import os
import torch
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

def get_upfd_dataset(root="data/part2", dataset_name="gossipcop", feature="bert", split="train", undirected=True):
    os.makedirs(root, exist_ok=True)
    transform = T.ToUndirected() if undirected else None
    return UPFD(root=root, name=dataset_name, feature=feature, split=split, transform=transform)

def get_upfd_loaders(root="data/part2", dataset_name="gossipcop", feature="bert", batch_size=128, undirected=True):
    train_dataset = get_upfd_dataset(root=root, dataset_name=dataset_name, feature=feature, split="train", undirected=undirected)
    val_dataset = get_upfd_dataset(root=root, dataset_name=dataset_name, feature=feature, split="val", undirected=undirected)
    test_dataset = get_upfd_dataset(root=root, dataset_name=dataset_name, feature=feature, split="test", undirected=undirected)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.num_features

def compute_dataset_stats(train_loader, device):
    all_x = []
    for batch in train_loader:
        all_x.append(batch.x)
    cat_x = torch.cat(all_x, dim=0).to(device)
    mean = cat_x.mean(dim=0, keepdim=True)
    std = cat_x.std(dim=0, keepdim=True).clamp(min=1e-5)
    return mean, std
