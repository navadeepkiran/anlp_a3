import os
import argparse
import copy
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from src.utils import set_seed, get_device, load_config
from src.part2_data import get_upfd_loaders, compute_dataset_stats
from src.part2_models.mlp import RootMLP
from src.part2_models.gcn import GCNModel
from src.part2_models.sage import GraphSAGEModel
from src.part2_models.gat import GATModel
from src.part2_models.bigcn import BiGCNModel

def instantiate_model(model_type, in_channels, hidden_dim=128, dropout=0.5):
    if model_type == "mlp":
        return RootMLP(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == "gcn":
        return GCNModel(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == "sage":
        return GraphSAGEModel(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == "gat":
        return GATModel(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == "bigcn":
        return BiGCNModel(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError("Unsupported model: " + str(model_type))

def evaluate_loader(model, loader, device, mean=None, std=None):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ptr = getattr(batch, "ptr", None)
            x = (batch.x - mean) / std if (mean is not None and std is not None) else batch.x
            logits = model(x, batch.edge_index, batch.batch, ptr=ptr)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().numpy()
            targets = batch.y.long().cpu().numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    acc = float(accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, zero_division=0))
    rec = float(recall_score(all_targets, all_preds, zero_division=0))
    f1 = float(f1_score(all_targets, all_preds, zero_division=0))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

def train_single_run(model_type, feature, seed, root="data/part2", epochs=60, lr=0.001, weight_decay=1e-4, hidden_dim=128, dropout=0.5, batch_size=128):
    set_seed(seed)
    device = get_device()

    undirected = (model_type != "bigcn")
    train_loader, val_loader, test_loader, in_channels = get_upfd_loaders(
        root=root,
        dataset_name="gossipcop",
        feature=feature,
        batch_size=batch_size,
        undirected=undirected
    )

    mean, std = compute_dataset_stats(train_loader, device)

    model = instantiate_model(model_type, in_channels, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            ptr = getattr(batch, "ptr", None)
            x = (batch.x - mean) / std
            optimizer.zero_grad()
            logits = model(x, batch.edge_index, batch.batch, ptr=ptr)
            loss = criterion(logits, batch.y.float())
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_loader(model, val_loader, device, mean=mean, std=std)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)

    test_metrics = evaluate_loader(model, test_loader, device, mean=mean, std=std)
    return test_metrics

def run_experiment(model_type, feature, cfg):
    t_cfg = cfg.get("training", {})
    m_cfg = cfg.get("model", {})
    seeds = t_cfg.get("seeds", [42, 123, 456, 789, 1024])
    epochs = t_cfg.get("epochs", 60)
    lr = t_cfg.get("learning_rate", 0.001)
    weight_decay = t_cfg.get("weight_decay", 0.0001)
    batch_size = t_cfg.get("batch_size", 128)
    hidden_dim = m_cfg.get("hidden_dim", 128)
    dropout = m_cfg.get("dropout", 0.5)

    seed_results = []
    for s in seeds:
        res = train_single_run(
            model_type=model_type,
            feature=feature,
            seed=s,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_size=batch_size
        )
        seed_results.append(res)

    mean_metrics = {
        "accuracy": float(np.mean([r["accuracy"] for r in seed_results])),
        "precision": float(np.mean([r["precision"] for r in seed_results])),
        "recall": float(np.mean([r["recall"] for r in seed_results])),
        "f1": float(np.mean([r["f1"] for r in seed_results]))
    }
    std_metrics = {
        "accuracy": float(np.std([r["accuracy"] for r in seed_results])),
        "precision": float(np.std([r["precision"] for r in seed_results])),
        "recall": float(np.std([r["recall"] for r in seed_results])),
        "f1": float(np.std([r["f1"] for r in seed_results]))
    }
    return mean_metrics, std_metrics, seed_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ta", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output:
        target_output = args.output
    elif args.ta:
        target_output = "output/part2/results_ta.json"
    else:
        target_output = "output/part2/results.json"

    models_part_a = ["mlp", "gcn", "sage", "gat", "bigcn"]
    features_part_b = ["profile", "spacy", "bert", "content"]

    results_data = {
        "part_a": {},
        "part_b": {}
    }
    detailed_data = {
        "part_a": {},
        "part_b": {}
    }

    sage_cfg_path = os.path.join("config", "part2", "sage.yaml")
    sage_cfg = load_config(sage_cfg_path) if os.path.exists(sage_cfg_path) else {}

    wandb_run = None
    try:
        import wandb
        wandb_key = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY", "")
        if wandb_key:
            wandb.login(key=wandb_key)
            wandb_run = wandb.init(
                project="Anlp_A3",
                name="part2_gnn_evaluation",
                config={"models": models_part_a, "features": features_part_b}
            )
    except Exception:
        pass

    for m in models_part_a:
        cfg_path = os.path.join("config", "part2", m + ".yaml")
        cfg = load_config(cfg_path) if os.path.exists(cfg_path) else {}
        print("Running Part A: " + str(m) + " with bert features...", flush=True)
        means, stds, seeds_data = run_experiment(model_type=m, feature="bert", cfg=cfg)
        results_data["part_a"][m] = means
        detailed_data["part_a"][m] = {
            "mean": means,
            "std": stds,
            "seeds": seeds_data
        }
        if wandb_run:
            wandb.log({
                f"part_a/{m}/accuracy": means["accuracy"],
                f"part_a/{m}/precision": means["precision"],
                f"part_a/{m}/recall": means["recall"],
                f"part_a/{m}/f1": means["f1"]
            })

    for f in features_part_b:
        if f == "bert" and "sage" in results_data["part_a"]:
            print("Reusing Part A sage results for bert feature...", flush=True)
            means = results_data["part_a"]["sage"]
            stds = detailed_data["part_a"]["sage"]["std"]
            seeds_data = detailed_data["part_a"]["sage"]["seeds"]
        else:
            print("Running Part B: sage with " + str(f) + " features...", flush=True)
            means, stds, seeds_data = run_experiment(model_type="sage", feature=f, cfg=sage_cfg)
        results_data["part_b"][f] = means
        detailed_data["part_b"][f] = {
            "mean": means,
            "std": stds,
            "seeds": seeds_data
        }
        if wandb_run:
            wandb.log({
                f"part_b/{f}/accuracy": means["accuracy"],
                f"part_b/{f}/precision": means["precision"],
                f"part_b/{f}/recall": means["recall"],
                f"part_b/{f}/f1": means["f1"]
            })

    os.makedirs("output/part2", exist_ok=True)
    with open(target_output, "w", encoding="utf-8") as out_f:
        json.dump(results_data, out_f, indent=2)
    print("Part 2 evaluation complete. Saved means to " + str(target_output), flush=True)

    other_target = "output/part2/results_ta.json" if not args.ta else "output/part2/results.json"
    with open(other_target, "w", encoding="utf-8") as out_f:
        json.dump(results_data, out_f, indent=2)

    detailed_path = os.path.join("results", "part2", "results_detailed.json")
    os.makedirs(os.path.dirname(detailed_path), exist_ok=True)
    with open(detailed_path, "w", encoding="utf-8") as out_f:
        json.dump(detailed_data, out_f, indent=2)
    print("Detailed metrics (means, stds, seeds) saved to " + str(detailed_path), flush=True)

    if wandb_run:
        wandb.save(target_output)
        wandb.save(detailed_path)
        wandb.finish()

if __name__ == "__main__":
    main()
