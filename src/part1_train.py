import os
import time
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from src.utils import set_seed, load_config, get_device
from src.part1_data import CipherTokenizer, PlainTokenizer, get_cipher_dataloader
from src.part1_models.transformer import TransformerCipherModel
from src.part1_models.mamba import MambaCipherModel
from src.part1_models.deltanet import DeltaNetCipherModel
from src.part1_models.hybrid import HybridCipherModel
from src.push_to_hf import push_model_to_hf

def build_model(cfg, in_vocab_size, out_vocab_size):
    m_cfg = cfg["model"]
    m_type = m_cfg.get("type", "hybrid")
    d_model = m_cfg.get("d_model", 256)
    n_layers = m_cfg.get("n_layers", 4)
    dropout = m_cfg.get("dropout", 0.1)
    d_ff = m_cfg.get("d_ff", 1024)

    if m_type == "transformer":
        return TransformerCipherModel(
            vocab_size=in_vocab_size,
            num_classes=out_vocab_size,
            d_model=d_model,
            n_heads=m_cfg.get("n_heads", 8),
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        )
    elif m_type == "mamba":
        return MambaCipherModel(
            vocab_size=in_vocab_size,
            num_classes=out_vocab_size,
            d_model=d_model,
            d_state=m_cfg.get("d_state", 16),
            d_conv=m_cfg.get("d_conv", 4),
            expand=m_cfg.get("expand", 2),
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout
        )
    elif m_type == "deltanet":
        return DeltaNetCipherModel(
            vocab_size=in_vocab_size,
            num_classes=out_vocab_size,
            d_model=d_model,
            n_heads=m_cfg.get("n_heads", 4),
            d_head=m_cfg.get("d_head", 64),
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout
        )
    elif m_type == "hybrid":
        return HybridCipherModel(
            vocab_size=in_vocab_size,
            num_classes=out_vocab_size,
            d_model=d_model,
            d_state=m_cfg.get("d_state", 16),
            d_conv=m_cfg.get("d_conv", 4),
            expand=m_cfg.get("expand", 2),
            n_heads=m_cfg.get("n_heads", 4),
            d_head=m_cfg.get("d_head", 64),
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout
        )
    else:
        raise ValueError("Unknown model type: " + str(m_type))

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    bucket_total = {"0-1k": 0, "1-2k": 0, "2k+": 0}
    bucket_correct = {"0-1k": 0, "1-2k": 0, "2k+": 0}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

            valid_mask = (labels != -100)
            preds = logits.argmax(dim=-1)
            correct = (preds == labels) & valid_mask

            num_valid = valid_mask.sum().item()
            total_loss += loss.item() * num_valid
            total_tokens += num_valid
            correct_tokens += correct.sum().item()

            b0_v = valid_mask[:, :1000].sum().item()
            b0_c = correct[:, :1000].sum().item()
            b1_v = valid_mask[:, 1000:2000].sum().item()
            b1_c = correct[:, 1000:2000].sum().item()
            b2_v = valid_mask[:, 2000:].sum().item()
            b2_c = correct[:, 2000:].sum().item()

            bucket_total["0-1k"] += b0_v
            bucket_correct["0-1k"] += b0_c
            bucket_total["1-2k"] += b1_v
            bucket_correct["1-2k"] += b1_c
            bucket_total["2k+"] += b2_v
            bucket_correct["2k+"] += b2_c

    avg_loss = total_loss / max(1, total_tokens)
    accuracy = correct_tokens / max(1, total_tokens)
    bucket_acc = {k: (bucket_correct[k] / max(1, bucket_total[k])) for k in bucket_total}
    return avg_loss, accuracy, bucket_acc

def get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(config_path):
    cfg = load_config(config_path)
    set_seed(42)
    device = get_device()
    print("Using device: " + str(device), flush=True)

    in_tok = CipherTokenizer()
    out_tok = PlainTokenizer()

    t_cfg = cfg["training"]
    d_cfg = cfg["data"]
    batch_size = t_cfg.get("batch_size", 4)
    grad_accum_steps = t_cfg.get("gradient_accumulation_steps", 4)
    epochs = t_cfg.get("epochs", 5)
    lr = t_cfg.get("learning_rate", 0.0005)
    weight_decay = t_cfg.get("weight_decay", 0.01)
    max_len = d_cfg.get("max_len", 2048)
    max_steps_per_epoch = t_cfg.get("max_steps_per_epoch", 1000)
    warmup_steps = t_cfg.get("warmup_steps", 200)
    grad_clip = t_cfg.get("gradient_clip", 1.0)

    train_loader = get_cipher_dataloader(
        cipher_path=d_cfg["train_cipher"],
        clean_path=d_cfg["train_clean"],
        in_tok=in_tok,
        out_tok=out_tok,
        batch_size=batch_size,
        shuffle=True,
        max_len=max_len,
        num_workers=t_cfg.get("num_workers", 0)
    )

    val_loader = get_cipher_dataloader(
        cipher_path=d_cfg["val_cipher"],
        clean_path=d_cfg["val_clean"],
        in_tok=in_tok,
        out_tok=out_tok,
        batch_size=1,
        shuffle=False,
        max_len=16384,
        num_workers=0
    )

    print("Train samples: " + str(len(train_loader.dataset)) + " | Val samples: " + str(len(val_loader.dataset)), flush=True)

    model = build_model(cfg, in_tok.vocab_size, out_tok.vocab_size).to(device)
    m_type = cfg["model"].get("type", "hybrid")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model initialized: " + str(m_type) + " (" + str(num_params) + " parameters)", flush=True)

    total_steps = epochs * (max_steps_per_epoch if max_steps_per_epoch else len(train_loader))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)

    save_dir = os.path.join("models", "part1")
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, m_type + "_v2_best.pt")

    wandb_run = None
    try:
        import wandb
        wandb_key = os.environ.get("WANDB_KEY") or os.environ.get("WANDB_API_KEY", "")
        if wandb_key:
            wandb.login(key=wandb_key)
            wandb_run = wandb.init(
                project=cfg.get("wandb", {}).get("project", "Anlp_A3"),
                name=cfg.get("wandb", {}).get("run_name", m_type + "_v2"),
                config=cfg
            )
    except Exception:
        pass

    best_val_acc = -1.0
    global_step = 0

    is_cuda = (device.type == "cuda" if isinstance(device, torch.device) else "cuda" in str(device))
    use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    device_type = "cuda" if is_cuda else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=(is_cuda and not use_bf16)) if is_cuda else torch.amp.GradScaler("cpu", enabled=False)

    print("Starting training on " + str(device) + " (" + str(epochs) + " epochs, " + str(min(max_steps_per_epoch, len(train_loader))) + " steps/epoch, max_len=" + str(max_len) + ")", flush=True)
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        step_in_epoch = 0
        accum_loss = 0.0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type, enabled=is_cuda, dtype=amp_dtype):
                logits = model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                loss_scaled = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            accum_loss += loss.item()

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                if scaler.is_enabled():
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                step_in_epoch += 1

                if global_step == 1 or global_step % 10 == 0:
                    lr_cur = scheduler.get_last_lr()[0]
                    step_loss = accum_loss / grad_accum_steps
                    elapsed = round(time.time() - t0)
                    print(global_step, round(step_loss, 4), elapsed, flush=True)
                    if wandb_run and (global_step == 1 or global_step % 50 == 0):
                        wandb.log({
                            "train/loss": step_loss,
                            "train/lr": lr_cur,
                            "train/epoch": epoch + 1,
                            "train/step": global_step,
                            "train/elapsed_time": elapsed
                        })
                accum_loss = 0.0

            if max_steps_per_epoch and step_in_epoch >= max_steps_per_epoch:
                break

        print("Running validation for epoch " + str(epoch + 1) + "...", flush=True)
        val_loss, val_acc, bucket_acc = evaluate_model(model, val_loader, device)
        print("Epoch " + str(epoch + 1) + "/" + str(epochs) + " - Val Loss: " + str(round(val_loss, 4)) + " - Val Acc: " + str(round(val_acc, 4)) + " - Buckets: " + str({k: round(v, 4) for k, v in bucket_acc.items()}), flush=True)

        if wandb_run:
            wandb.log({
                "val/loss": val_loss,
                "val/char_accuracy": val_acc,
                "val/bucket_0_1k": bucket_acc["0-1k"],
                "val/bucket_1_2k": bucket_acc["1-2k"],
                "val/bucket_2k_plus": bucket_acc["2k+"],
                "epoch": epoch + 1
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print("Saved new best model checkpoint to " + str(best_ckpt_path) + " (Val Acc: " + str(round(best_val_acc, 4)) + ")", flush=True)

    if wandb_run:
        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.save(best_ckpt_path)
        wandb.finish()

    hf_repo = cfg.get("huggingface", {}).get("repo_id")
    if hf_repo and os.path.exists(best_ckpt_path):
        try:
            push_model_to_hf(best_ckpt_path, hf_repo, m_type + "_v2_best.pt")
            print("Successfully uploaded model to Hugging Face: " + str(hf_repo), flush=True)
        except Exception as e:
            print("Warning: Hugging Face upload skipped: " + str(e), flush=True)

    print("Training finished for " + str(m_type) + ". Best Val Acc: " + str(round(best_val_acc, 4)), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)

if __name__ == "__main__":
    main()
