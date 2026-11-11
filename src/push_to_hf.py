import os
import argparse
import yaml
from src.utils import upload_hf_checkpoint

def push_model_to_hf(ckpt_path, repo_id, filename=None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Checkpoint not found: " + str(ckpt_path))
    target_name = filename if filename else os.path.basename(ckpt_path)
    url = upload_hf_checkpoint(ckpt_path, repo_id, path_in_repo=target_name)
    if not url:
        raise RuntimeError("Failed to upload " + str(ckpt_path) + " to Hugging Face repo " + str(repo_id))
    return url

def push_model(model_type, ckpt_path=None):
    cfg_file = os.path.join("config", "part1", model_type + ".yaml")
    if not os.path.exists(cfg_file):
        raise FileNotFoundError("Config not found: " + str(cfg_file))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    repo_id = cfg.get("huggingface", {}).get("repo_id")
    if not repo_id:
        raise ValueError("No repo_id found in config for " + str(model_type))
    if ckpt_path is None:
        v2_path = os.path.join("models", "part1", model_type + "_v2_best.pt")
        v1_path = os.path.join("models", "part1", model_type + "_best.pt")
        ckpt_path = v2_path if os.path.exists(v2_path) else v1_path
    return push_model_to_hf(ckpt_path, repo_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["all", "transformer", "mamba", "deltanet", "hybrid"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    models = ["transformer", "mamba", "deltanet", "hybrid"] if args.model == "all" else [args.model]
    for m in models:
        push_model(m, args.checkpoint)

if __name__ == "__main__":
    main()
