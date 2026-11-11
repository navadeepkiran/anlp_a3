import os
import json
import random
import yaml
import torch
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def download_hf_checkpoint(repo_id, filename, local_dir):
    from huggingface_hub import hf_hub_download
    os.makedirs(local_dir, exist_ok=True)
    token = os.environ.get("HUGGINGFACE_KEY") or os.environ.get("HF_TOKEN")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        token=token
    )

def upload_hf_checkpoint(local_path, repo_id, path_in_repo=None):
    from huggingface_hub import HfApi
    token = os.environ.get("HUGGINGFACE_KEY") or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_KEY or HF_TOKEN environment variable not set.")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    target = path_in_repo if path_in_repo else os.path.basename(local_path)
    url = api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=target,
        repo_id=repo_id,
        repo_type="model"
    )
    return url
