import os
import argparse
import json
import torch

from src.utils import load_config, get_device, download_hf_checkpoint
from src.part1_data import CipherTokenizer, PlainTokenizer, strip_noise
from src.part1_models.transformer import TransformerCipherModel
from src.part1_models.mamba import MambaCipherModel
from src.part1_models.deltanet import DeltaNetCipherModel
from src.part1_models.hybrid import HybridCipherModel

def load_inference_model(config_path, checkpoint_path=None, device="cpu", from_hf=True):
    cfg = load_config(config_path)
    in_tok = CipherTokenizer()
    out_tok = PlainTokenizer()

    m_cfg = cfg["model"]
    m_type = m_cfg.get("type", "hybrid")
    d_model = m_cfg.get("d_model", 256)
    n_layers = m_cfg.get("n_layers", 4)
    dropout = m_cfg.get("dropout", 0.0)
    d_ff = m_cfg.get("d_ff", 1024)

    if m_type == "transformer":
        model = TransformerCipherModel(
            vocab_size=in_tok.vocab_size,
            num_classes=out_tok.vocab_size,
            d_model=d_model,
            n_heads=m_cfg.get("n_heads", 8),
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        )
    elif m_type == "mamba":
        model = MambaCipherModel(
            vocab_size=in_tok.vocab_size,
            num_classes=out_tok.vocab_size,
            d_model=d_model,
            d_state=m_cfg.get("d_state", 16),
            d_conv=m_cfg.get("d_conv", 4),
            expand=m_cfg.get("expand", 2),
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout
        )
    elif m_type == "deltanet":
        model = DeltaNetCipherModel(
            vocab_size=in_tok.vocab_size,
            num_classes=out_tok.vocab_size,
            d_model=d_model,
            n_heads=m_cfg.get("n_heads", 4),
            d_head=m_cfg.get("d_head", 64),
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout
        )
    elif m_type == "hybrid":
        model = HybridCipherModel(
            vocab_size=in_tok.vocab_size,
            num_classes=out_tok.vocab_size,
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
        raise ValueError("Unknown model: " + str(m_type))

    ckpt_file = checkpoint_path
    repo_id = cfg.get("huggingface", {}).get("repo_id")
    default_local = os.path.join("models", "part1", m_type + "_v3_best.pt")

    if from_hf and repo_id:
        try:
            pulled = download_hf_checkpoint(
                repo_id=repo_id,
                filename=m_type + "_v3_best.pt",
                local_dir=os.path.join("models", "part1")
            )
            if pulled and os.path.exists(pulled):
                ckpt_file = pulled
        except Exception as e:
            print("Warning: Hugging Face download failed (" + str(e) + "). Attempting local fallback...", flush=True)

    if ckpt_file is None or not os.path.exists(ckpt_file):
        if os.path.exists(default_local):
            ckpt_file = default_local
        else:
            raise FileNotFoundError("Checkpoint not found locally or on Hugging Face for " + str(m_type) + ". Checked: " + str(ckpt_file) + " and " + str(default_local))

    state_dict = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, in_tok, out_tok

def decrypt_sample(model, in_tok, out_tok, ciphertext, device="cpu"):
    c_strip = strip_noise(ciphertext)
    if not c_strip:
        return ""
    input_ids = in_tok.encode(c_strip)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    use_cuda = device.type == "cuda" if isinstance(device, torch.device) else "cuda" in str(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_cuda, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits = model(input_tensor)
            pred_ids = logits[0].argmax(dim=-1).tolist()

    return out_tok.decode(pred_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--config", type=str, default="config/part1/hybrid.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_hf", action="store_true")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--roll_number", type=str, default="2024101056")
    args = parser.parse_args()

    device = get_device()
    from_hf = not args.no_hf
    model, in_tok, out_tok = load_inference_model(args.config, args.checkpoint, device=device, from_hf=from_hf)

    if args.heldout:
        if args.input_file:
            input_path = args.input_file
        else:
            heldout_dir = "data/part1/heldout"
            input_path = os.path.join(heldout_dir, "cipher.jsonl")
            if not os.path.exists(input_path) and os.path.exists(heldout_dir):
                candidates = [f for f in os.listdir(heldout_dir) if f.endswith(".jsonl") and "cipher" in f]
                if candidates:
                    input_path = os.path.join(heldout_dir, candidates[0])
                else:
                    candidates = [f for f in os.listdir(heldout_dir) if f.endswith(".jsonl")]
                    if candidates:
                        input_path = os.path.join(heldout_dir, candidates[0])

        output_path = args.output_file if args.output_file else f"output/part1/heldout_result_{args.roll_number}.jsonl"
    else:
        input_path = args.input_file if args.input_file else "data/part1/test/cipher.jsonl"
        output_path = args.output_file if args.output_file else "output/part1/test_decrypt.jsonl"

    if not os.path.exists(input_path):
        raise FileNotFoundError("Input cipher file not found: " + str(input_path))

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print("Running inference on " + str(device) + " for " + str(len(records)) + " samples...", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, rec in enumerate(records):
            item_id = rec.get("id", "")
            ciphertext = rec.get("ciphertext", "")
            decrypted_text = decrypt_sample(model, in_tok, out_tok, ciphertext, device=device)
            out_obj = {"id": item_id, "text": decrypted_text}
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            if (idx + 1) % 100 == 0 or (idx + 1) == len(records):
                print("Decrypted [" + str(idx + 1) + "/" + str(len(records)) + "] samples", flush=True)

    print("Inference completed. Results saved to " + str(output_path), flush=True)

if __name__ == "__main__":
    main()
