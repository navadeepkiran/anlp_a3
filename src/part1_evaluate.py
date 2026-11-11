import os
import json
import argparse
from collections import Counter
import torch

from src.part1_data import CipherTokenizer, PlainTokenizer, strip_noise
from src.part1_inference import load_inference_model

def compute_levenshtein(pred, target):
    try:
        import editdistance
        return editdistance.eval(pred, target)
    except ImportError:
        m, n = len(pred), len(target)
        dp = list(range(n + 1))
        for i, c1 in enumerate(pred):
            new_dp = [i + 1] * (n + 1)
            for j, c2 in enumerate(target):
                cost = 0 if c1 == c2 else 1
                new_dp[j + 1] = min(dp[j + 1] + 1, new_dp[j] + 1, dp[j] + cost)
            dp = new_dp
        return dp[n]

def compute_char_accuracy(pred, target, dist=None):
    max_len = max(len(pred), len(target))
    if max_len == 0:
        return 1.0
    if dist is None:
        dist = compute_levenshtein(pred, target)
    return max(0.0, 1.0 - (dist / max_len))

def compute_word_accuracy(pred, target):
    p_words = pred.strip().split()
    t_words = target.strip().split()
    if not p_words and not t_words:
        return 1.0
    if not p_words or not t_words:
        return 0.0
    matches = sum(1 for pw, tw in zip(p_words, t_words) if pw == tw)
    return matches / max(len(p_words), len(t_words))

def compute_precision_recall_f1(pred, target):
    p_tokens = pred.strip().split()
    t_tokens = target.strip().split()
    if not p_tokens and not t_tokens:
        return 1.0, 1.0, 1.0
    if not p_tokens or not t_tokens:
        return 0.0, 0.0, 0.0
    p_counter = Counter(p_tokens)
    t_counter = Counter(t_tokens)
    intersection = sum((p_counter & t_counter).values())
    precision = intersection / len(p_tokens)
    recall = intersection / len(t_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, sample_input_ids):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as mode:
            model(sample_input_ids)
        return int(mode.get_total_flops())
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            analysis = FlopCountAnalysis(model, sample_input_ids)
            return int(analysis.total())
        except Exception:
            return count_parameters(model) * sample_input_ids.shape[1] * 2

def evaluate_dict(preds_dict, targets_dict):
    common_ids = sorted(list(set(preds_dict.keys()) & set(targets_dict.keys())))
    if not common_ids:
        raise ValueError("No common IDs between predictions and targets!")

    char_accs = []
    word_accs = []
    precisions = []
    recalls = []
    f1s = []
    levs = []

    phase_totals = [0] * 6
    phase_corrects = [0] * 6

    pos_totals = [0] * 20
    pos_corrects = [0] * 20

    bucket_totals = {"0-1k": 0, "1-2k": 0, "2k+": 0}
    bucket_corrects = {"0-1k": 0, "1-2k": 0, "2k+": 0}

    for item_id in common_ids:
        p = preds_dict[item_id]
        t = targets_dict[item_id]
        lev = compute_levenshtein(p, t)
        ca = compute_char_accuracy(p, t, dist=lev)
        wa = compute_word_accuracy(p, t)
        prec, rec, f1 = compute_precision_recall_f1(p, t)

        levs.append(lev)
        char_accs.append(ca)
        word_accs.append(wa)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        min_len = min(len(p), len(t))
        for idx in range(min(min_len, 20)):
            pos_totals[idx] += 1
            if p[idx] == t[idx]:
                pos_corrects[idx] += 1

        for idx in range(min_len):
            ph = idx % 6
            phase_totals[ph] += 1
            if p[idx] == t[idx]:
                phase_corrects[ph] += 1

            if idx < 1000:
                b_key = "0-1k"
            elif idx < 2000:
                b_key = "1-2k"
            else:
                b_key = "2k+"

            bucket_totals[b_key] += 1
            if p[idx] == t[idx]:
                bucket_corrects[b_key] += 1

    n = len(common_ids)
    pos_accuracies = {
        "pos_" + str(i): float(pos_corrects[i] / max(1, pos_totals[i]))
        for i in range(20)
    }
    phase_accuracies = {
        "phase_" + str(i): float(phase_corrects[i] / max(1, phase_totals[i]))
        for i in range(6)
    }
    bucket_accuracies = {
        k: float(bucket_corrects[k] / max(1, bucket_totals[k]))
        for k in bucket_totals
    }

    return {
        "samples_evaluated": n,
        "character_accuracy": float(sum(char_accs) / n),
        "word_accuracy": float(sum(word_accs) / n),
        "precision": float(sum(precisions) / n),
        "recall": float(sum(recalls) / n),
        "f1": float(sum(f1s) / n),
        "levenshtein_distance": float(sum(levs) / n),
        "per_position_accuracy_first20": pos_accuracies,
        "phase_accuracies_mod6": phase_accuracies,
        "bucket_accuracies": bucket_accuracies
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default="output/part1/test_decrypt.jsonl")
    parser.add_argument("--target", type=str, default="data/part1/test/cleaned.jsonl")
    parser.add_argument("--model_name", type=str, default="hybrid")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    preds_dict = {}
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                preds_dict[obj["id"]] = obj.get("text", "")

    targets_dict = {}
    with open(args.target, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                targets_dict[obj["id"]] = obj.get("text", "")

    metrics = evaluate_dict(preds_dict, targets_dict)

    cfg_file = args.config if args.config else os.path.join("config", "part1", args.model_name + ".yaml")
    if os.path.exists(cfg_file):
        try:
            model, _, _ = load_inference_model(cfg_file, from_hf=False, device="cpu")
            param_count = count_parameters(model)
            dummy_input = torch.randint(0, 65, (1, 512))
            flops = estimate_flops(model, dummy_input)
            metrics["parameter_count"] = param_count
            metrics["flops_seq512"] = flops
        except Exception as e:
            metrics["parameter_count_error"] = str(e)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(str(k) + ": " + str(round(v, 4)))
        else:
            print(str(k) + ": " + str(v))

    out_file = args.output_json if args.output_json else os.path.join("results", "part1", args.model_name + ".json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to " + str(out_file), flush=True)

if __name__ == "__main__":
    main()
