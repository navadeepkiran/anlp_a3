import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

NOISE_CHARS = set("!'+,-./:;<=>?[]_{}")

def strip_noise(text):
    return "".join(ch for ch in text if ch not in NOISE_CHARS)

class CipherTokenizer:
    def __init__(self):
        chars = [" "] + [str(i) for i in range(10)] + [chr(i) for i in range(ord("A"), ord("Z") + 1)] + [chr(i) for i in range(ord("a"), ord("z") + 1)]
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.special = [self.pad_token, self.unk_token]
        self.vocab = self.special + sorted(list(set(chars)))
        self.c2i = {c: i for i, c in enumerate(self.vocab)}
        self.i2c = {i: c for i, c in enumerate(self.vocab)}
        self.pad_id = self.c2i[self.pad_token]
        self.unk_id = self.c2i[self.unk_token]
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return [self.c2i.get(c, self.unk_id) for c in text]

    def decode(self, ids):
        chars = []
        for i in ids:
            if i in (self.pad_id, self.unk_id):
                continue
            chars.append(self.i2c.get(i, ""))
        return "".join(chars)

class PlainTokenizer:
    def __init__(self):
        chars = [" "] + [chr(i) for i in range(ord("a"), ord("z") + 1)]
        self.vocab = sorted(list(set(chars)))
        self.c2i = {c: i for i, c in enumerate(self.vocab)}
        self.i2c = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return [self.c2i.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.i2c.get(i, " ") for i in ids)

class AlignedCipherDataset(Dataset):
    def __init__(self, cipher_path, clean_path=None, in_tok=None, out_tok=None, max_len=1024):
        self.in_tok = in_tok if in_tok is not None else CipherTokenizer()
        self.out_tok = out_tok if out_tok is not None else PlainTokenizer()
        self.max_len = max_len
        self.samples = []

        if not os.path.exists(cipher_path):
            raise FileNotFoundError("Cipher file not found: " + str(cipher_path))

        print("Loading data from " + str(cipher_path) + " ...", flush=True)

        with open(cipher_path, "r", encoding="utf-8") as fc:
            if clean_path is not None and os.path.exists(clean_path):
                with open(clean_path, "r", encoding="utf-8") as fp:
                    for c_line, p_line in zip(fc, fp):
                        c_line = c_line.strip()
                        p_line = p_line.strip()
                        if c_line and p_line:
                            c_obj = json.loads(c_line)
                            p_obj = json.loads(p_line)
                            self.samples.append((c_obj["id"], c_obj["ciphertext"], p_obj["text"]))
            else:
                for c_line in fc:
                    c_line = c_line.strip()
                    if c_line:
                        c_obj = json.loads(c_line)
                        self.samples.append((c_obj["id"], c_obj["ciphertext"], ""))

        print("Loaded " + str(len(self.samples)) + " samples from " + str(cipher_path), flush=True)

        if len(self.samples) == 0:
            raise ValueError("Loaded 0 samples from " + str(cipher_path) + ". Please check if the file is empty or still copying.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item_id, c_raw, p_raw = self.samples[idx]

        c_strip = strip_noise(c_raw)
        if len(c_strip) > self.max_len:
            c_strip = c_strip[:self.max_len]
            p_raw = p_raw[:self.max_len] if p_raw else ""

        input_ids = self.in_tok.encode(c_strip)
        if p_raw:
            p_sub = p_raw[:len(input_ids)]
            labels = self.out_tok.encode(p_sub)
            if len(labels) < len(input_ids):
                labels = labels + [-100] * (len(input_ids) - len(labels))
        else:
            labels = [-100] * len(input_ids)

        return {
            "id": item_id,
            "ciphertext": c_raw,
            "plaintext": p_raw,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def aligned_collate_fn(batch):
    ids = [b["id"] for b in batch]
    ciphers = [b["ciphertext"] for b in batch]
    plains = [b["plaintext"] for b in batch]

    max_len = max(len(b["input_ids"]) for b in batch)
    if max_len == 0:
        max_len = 1

    batch_input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    batch_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.float32)

    for i, b in enumerate(batch):
        seq_len = len(b["input_ids"])
        if seq_len > 0:
            batch_input_ids[i, :seq_len] = b["input_ids"]
            batch_labels[i, :seq_len] = b["labels"]
            attention_mask[i, :seq_len] = 1.0

    return {
        "ids": ids,
        "ciphers": ciphers,
        "plains": plains,
        "input_ids": batch_input_ids,
        "labels": batch_labels,
        "attention_mask": attention_mask
    }

def get_cipher_dataloader(cipher_path, clean_path=None, in_tok=None, out_tok=None, batch_size=64, shuffle=True, max_len=1024, num_workers=0):
    ds = AlignedCipherDataset(cipher_path, clean_path, in_tok=in_tok, out_tok=out_tok, max_len=max_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=aligned_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
