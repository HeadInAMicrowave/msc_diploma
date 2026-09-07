#!/usr/bin/env python3
"""
train_export.py  --  the ONLY Python in the pipeline (build-time tool).

Input JSONL format (one object per line), tags are label STRINGS:
    {"tokens": ["you","are","a","noob","!"], "tags": ["KEEP","KEEP","KEEP","PROF","KEEP"]}

Run (use Python 3.11 or 3.12: torch.jit.trace is unsupported on 3.14+ and warns it may break):
    pip install "torch" "transformers"
    python train_export.py --data data/train.jsonl --schema ../config/schema.json --out artifacts/
"""

import argparse, json, os, random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

IGNORE_INDEX = -100  # standard "don't compute loss here" label for subword continuations


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class TokenTagDataset(Dataset):
    """Aligns word-level labels to subword tokens: the FIRST subword of each word
    carries the label, continuation subwords and specials get IGNORE_INDEX. This
    matches how the C++ side reads predictions back (first_piece per word)."""

    def __init__(self, rows, tokenizer, label2id, max_length):
        self.rows = rows
        self.tok = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tok(
            row["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        word_ids = enc.word_ids()
        labels, prev = [], None
        for wid in word_ids:
            if wid is None:
                labels.append(IGNORE_INDEX)
            elif wid != prev:
                labels.append(self.label2id[row["tags"][wid]])
            else:
                labels.append(IGNORE_INDEX)
            prev = wid
        enc["labels"] = labels
        return enc


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)

    def pad(seq, val):
        return seq + [val] * (maxlen - len(seq))

    input_ids = torch.tensor([pad(b["input_ids"], pad_id) for b in batch], dtype=torch.long)
    attn = torch.tensor([pad(b["attention_mask"], 0) for b in batch], dtype=torch.long)
    labels = torch.tensor([pad(b["labels"], IGNORE_INDEX) for b in batch], dtype=torch.long)
    return input_ids, attn, labels


class LogitsOnly(torch.nn.Module):
    """TorchScript-friendly wrapper: HF models return a dataclass that trace()
    dislikes. Expose a plain (input_ids, attention_mask) -> logits signature."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="teacher-labeled train JSONL")
    ap.add_argument("--schema", default="../config/schema.json")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    schema = json.load(open(args.schema, encoding="utf-8"))
    label2id = {k: int(v) for k, v in schema["label2id"].items()}
    id2label = {int(k): v for k, v in schema["id2label"].items()}
    base_model = schema["base_model"]
    max_length = int(schema["max_length"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] base={base_model} device={device} labels={id2label}")

    # strip_accents=False keeps character offsets stable between Python and the
    # C++ WordPiece reimplementation (see wordpiece.hpp scope notes).
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, do_lower_case=schema["do_lower_case"], strip_accents=schema["strip_accents"]
    )
    # attn_implementation="eager": the pure-tensor attention path. Newer transformers
    # default to SDPA + masking_utils, which contain Python-level branches on tensor
    # values that torch.jit.trace freezes at the example input's shape (the
    # TracerWarnings).
    model = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=len(label2id), id2label=id2label, label2id=label2id,
        attn_implementation="eager",
    ).to(device)

    rows = load_jsonl(args.data)
    print(f"[train] {len(rows)} examples")
    ds = TokenTagDataset(rows, tokenizer, label2id, max_length)
    pad_id = tokenizer.pad_token_id or 0
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, pad_id))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for input_ids, attn, labels in dl:
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            out_m = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out_m.loss.backward()
            opt.step(); opt.zero_grad()
            total += out_m.loss.item()
        print(f"[train] epoch {epoch+1}/{args.epochs} loss={total/max(1,len(dl)):.4f}")

    # export TorchScript
    model.eval()
    wrapper = LogitsOnly(model).to(device).eval()
    ex_ids = torch.ones(1, min(8, max_length), dtype=torch.long, device=device)
    ex_mask = torch.ones_like(ex_ids)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (ex_ids, ex_mask))

    lengths = sorted({3, 8, 17, 33, min(64, max_length)})
    with torch.no_grad():
        for L in lengths:
            ids = torch.randint(999, tokenizer.vocab_size, (1, L), dtype=torch.long, device=device)
            ids[0, 0], ids[0, -1] = tokenizer.cls_token_id, tokenizer.sep_token_id
            mask = torch.ones_like(ids)
            ref, got = wrapper(ids, mask), traced(ids, mask)
            if ref.shape != got.shape or not torch.allclose(ref, got, atol=1e-4, rtol=1e-4):
                raise RuntimeError(
                    f"traced model diverges from the live model at seq_len={L} "
                    f"(max abs diff {(ref - got).abs().max().item() if ref.shape == got.shape else 'shape mismatch'}). "
                    "model.pt NOT written. Use Python <= 3.12 for this step and keep attn_implementation='eager'.")
    print(f"[export] trace verified against the live model at seq lens {lengths}")
    traced.save(str(out / "model.pt"))
    print(f"[export] wrote {out/'model.pt'}")

    vocab = tokenizer.get_vocab()                      # token -> id
    inv = {i: t for t, i in vocab.items()}
    if sorted(inv) != list(range(len(inv))):
        raise RuntimeError("tokenizer vocab ids are not dense 0..V-1; wordpiece.hpp cannot load it")
    with open(out / "vocab.txt", "w", encoding="utf-8", newline="\n") as f:
        for i in range(len(inv)):
            f.write(inv[i] + "\n")
    if not (out / "vocab.txt").exists():
        raise RuntimeError("vocab.txt was not written")
    print(f"[export] wrote {out/'vocab.txt'} ({len(inv)} tokens)")
    json.dump(
        {"id2label": schema["id2label"], "label2id": schema["label2id"],
         "max_length": max_length, "do_lower_case": schema["do_lower_case"]},
        open(out / "labels.json", "w", encoding="utf-8"), indent=2,
    )
    print(f"[export] wrote {out/'labels.json'}")

    probes = [
        "You are a NOOB!",
        "gg wp everyone",
        "what the f u c k was that",
        "nice shot :)",
        "l33t h4x0r spotted",
    ]
    fixture = []
    for s in probes:
        ids = tokenizer(s, truncation=True, max_length=max_length)["input_ids"]
        fixture.append({"text": s, "input_ids": ids})
    json.dump(fixture, open(out / "parity_expected.json", "w", encoding="utf-8"), indent=2)
    print(f"[export] wrote {out/'parity_expected.json'} ({len(probes)} probes)")
    print("[done] Run the C++ parity_test against these artifacts BEFORE trusting inference.")


if __name__ == "__main__":
    main()
