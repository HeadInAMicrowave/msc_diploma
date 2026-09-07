#!/usr/bin/env python3
"""
augment_obfuscation.py - generate disguised-profanity variants of training rows.

Usage:
  python augment_obfuscation.py --data data/train.jsonl --out data/train_aug.jsonl \
      --avoid data/test.jsonl data/review.jsonl \
      [--per-row 3] [--seed 42] \
      [--basic-dump build/Release/basic_dump.exe --vocab artifacts/m2/vocab.txt]
"""

import argparse
import json
import random
import subprocess
import sys
from collections import Counter

STEMS = ("fuck", "shit", "bitch", "ass", "dick", "damn", "crap", "piss", "prick", "bastard", "cunt")
# Never disguise these even if they land in a one-token span: nobody evades a
# filter by writing "y0u", and the variants would only add noise.
STOPWORDS = {"you", "your", "youre", "the", "and", "this", "that", "are", "use", "what", "with", "for",
             "just", "get", "stop", "how", "why", "all", "not", "was", "were", "have", "has", "can"}
LEET_DIGIT = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
LEET_SYMBOL = {"a": "@", "i": "!", "s": "$"}


def is_ascii_punct(ch):
    o = ord(ch)
    return 33 <= o <= 47 or 58 <= o <= 64 or 91 <= o <= 96 or 123 <= o <= 126


def basic_split(surface):
    """Mirror of wp::WordPieceTokenizer::basic_tokenize for ASCII input:
    lowercase A-Z, whitespace separates, each ASCII punctuation char is its own
    token. Verified against the compiled tokenizer when --basic-dump is given."""
    toks, cur = [], ""
    for ch in surface:
        if ch.isspace():
            if cur:
                toks.append(cur)
                cur = ""
        elif ord(ch) < 128 and is_ascii_punct(ch):
            if cur:
                toks.append(cur)
                cur = ""
            toks.append(ch)
        else:
            cur += ch.lower() if "A" <= ch <= "Z" else ch
    if cur:
        toks.append(cur)
    return toks


def variants(word, rng):
    """Disguised surface forms of a lowercase alphabetic word."""
    out = set()
    digit_pos = [i for i, c in enumerate(word) if c in LEET_DIGIT]
    if digit_pos:
        for _ in range(2):
            k = rng.sample(digit_pos, min(len(digit_pos), rng.choice([1, 1, 2])))
            w = list(word)
            for i in k:
                w[i] = LEET_DIGIT[word[i]]
            out.add("".join(w))
    sym_pos = [i for i, c in enumerate(word) if c in LEET_SYMBOL]
    if sym_pos:
        i = rng.choice(sym_pos)
        w = list(word)
        w[i] = LEET_SYMBOL[word[i]]
        out.add("".join(w))
    dropped = word[0] + "".join(c for c in word[1:] if c not in "aeiou")
    if len(dropped) >= 3 and dropped != word:
        out.add(dropped)
    if len(word) <= 6:
        out.add(" ".join(word))
    if len(word) >= 3:
        i = rng.randrange(1, len(word) - 1)
        out.add(word[:i] + word[i] * rng.choice([2, 3]) + word[i + 1:])
    mid_vowels = [i for i, c in enumerate(word) if c in "aeiou" and 0 < i < len(word) - 1]
    if mid_vowels:
        i = rng.choice(mid_vowels)
        out.add(word[:i] + "*" + word[i + 1:])
    out.discard(word)
    return out


def toxic_spans(tags):
    spans, i = [], 0
    while i < len(tags):
        if tags[i] != "TOXIC":
            i += 1
            continue
        j = i
        while j < len(tags) and tags[j] == "TOXIC":
            j += 1
        spans.append((i, j))
        i = j
    return spans


def candidate_positions(tokens, tags):
    cand = []
    for a, b in toxic_spans(tags):
        for k in range(a, b):
            t = tokens[k]
            if t in STOPWORDS:
                continue
            if t.isalpha() and len(t) >= 3 and (b - a == 1 or any(s in t for s in STEMS)):
                cand.append(k)
    return cand


def augment_row(row, rng, per_row):
    cand = candidate_positions(row["tokens"], row["tags"])
    if not cand:
        return []
    out, seen, tries = [], set(), 0
    while len(out) < per_row and tries < per_row * 8:
        tries += 1
        k = rng.choice(cand)
        vs = sorted(variants(row["tokens"][k], rng))
        if not vs:
            continue
        pieces = basic_split(rng.choice(vs))
        if not pieces:
            continue
        toks = row["tokens"][:k] + pieces + row["tokens"][k + 1:]
        tags = row["tags"][:k] + ["TOXIC"] * len(pieces) + row["tags"][k + 1:]
        key = " ".join(toks)
        if key in seen:
            continue
        seen.add(key)
        out.append({"tokens": toks, "tags": tags, "category": row.get("category", "UNCATEGORIZED") + "_AUG"})
    return out


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def verify_with_basic_dump(rows, exe, vocab):
    """Feed each row's spaced join to the compiled tokenizer; its output must equal
    the row's tokens exactly. Any mismatch means the runtime would see different
    tokens than the model was trained on."""
    lines = "\n".join(" ".join(r["tokens"]) for r in rows) + "\n"
    p = subprocess.run([exe, vocab], input=lines, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"[error] basic_dump failed: {p.stderr.strip()}")
    got = p.stdout.split("\n")
    bad = 0
    for r, line in zip(rows, got):
        if line.split(" ") != r["tokens"] and not (line == "" and not r["tokens"]):
            bad += 1
            if bad <= 5:
                print(f"[MISMATCH] want {r['tokens']}\n           got  {line.split(' ')}", file=sys.stderr)
    return bad


def main():
    ap = argparse.ArgumentParser(description="Generate obfuscated-profanity variants of TRAIN rows.")
    ap.add_argument("--data", required=True, help="train.jsonl (never the test split)")
    ap.add_argument("--out", required=True, help="output JSONL: original rows + generated rows")
    ap.add_argument("--avoid", nargs="*", default=[], help="held-out JSONL files; collisions are dropped")
    ap.add_argument("--per-row", type=int, default=3, help="max variants per eligible row (default 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--basic-dump", help="path to the compiled basic_dump tool for parity verification")
    ap.add_argument("--vocab", help="vocab.txt for basic_dump (required with --basic-dump)")
    args = ap.parse_args()
    if args.basic_dump and not args.vocab:
        sys.exit("[error] --vocab is required with --basic-dump")
    if not args.avoid:
        print("[warn] no --avoid files given: generated rows may collide with test set", file=sys.stderr)

    rng = random.Random(args.seed)
    rows = load_jsonl(args.data)
    existing = {" ".join(r["tokens"]) for r in rows}
    avoid = set()
    for path in args.avoid:
        avoid.update(" ".join(r["tokens"]) for r in load_jsonl(path))

    generated, dup, leak, eligible_rows = [], 0, 0, 0
    per_source = Counter()
    for r in rows:
        new = augment_row(r, rng, args.per_row)
        if new:
            eligible_rows += 1
        for n in new:
            key = " ".join(n["tokens"])
            if key in existing:
                dup += 1
                continue
            if key in avoid:
                leak += 1
                continue
            existing.add(key)
            generated.append(n)
            per_source[r.get("category", "UNCATEGORIZED")] += 1

    if args.basic_dump:
        bad = verify_with_basic_dump(generated, args.basic_dump, args.vocab)
        if bad:
            sys.exit(f"[error] {bad} generated row(s) tokenize differently in the runtime; aborting, nothing written")
        print(f"[verify] {len(generated)} generated rows tokenize identically in the runtime", file=sys.stderr)
    else:
        print("[warn] --basic-dump not given: generated tokenization is unverified against the runtime", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows + generated:
            f.write(json.dumps(r) + "\n")

    print(f"[augment] {len(rows)} rows in, {eligible_rows} eligible, {len(generated)} generated "
          f"(dropped: {dup} duplicate, {leak} test-set collision) -> {len(rows) + len(generated)} rows -> {args.out}",
          file=sys.stderr)
    print("[augment] generated per source category: " +
          ", ".join(f"{c}={n}" for c, n in sorted(per_source.items())), file=sys.stderr)


if __name__ == "__main__":
    main()
