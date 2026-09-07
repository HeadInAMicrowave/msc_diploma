#!/usr/bin/env python3
"""
split_dataset.py - stratified train/test split of the teacher-labeled JSONL.

Input rows (one per line), exactly as teacher_client writes them:
    {"tokens": [...], "tags": ["OUT"|"TOXIC", ...], "category": "..."}

Determinism: the shuffle is driven by --seed, which is recorded in
stats.json.

Usage:
    python split_dataset.py --data data/labeled.jsonl --out data/ \
        [--test-frac 0.2] [--seed 42] [--exclude SARCASM_REVIEW]
    Pass --exclude "" to hold nothing out.
"""

import argparse
import json
import random
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

LEGAL_TAGS = {"OUT", "TOXIC"}


def load_rows(path):
    rows, skipped = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                print(f"[skip] line {ln}: not valid JSON", file=sys.stderr)
                continue
            toks, tags, cat = r.get("tokens"), r.get("tags"), r.get("category")
            ok = (isinstance(toks, list) and isinstance(tags, list)
                  and len(toks) == len(tags) and len(toks) > 0
                  and all(t in LEGAL_TAGS for t in tags)
                  and isinstance(cat, str) and cat)
            if not ok:
                skipped += 1
                print(f"[skip] line {ln}: malformed row", file=sys.stderr)
                continue
            rows.append({"tokens": toks, "tags": tags, "category": cat})
    return rows, skipped


def dedupe(rows):
    """Keep the first occurrence of each token sequence. Return the kept rows,
    the number of same-category duplicates dropped, and the list of
    cross-category duplicates (message, first category, later category)."""
    first_cat = {}
    kept, same, cross = [], 0, []
    for r in rows:
        key = " ".join(r["tokens"])
        if key in first_cat:
            if first_cat[key] == r["category"]:
                same += 1
            else:
                cross.append((key, first_cat[key], r["category"]))
            continue
        first_cat[key] = r["category"]
        kept.append(r)
    return kept, same, cross


def branch(tags):
    """The branch the teacher chose: no span, some spans, or whole message."""
    n = sum(t == "TOXIC" for t in tags)
    if n == 0:
        return "out"
    return "fused" if n == len(tags) else "partial"


def split_category(rows, frac, rng):
    """Seeded shuffle, then hold out a fraction with floors: at least one test
    row and at least one train row. A singleton cannot be split."""
    n = len(rows)
    if n < 2:
        return list(rows), []
    n_test = max(1, round(n * frac))
    n_test = min(n_test, n - 1)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[n_test:], shuffled[:n_test]


def stats_for(rows):
    msgs = len(rows)
    toks = sum(len(r["tokens"]) for r in rows)
    toxic = sum(sum(t == "TOXIC" for t in r["tags"]) for r in rows)
    return {
        "messages": msgs,
        "tokens": toks,
        "toxic_tokens": toxic,
        "out_tokens": toks - toxic,
        "toxic_ratio": round(toxic / toks, 4) if toks else 0.0,
    }


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="teacher-labeled JSONL from teacher_client")
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="fraction of each category held out to test (default 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed, recorded in stats.json")
    ap.add_argument("--frozen-test", help="hand-labelled test JSONL to use verbatim as the test set; "
                                          "its messages are removed from train/review")
    ap.add_argument("--exclude", default="SARCASM_REVIEW",
                    help="comma-separated categories written to review.jsonl instead of "
                         "train/test (default SARCASM_REVIEW; pass '' for none)")
    args = ap.parse_args()

    if not (0.0 < args.test_frac < 1.0):
        print("[error] --test-frac must be strictly between 0 and 1", file=sys.stderr)
        sys.exit(1)
    exclude = {c.strip() for c in args.exclude.split(",") if c.strip()}
    rng = random.Random(args.seed)

    rows, skipped = load_rows(args.data)
    if not rows:
        print("[error] no valid rows in input", file=sys.stderr)
        sys.exit(1)
    input_rows = len(rows) + skipped

    rows, dup_same, dup_cross = dedupe(rows)
    for key, first, later in dup_cross:
        print(f"[warn] same message filed under two categories ({first} vs {later}): "
              f"\"{key}\" -- kept the first, check the seed", file=sys.stderr)

    frozen, frozen_keys, frozen_missing = [], set(), 0
    if args.frozen_test:
        with open(args.frozen_test, "r", encoding="utf-8") as f:
            frozen = [json.loads(l) for l in f if l.strip()]
        frozen_keys = {" ".join(r["tokens"]) for r in frozen}
        present = {" ".join(r["tokens"]) for r in rows}
        frozen_missing = sum(1 for k in frozen_keys if k not in present)
        rows = [r for r in rows if " ".join(r["tokens"]) not in frozen_keys]

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    train, test, review = [], list(frozen), []
    per_cat = OrderedDict()
    singletons = []
    for cat in sorted(by_cat):
        cat_rows = by_cat[cat]
        branches = {"out": 0, "partial": 0, "fused": 0}
        for r in cat_rows:
            branches[branch(r["tags"])] += 1

        if cat in exclude:
            review.extend(cat_rows)
            per_cat[cat] = {"count": len(cat_rows), "train": 0, "test": 0,
                            "review": len(cat_rows), **branches}
            continue

        if frozen:
            # The test set is fixed by the frozen file;
            tr, te = list(cat_rows), []
        else:
            tr, te = split_category(cat_rows, args.test_frac, rng)
            if not te:
                singletons.append(cat)
        train.extend(tr)
        test.extend(te)
        per_cat[cat] = {"count": len(cat_rows), "train": len(tr), "test": len(te),
                        "review": 0, **branches}
    if frozen:
        for r in frozen:
            c = r.get("category", "UNCATEGORIZED")
            per_cat.setdefault(c, {"count": 0, "train": 0, "test": 0, "review": 0, "out": 0, "partial": 0, "fused": 0})
            per_cat[c]["test"] += 1

    # Leakage guard
    train_keys = {" ".join(r["tokens"]) for r in train}
    test_keys = {" ".join(r["tokens"]) for r in test}
    assert not (train_keys & test_keys), "train/test leak detected"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "train.jsonl", train)
    dump(out / "test.jsonl", test)
    dump(out / "review.jsonl", review)

    stats = {
        "seed": args.seed,
        "test_frac": args.test_frac if not frozen else None,
        "frozen_test": args.frozen_test,
        "frozen_test_rows": len(frozen) if frozen else None,
        "frozen_rows_absent_from_input": frozen_missing if frozen else None,
        "excluded_categories": sorted(exclude),
        "singleton_categories_all_train": singletons,
        "input_rows": input_rows,
        "skipped_malformed": skipped,
        "duplicates_removed_same_category": dup_same,
        "duplicates_cross_category": [
            {"message": k, "first": a, "later": b} for k, a, b in dup_cross
        ],
        "train": stats_for(train),
        "test": stats_for(test),
        "review": stats_for(review),
        "by_category": per_cat,
    }
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Terminal summary. The out/partial/fused columns, read against each
    # category's author intent, are the author-vs-teacher disagreement view.
    w = max(len(c) for c in per_cat) if per_cat else 8
    print(f"\n{'category':<{w}}  count  train  test  review | out  partial  fused", file=sys.stderr)
    for cat, c in per_cat.items():
        print(f"{cat:<{w}}  {c['count']:>5}  {c['train']:>5}  {c['test']:>4}  {c['review']:>6} | "
              f"{c['out']:>3}  {c['partial']:>7}  {c['fused']:>5}", file=sys.stderr)
    print(f"\n[done] train={len(train)} test={len(test)} review={len(review)} "
          f"(seed={args.seed}, test_frac={args.test_frac}, "
          f"dup_same={dup_same}, dup_cross={len(dup_cross)}, skipped={skipped}) -> {out}/",
          file=sys.stderr)
    if singletons:
        print(f"[note] singleton categories placed entirely in train: "
              f"{', '.join(singletons)}", file=sys.stderr)
    if frozen:
        print(f"[frozen] test set taken verbatim from {args.frozen_test} ({len(frozen)} rows, human labels kept); "
              f"{frozen_missing} of them absent from this input",
              file=sys.stderr)


if __name__ == "__main__":
    main()
