#!/usr/bin/env python3
"""
evaluate.py -- evaluation harness for the two-model detoxification system.

Usage:
  python evaluate.py prepare --gold data/test.jsonl --out messages.txt
  orchestrator --m1 m1/ --m2 m2/ --json < messages.txt > results.jsonl
  orchestrator --m2 m2/ --tag-only      < messages.txt > spans.jsonl
  python evaluate.py score --gold data/test.jsonl --results results.jsonl --report report.json
  python evaluate.py score --gold data/test.jsonl --results spans.jsonl   --report tagger.json

  python evaluate.py blind-sheet --gold data/test.jsonl --out sheet.txt --seed 7
  ... mark up sheet.txt by hand ...
  python evaluate.py ingest --gold data/test.jsonl --sheet sheet.txt --out data/test_human.jsonl
  python evaluate.py agree  --a data/test_human.jsonl --b data/test.jsonl --report agreement.json
  python evaluate.py score  --gold data/test_human.jsonl --results results.jsonl --report report.json
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict

NEG_INF = float("-inf")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    sys.exit(f"[error] {path} line {ln}: not valid JSON")
    return rows


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def semeval_f1(pred, gold):
    """Per-message F1 over token positions, SemEval-2021 Task 5 convention:
    both empty -> 1.0, exactly one empty -> 0.0."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    return prf(tp, len(pred - gold), len(gold - pred))[2]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def is_punct_token(t):
    """A single ASCII punctuation character. The basic tokenizer emits punctuation
    as single-character tokens, so this identifies them exactly."""
    if len(t) != 1:
        return False
    o = ord(t)
    return 33 <= o <= 47 or 58 <= o <= 64 or 91 <= o <= 96 or 123 <= o <= 126


def token_offsets(tokens):
    """Byte offsets of each token inside ' '.join(tokens)."""
    offs, pos = [], 0
    for t in tokens:
        offs.append((pos, pos + len(t.encode("utf-8"))))
        pos += len(t.encode("utf-8")) + 1
    return offs


def align_charitable(inp, out, tags):
    """Embed `out` into `inp` as a subsequence, maximising the number of matched
    positions whose gold tag is OUT. Returns the set of surviving input
    positions, or None if `out` is not a subsequence of `inp`."""
    n, m = len(inp), len(out)
    dp = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][m] = 0.0
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best = dp[i + 1][j]
            if inp[i] == out[j] and dp[i + 1][j + 1] > NEG_INF:
                best = max(best, (1.0 if tags[i] == "OUT" else 0.0) + dp[i + 1][j + 1])
            dp[i][j] = best
    if dp[0][0] == NEG_INF:
        return None
    survived, i, j = set(), 0, 0
    while j < m:
        take = (inp[i] == out[j] and dp[i + 1][j + 1] > NEG_INF and
                (1.0 if tags[i] == "OUT" else 0.0) + dp[i + 1][j + 1] == dp[i][j])
        if take:
            survived.add(i)
            j += 1
        i += 1
    return survived



def cmd_prepare(args):
    gold = load_jsonl(args.gold)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in gold:
            f.write(" ".join(r["tokens"]) + "\n")
    print(f"[prepare] wrote {len(gold)} messages -> {args.out}", file=sys.stderr)



class Acc:
    """Metric accumulator; one per category plus one for ALL.

    ignore_punct=True computes every token-level figure with punctuation tokens
    treated as don't-care on both sides. The blind pass showed that most human vs
    teacher disagreement is whether a binding comma sits inside the span; the
    strict figure counts that, the punctuation-blind figure does not. Both are
    reported."""

    def __init__(self, ignore_punct=False):
        self.ignore_punct = ignore_punct
        self.n = 0
        self.actions = Counter()
        self.gate = Counter()                 # tp fp fn tn at message level
        self.out_kept = self.out_total = 0    # content
        self.tox_removed = self.tox_total = 0 # toxicity
        self.removed_total = 0                # for removal precision
        self.tok_tp = self.tok_fp = self.tok_fn = 0
        self.content_macro = []
        self.toxic_macro = []
        self.span_f1 = []
        self.exact = 0
        self.deleted = 0
        self.cleared = 0
        # tagger-only extras
        self.fused_rows = self.fused_detected = 0
        self.clean_rows = self.clean_false_alarm = 0

    def add_tokens(self, tags, removed_positions, tokens=None):
        keep = set(range(len(tags)))
        if self.ignore_punct and tokens is not None:
            keep = {i for i in keep if not is_punct_token(tokens[i])}
        gold_tox = {i for i in keep if tags[i] == "TOXIC"}
        gold_out = keep - gold_tox
        removed_positions = set(removed_positions) & keep
        kept = gold_out - removed_positions
        self.out_kept += len(kept)
        self.out_total += len(gold_out)
        self.tox_removed += len(gold_tox & removed_positions)
        self.tox_total += len(gold_tox)
        self.removed_total += len(removed_positions)
        self.tok_tp += len(gold_tox & removed_positions)
        self.tok_fp += len(removed_positions - gold_tox)
        self.tok_fn += len(gold_tox - removed_positions)
        if gold_out:
            self.content_macro.append(len(kept) / len(gold_out))
        if gold_tox:
            self.toxic_macro.append(len(gold_tox & removed_positions) / len(gold_tox))
        self.span_f1.append(semeval_f1(removed_positions, gold_tox))
        if removed_positions == gold_tox:
            self.exact += 1
        if gold_tox and not gold_out:
            self.fused_rows += 1
            if removed_positions == gold_tox:
                self.fused_detected += 1
        if not gold_tox:
            self.clean_rows += 1
            if removed_positions:
                self.clean_false_alarm += 1

    def report(self, mode):
        gp, gr, gf = prf(self.gate["tp"], self.gate["fp"], self.gate["fn"])
        tp, tr, tf = prf(self.tok_tp, self.tok_fp, self.tok_fn)
        rep = {
            "n": self.n,
            "content_kept_micro": self.out_kept / self.out_total if self.out_total else None,
            "content_kept_macro": mean(self.content_macro) if self.content_macro else None,
            "toxic_removed_micro": self.tox_removed / self.tox_total if self.tox_total else None,
            "toxic_removed_macro": mean(self.toxic_macro) if self.toxic_macro else None,
            "removal_precision_micro": self.tok_tp / self.removed_total if self.removed_total else None,
            "token_precision": tp, "token_recall": tr, "token_f1": tf,
            "span_f1_macro": mean(self.span_f1),
            "exact_match_rate": self.exact / self.n if self.n else 0.0,
            "fused_detected_rate": self.fused_detected / self.fused_rows if self.fused_rows else None,
            "clean_false_alarm_rate": self.clean_false_alarm / self.clean_rows if self.clean_rows else None,
        }
        if mode == "system":
            rep.update({
                "actions": dict(self.actions),
                "deleted_rate": self.deleted / self.n if self.n else 0.0,
                "gate_clearance_rate_self_judged": self.cleared / self.n if self.n else 0.0,
                "gate": {"tp": self.gate["tp"], "fp": self.gate["fp"], "fn": self.gate["fn"],
                         "tn": self.gate["tn"], "precision": gp, "recall": gr, "f1": gf},
            })
        return rep



def pair_rows(gold, results):
    if len(gold) != len(results):
        sys.exit(f"[error] gold has {len(gold)} rows but results has {len(results)}; "
                 f"regenerate results from `prepare` output")
    for k, (g, r) in enumerate(zip(gold, results)):
        if r.get("input") != " ".join(g["tokens"]):
            sys.exit(f"[error] row {k}: results input does not match gold tokens; order is corrupted")
    return list(zip(gold, results))


def score_system(gold, results):
    accs = defaultdict(Acc)
    accs_p = defaultdict(lambda: Acc(ignore_punct=True))
    breaches = []
    lat, iters, gcalls, tcalls = [], [], [], []

    paired = pair_rows(gold, results)
    for k, (g, r) in enumerate(paired):
        tags = g["tags"]
        gold_tox_msg = "TOXIC" in tags
        pred_tox_msg = r["action"] != "pass"
        out_tokens = r["output"].split()
        survived = align_charitable(g["tokens"], out_tokens, tags)
        if survived is None:
            breaches.append({"row": k, "input": r["input"], "output": r["output"],
                             "reason": "output is not a subsequence of input"})
            continue
        removed = set(range(len(tags))) - survived

        cat = g.get("category", "UNCATEGORIZED")
        for a in (accs["ALL"], accs[cat], accs_p["ALL"], accs_p[cat]):
            a.n += 1
            a.actions[r["action"]] += 1
            a.gate["tp" if gold_tox_msg and pred_tox_msg else
                   "fp" if pred_tox_msg else
                   "fn" if gold_tox_msg else "tn"] += 1
            a.add_tokens(tags, removed, g["tokens"])
            a.deleted += 1 if r.get("deleted") else 0
            a.cleared += 0 if r["action"].startswith("kept_partial") else 1

        lat.append(r.get("elapsed_ms", 0.0))
        iters.append(r.get("iterations", 0))
        gcalls.append(r.get("gate_calls", 0))
        tcalls.append(r.get("tagger_calls", 0))

    # Gate calibration: initial_score does not depend on the orchestrator's threshold,
    # so the gate's message-level P/R/F1 at every threshold is available from one run.
    scored = [(r["initial_score"], "TOXIC" in g["tags"]) for g, r in paired]
    sweep = []
    for t in [i / 20 for i in range(1, 20)]:
        tp = sum(1 for sc, y in scored if sc >= t and y)
        fp = sum(1 for sc, y in scored if sc >= t and not y)
        fn = sum(1 for sc, y in scored if sc < t and y)
        p_, r_, f_ = prf(tp, fp, fn)
        sweep.append({"threshold": round(t, 2), "precision": p_, "recall": r_, "f1": f_})
    unk = [r["gate_unk_rate"] for _, r in paired if "gate_unk_rate" in r]

    report = {
        "mode": "system",
        "rows": len(gold),
        "scored": accs["ALL"].n,
        "invariant_breaches": breaches,
        "gate_threshold_sweep": sweep,
        "gate_unk_rate_mean": mean(unk) if unk else None,
        "overall": accs["ALL"].report("system"),
        "by_category": {c: a.report("system") for c, a in sorted(accs.items()) if c != "ALL"},
        "overall_punct_ignored": accs_p["ALL"].report("system"),
        "by_category_punct_ignored": {c: a.report("system") for c, a in sorted(accs_p.items()) if c != "ALL"},
        "latency": {
            "elapsed_ms_mean": mean(lat), "elapsed_ms_median": statistics.median(lat) if lat else 0.0,
            "elapsed_ms_p95": percentile(lat, 0.95), "elapsed_ms_max": max(lat) if lat else 0.0,
            "iterations_hist": dict(sorted(Counter(iters).items())),
            "gate_calls_mean": mean(gcalls), "tagger_calls_mean": mean(tcalls),
        },
        "notes": [
            "gate_clearance_rate_self_judged: model 1 judging its own loop's output.",
            "fluency is not measured automatically.",
            "deleted messages count as toxicity removed but content lost..",
        ],
    }
    return report


def score_tagger(gold, results):
    accs = defaultdict(Acc)
    accs_p = defaultdict(lambda: Acc(ignore_punct=True))
    for g, r in pair_rows(gold, results):
        tags = g["tags"]
        offs = token_offsets(g["tokens"])
        pred = set()
        for s in r.get("spans", []):
            for i, (a, b) in enumerate(offs):
                if a < s["end"] and s["start"] < b:
                    pred.add(i)
        cat = g.get("category", "UNCATEGORIZED")
        for a in (accs["ALL"], accs[cat], accs_p["ALL"], accs_p[cat]):
            a.n += 1
            a.add_tokens(tags, pred, g["tokens"])
    return {
        "mode": "tagger",
        "rows": len(gold),
        "overall": accs["ALL"].report("tagger"),
        "by_category": {c: a.report("tagger") for c, a in sorted(accs.items()) if c != "ALL"},
        "overall_punct_ignored": accs_p["ALL"].report("tagger"),
        "by_category_punct_ignored": {c: a.report("tagger") for c, a in sorted(accs_p.items()) if c != "ALL"},
        "notes": ["model 2 evaluated alone: its raw spans against gold, no gate, no loop."],
    }


def fmt(x, pct=True):
    if x is None:
        return "   -  "
    return f"{100 * x:5.1f}%" if pct else f"{x:6.2f}"


def print_report(rep):
    o = rep["overall"]
    print(f"\n== {rep['mode']} evaluation: {rep.get('scored', rep['rows'])} rows scored ==")
    if rep.get("invariant_breaches"):
        print(f"!! {len(rep['invariant_breaches'])} invariant breach(es); those rows were not scored")
    print(f"content kept   (OUT recall)   micro {fmt(o['content_kept_micro'])}  macro {fmt(o['content_kept_macro'])}")
    print(f"toxic removed  (TOXIC recall) micro {fmt(o['toxic_removed_micro'])}  macro {fmt(o['toxic_removed_macro'])}")
    print(f"removal precision             micro {fmt(o['removal_precision_micro'])}")
    print(f"token P/R/F1                  {fmt(o['token_precision'])} {fmt(o['token_recall'])} {fmt(o['token_f1'])}")
    print(f"span F1 (SemEval, macro)      {fmt(o['span_f1_macro'])}")
    print(f"exact match (perfect rewrite) {fmt(o['exact_match_rate'])}")
    print(f"fused detected / clean false-alarm   {fmt(o['fused_detected_rate'])} / {fmt(o['clean_false_alarm_rate'])}")
    q = rep.get("overall_punct_ignored")
    if q:
        print(f"-- punctuation tokens as don't-care:  content {fmt(q['content_kept_micro'])}  toxic {fmt(q['toxic_removed_micro'])}  "
              f"token F1 {fmt(q['token_f1'])}  span F1 {fmt(q['span_f1_macro'])}  exact {fmt(q['exact_match_rate'])}")
    if rep["mode"] == "system":
        g = o["gate"]
        print(f"gate (model 1, message-level) P/R/F1 {fmt(g['precision'])} {fmt(g['recall'])} {fmt(g['f1'])}"
              f"   tp={g['tp']} fp={g['fp']} fn={g['fn']} tn={g['tn']}")
        print(f"deleted rate {fmt(o['deleted_rate'])}   gate clearance (self-judged) {fmt(o['gate_clearance_rate_self_judged'])}")
        if rep.get("gate_unk_rate_mean") is not None:
            print(f"gate [UNK] rate (mean fraction of tokens outside model 1's vocab) {fmt(rep['gate_unk_rate_mean'])}")
        sw = rep.get("gate_threshold_sweep") or []
        if sw:
            best = max(sw, key=lambda x: x["f1"])
            print("gate threshold sweep (message-level):  " + "  ".join(
                f"τ={x['threshold']:.2f} F1={100*x['f1']:.0f}%" for x in sw if round(x["threshold"] * 20) % 2 == 0))
            print(f"  best F1 {fmt(best['f1'])} at τ={best['threshold']:.2f} "
                  f"(P {fmt(best['precision'])} R {fmt(best['recall'])}); run used τ per the [ready] line")
        print("actions: " + ", ".join(f"{k}={v}" for k, v in sorted(o["actions"].items())))
        L = rep["latency"]
        print(f"latency ms: mean {L['elapsed_ms_mean']:.2f}  median {L['elapsed_ms_median']:.2f}  "
              f"p95 {L['elapsed_ms_p95']:.2f}  max {L['elapsed_ms_max']:.2f}   iterations {L['iterations_hist']}")

    w = max((len(c) for c in rep["by_category"]), default=8)
    hdr = f"\n{'category':<{w}}    n  content  toxic   spanF1  exact "
    hdr += "  deleted  actions" if rep["mode"] == "system" else "  fused  false-alarm"
    print(hdr)
    for c, a in rep["by_category"].items():
        line = (f"{c:<{w}}  {a['n']:>3}  {fmt(a['content_kept_micro'])}  {fmt(a['toxic_removed_micro'])}  "
                f"{fmt(a['span_f1_macro'])}  {fmt(a['exact_match_rate'])}")
        if rep["mode"] == "system":
            acts = ", ".join(f"{k}={v}" for k, v in sorted(a["actions"].items()))
            line += f"  {fmt(a['deleted_rate'])}  {acts}"
        else:
            line += f"  {fmt(a['fused_detected_rate'])}  {fmt(a['clean_false_alarm_rate'])}"
        print(line)
    for n in rep.get("notes", []):
        print(f"[note] {n}")



def parse_markup(original, marked):
    """Python twin of sm::parse_markup (span_markup.hpp): single pass, same
    failure reasons. Returns (spans, "") or (None, reason). Char offsets."""
    OPEN, CLOSE = "<del>", "</del>"
    spans, i, o, in_del, start = [], 0, 0, False, 0
    while i < len(marked):
        if marked.startswith(OPEN, i):
            if in_del:
                return None, "nested <del>"
            in_del, start, i = True, o, i + len(OPEN)
            continue
        if marked.startswith(CLOSE, i):
            if not in_del:
                return None, "stray </del>"
            if o == start:
                return None, "empty <del></del>"
            spans.append((start, o))
            in_del, i = False, i + len(CLOSE)
            continue
        if o >= len(original) or marked[i] != original[o]:
            return None, "untagged text differs from original"
        i += 1
        o += 1
    if in_del:
        return None, "unclosed <del>"
    if o != len(original):
        return None, "output shorter than original"
    return spans, ""


def token_char_offsets(tokens):
    offs, pos = [], 0
    for t in tokens:
        offs.append((pos, pos + len(t)))
        pos += len(t) + 1
    return offs


def render(tokens, tags):
    """Bracket TOXIC runs for human reading: 'you [fucking noob ,] group mid'."""
    out, in_run = [], False
    for t, g in zip(tokens, tags):
        if g == "TOXIC" and not in_run:
            out.append("[" + t)
            in_run = True
        elif g != "TOXIC" and in_run:
            out[-1] += "]"
            out.append(t)
            in_run = False
        else:
            out.append(t)
    if in_run:
        out[-1] += "]"
    return " ".join(out)


def cmd_blind_sheet(args):
    gold = load_jsonl(args.gold)
    order = list(range(len(gold)))
    random.Random(args.seed).shuffle(order)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for i in order:
            f.write(" ".join(gold[i]["tokens"]) + "\n")
    with open(args.out + ".order.json", "w", encoding="utf-8") as f:
        json.dump({"gold": args.gold, "seed": args.seed, "order": order}, f)
    print(f"[blind-sheet] {len(gold)} messages, shuffled with seed {args.seed} -> {args.out}\n"
          f"              line->row map -> {args.out}.order.json (keep it next to the sheet)\n"
          f"  Mark up each line in place with <del>...</del>: deletion only, no rewording;\n"
          f"  wrap the whole line if fused; leave clean lines untouched; prefix '?' if the\n"
          f"  policy does not settle it. Do not add, remove or reorder lines. Do this before\n"
          f"  reading the teacher's labels, then freeze the file.", file=sys.stderr)


def cmd_ingest(args):
    gold = load_jsonl(args.gold)
    with open(args.sheet + ".order.json", "r", encoding="utf-8") as f:
        order = json.load(f)["order"]
    with open(args.sheet, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    while lines and lines[-1].strip() == "":
        lines.pop()
    if len(lines) != len(order):
        sys.exit(f"[error] sheet has {len(lines)} lines but the order file expects {len(order)}; "
                 f"lines were added or removed")

    rows, bad, flagged = [None] * len(gold), [], 0
    for ln, (line, gi) in enumerate(zip(lines, order), 1):
        g = gold[gi]
        original = " ".join(g["tokens"])
        marked = line.rstrip("\r")
        flag = marked.startswith("?")
        if flag:
            marked = marked[1:].lstrip()
        spans, why = parse_markup(original, marked)
        if spans is None:
            bad.append((ln, why, marked))
            continue
        offs = token_char_offsets(g["tokens"])
        tags = ["OUT"] * len(g["tokens"])
        misaligned = False
        for s0, e0 in spans:
            for i, (a, b) in enumerate(offs):
                if a < s0 < b or a < e0 < b:
                    misaligned = True
                if a < e0 and s0 < b:
                    tags[i] = "TOXIC"
        if misaligned:
            bad.append((ln, "span boundary falls inside a word", marked))
            continue
        row = {"tokens": g["tokens"], "tags": tags, "tags_teacher": g["tags"],
               "category": g.get("category", "UNCATEGORIZED")}
        if flag:
            row["flag"] = "uncertain"
            flagged += 1
        rows[gi] = row

    if bad:
        for ln, why, text in bad:
            print(f"[bad] sheet line {ln}: {why}\n      {text}", file=sys.stderr)
        sys.exit(f"[error] {len(bad)} line(s) failed; fix them and re-run. Nothing was written.")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    exact = sum(r["tags"] == r["tags_teacher"] for r in rows)
    print(f"[ingest] {len(rows)} rows -> {args.out}; {flagged} flagged uncertain; "
          f"{exact}/{len(rows)} identical to the teacher's labels (run `agree` for the full picture)",
          file=sys.stderr)


def kappa(agree, n, pa, pb):
    """Cohen's kappa from observed agreement and the two marginal positive rates."""
    if n == 0:
        return None
    po = agree / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def cmd_agree(args):
    A, B = load_jsonl(args.a), load_jsonl(args.b)
    if len(A) != len(B):
        sys.exit(f"[error] {args.a} has {len(A)} rows, {args.b} has {len(B)}")
    def fresh():
        return {"n": 0, "tok_n": 0, "tok_agree": 0, "a_tox": 0, "b_tox": 0,
                "msg_agree": 0, "a_msg": 0, "b_msg": 0, "span_f1": [], "exact": 0}
    stats = defaultdict(fresh)            # strict
    stats_p = defaultdict(fresh)          # punctuation tokens as don't-care
    disagreements = []
    conv_only = 0                          # rows that agree once punctuation is ignored
    for k, (a, b) in enumerate(zip(A, B)):
        if a["tokens"] != b["tokens"]:
            sys.exit(f"[error] row {k}: token sequences differ; the files are not the same messages")
        if args.drop_flagged and (a.get("flag") or b.get("flag")):
            continue
        ta, tb = a["tags"], b["tags"]
        sa = {i for i, t in enumerate(ta) if t == "TOXIC"}
        sb = {i for i, t in enumerate(tb) if t == "TOXIC"}
        words = {i for i, t in enumerate(a["tokens"]) if not is_punct_token(t)}
        for target, sa_, sb_, idx in ((stats, sa, sb, set(range(len(ta)))),
                                      (stats_p, sa & words, sb & words, words)):
            for cat in ("ALL", a.get("category", "UNCATEGORIZED")):
                st = target[cat]
                st["n"] += 1
                st["tok_n"] += len(idx)
                st["tok_agree"] += sum(1 for i in idx if ta[i] == tb[i])
                st["a_tox"] += len(sa_)
                st["b_tox"] += len(sb_)
                st["msg_agree"] += int(bool(sa_) == bool(sb_))
                st["a_msg"] += int(bool(sa_))
                st["b_msg"] += int(bool(sb_))
                st["span_f1"].append(semeval_f1(sa_, sb_))
                st["exact"] += int(sa_ == sb_)
        if sa != sb:
            convention = (sa & words) == (sb & words)
            conv_only += int(convention)
            disagreements.append({"row": k, "category": a.get("category"), "flag": a.get("flag") or b.get("flag"),
                                  "punctuation_only": convention,
                                  "a": render(a["tokens"], ta), "b": render(b["tokens"], tb)})

    def summarize(st):
        return {
            "n": st["n"],
            "token_kappa": kappa(st["tok_agree"], st["tok_n"],
                                 st["a_tox"] / st["tok_n"] if st["tok_n"] else 0,
                                 st["b_tox"] / st["tok_n"] if st["tok_n"] else 0),
            "token_agreement": st["tok_agree"] / st["tok_n"] if st["tok_n"] else None,
            "message_kappa": kappa(st["msg_agree"], st["n"],
                                   st["a_msg"] / st["n"] if st["n"] else 0,
                                   st["b_msg"] / st["n"] if st["n"] else 0),
            "span_f1_macro": mean(st["span_f1"]),
            "exact_match_rate": st["exact"] / st["n"] if st["n"] else None,
        }

    rep = {"a": args.a, "b": args.b, "drop_flagged": args.drop_flagged,
           "overall": summarize(stats["ALL"]),
           "overall_punct_ignored": summarize(stats_p["ALL"]),
           "by_category": {c: summarize(st) for c, st in sorted(stats.items()) if c != "ALL"},
           "by_category_punct_ignored": {c: summarize(st) for c, st in sorted(stats_p.items()) if c != "ALL"},
           "disagreements_punctuation_only": conv_only,
           "disagreements": disagreements}
    o, q = rep["overall"], rep["overall_punct_ignored"]
    print(f"\n== annotator agreement: {o['n']} rows ({'flagged dropped' if args.drop_flagged else 'flagged included'}) ==")
    print(f"{'':28}strict    punct. don't-care")
    print(f"token-level Cohen's kappa   {o['token_kappa']:.3f}     {q['token_kappa']:.3f}   (raw agreement {fmt(o['token_agreement'])} / {fmt(q['token_agreement'])})")
    print(f"message-level Cohen's kappa {o['message_kappa']:.3f}     {q['message_kappa']:.3f}")
    print(f"span F1 (SemEval, macro)    {fmt(o['span_f1_macro'])}    {fmt(q['span_f1_macro'])}")
    print(f"exact match                 {fmt(o['exact_match_rate'])}    {fmt(q['exact_match_rate'])}")
    if disagreements:
        print(f"disagreements: {len(disagreements)} total, {conv_only} punctuation-only (convention, not judgement)")
    w = max((len(c) for c in rep["by_category"]), default=8)
    print(f"\n{'category':<{w}}    n  tok-k   msg-k  spanF1  exact")
    for c, st in rep["by_category"].items():
        tk = f"{st['token_kappa']:.2f}" if st["token_kappa"] is not None else "  -"
        mk = f"{st['message_kappa']:.2f}" if st["message_kappa"] is not None else "  -"
        print(f"{c:<{w}}  {st['n']:>3}  {tk:>5}  {mk:>6}  {fmt(st['span_f1_macro'])}  {fmt(st['exact_match_rate'])}")
    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s)  ([...] = tagged TOXIC):")
        for d in disagreements:
            f = "  (flagged)" if d["flag"] else ""
            c = "  [punctuation only]" if d["punctuation_only"] else ""
            print(f"  row {d['row']} {d['category']}{f}{c}\n    a: {d['a']}\n    b: {d['b']}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(f"\n[agree] report -> {args.report}", file=sys.stderr)


def cmd_score(args):
    gold = load_jsonl(args.gold)
    results = load_jsonl(args.results)
    if not results:
        sys.exit("[error] results file is empty")
    mode = "tagger" if "spans" in results[0] else "system"
    rep = score_tagger(gold, results) if mode == "tagger" else score_system(gold, results)
    print_report(rep)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(f"\n[score] report -> {args.report}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Evaluation harness for the detox system.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="write the orchestrator input file from test.jsonl")
    p.add_argument("--gold", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("score", help="score orchestrator --json or --tag-only output against gold")
    s.add_argument("--gold", required=True)
    s.add_argument("--results", required=True)
    s.add_argument("--report", help="write the full JSON report here")
    s.set_defaults(fn=cmd_score)

    b = sub.add_parser("blind-sheet", help="write a shuffled, header-free labelling sheet from test.jsonl")
    b.add_argument("--gold", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=7)
    b.set_defaults(fn=cmd_blind_sheet)

    g = sub.add_parser("ingest", help="turn a marked-up sheet into test_human.jsonl (fails on any bad line)")
    g.add_argument("--gold", required=True)
    g.add_argument("--sheet", required=True)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_ingest)

    a = sub.add_parser("agree", help="annotator agreement between two label files on the same messages")
    a.add_argument("--a", required=True)
    a.add_argument("--b", required=True)
    a.add_argument("--drop-flagged", action="store_true", help="exclude rows marked uncertain")
    a.add_argument("--report")
    a.set_defaults(fn=cmd_agree)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
