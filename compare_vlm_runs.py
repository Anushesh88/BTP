"""
Compare Phase 9 VLM runs on the paintings they all share (e.g. qwen2.5vl:3b vs :7b vs :7b blind).

  python compare_vlm_runs.py reports/vlm_eval_cases.json reports/vlm_7b_sample/vlm_eval_cases.json \
      reports/vlm_7b_blind_sample/vlm_eval_cases.json --notes reports/vlm_comparison_notes.md
"""

import argparse
import json
from pathlib import Path

TARGETS = ["raga", "school", "period"]


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d.get("vlm_model", Path(path).parent.name), {c["image_id"]: c for c in d["cases"]}


def vlm_pred(c, t):
    return (c["vlm_judgment"].get(f"predicted_{t}") or {}).get("value") or "uncertain"


def clf_pred(c, t):
    return c["classifier_top3"][t][0][0]


def score(cases, ids, get_pred):
    out = {}
    for t in TARGETS:
        n = correct = answered = correct_answered = agree = 0
        for i in ids:
            gt = cases[i]["ground_truth"][t]
            if gt is None:
                continue
            pred = get_pred(cases[i], t)
            n += 1
            correct += pred == gt
            agree += pred == clf_pred(cases[i], t)
            if pred != "uncertain":
                answered += 1
                correct_answered += pred == gt
        out[t] = {
            "n": n,
            "acc": correct / n if n else None,
            "abstain": 1 - answered / n if n else None,
            "selective": correct_answered / answered if answered else None,
            "agree": agree / n if n else None,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="vlm_eval_cases.json files")
    p.add_argument("--out", default="reports/vlm_model_comparison.md")
    p.add_argument("--notes", default=None, help="Markdown file appended as the interpretation section")
    args = p.parse_args()

    runs = [load(r) for r in args.runs]
    ids = [i for i in runs[-1][1] if all(i in cases for _, cases in runs)]
    ref = runs[0][1]
    pct = lambda v: f"{v*100:.0f}%" if v is not None else "N/A"

    md = [f"# VLM Model Comparison ({len(ids)} shared held-out paintings)", "",
          "| Model | Head | N | Accuracy | Abstained | Accuracy when answered | Agrees with classifier top-1 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    s = score(ref, ids, clf_pred)
    for t in TARGETS:
        md.append(f"| Classifier (calibrated multi-head) | {t} | {s[t]['n']} | {pct(s[t]['acc'])} | - | - | - |")
    for name, cases in runs:
        s = score(cases, ids, vlm_pred)
        for t in TARGETS:
            r = s[t]
            md.append(f"| {name} | {t} | {r['n']} | {pct(r['acc'])} | {pct(r['abstain'])} | {pct(r['selective'])} | {pct(r['agree'])} |")

    md += ["", "| Model | Routed to expert review |", "| --- | --- |"]
    for name, cases in runs:
        rate = sum(cases[i]["vlm_judgment"]["cross_check"]["route_to_expert_review"] for i in ids) / len(ids)
        md.append(f"| {name} | {pct(rate)} |")

    md += ["", "## Raga and school answers on the non-Mughal / Ragamala paintings", "",
           "| Painting | Ground truth (raga / school) | Classifier | " + " | ".join(n for n, _ in runs) + " |",
           "| --- | --- | --- | " + " | ".join("---" for _ in runs) + " |"]
    for i in ids:
        g = ref[i]["ground_truth"]
        if g["raga"] != "Non-Ragamala" or g["school"] != "Mughal":
            cells = [f"{vlm_pred(c[i], 'raga')} / {vlm_pred(c[i], 'school')}" for _, c in runs]
            md.append(f"| {i[:40]} | {g['raga']} / {g['school']} | {clf_pred(ref[i], 'raga')} / {clf_pred(ref[i], 'school')} | " + " | ".join(cells) + " |")

    nr = [i for i in ids if ref[i]["ground_truth"]["raga"] == "Non-Ragamala"]
    md += ["", "Non-Ragamala paintings correctly called 'Non-Ragamala': "
           + ", ".join(f"{n} {sum(vlm_pred(c[i], 'raga') == 'Non-Ragamala' for i in nr)}/{len(nr)}" for n, c in runs) + "."]

    if args.notes and Path(args.notes).exists():
        md += ["", Path(args.notes).read_text(encoding="utf-8").strip()]

    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
