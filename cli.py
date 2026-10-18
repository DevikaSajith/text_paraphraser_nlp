#!/usr/bin/env python
"""Command-line interface for the text paraphraser.

Examples
--------
  python cli.py paraphrase --text "The quick brown fox jumps over the lazy dog." --method rule -n 3
  python cli.py paraphrase --file essay.txt --method neural --strategy diverse -n 3
  python cli.py eval --input data/sample.csv --method rule --output results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys

from paraphraser import ParaphraseEngine, SemanticScorer, evaluate_dataset
from paraphraser.evaluate import load_csv


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--method", choices=["rule", "neural", "hybrid"], default="hybrid")
    p.add_argument("--model", default=None, help="HuggingFace seq2seq paraphrase model name")
    p.add_argument("--strategy", choices=["beam", "diverse", "sample"], default="diverse",
                   help="decoding strategy for the neural model")
    p.add_argument("--similarity-backend", choices=["auto", "embedding", "bow"], default="auto")
    p.add_argument("--seed", type=int, default=None)


def _engine(args) -> ParaphraseEngine:
    return ParaphraseEngine(neural_model=args.model, seed=args.seed,
                            scorer_backend=args.similarity_backend)


def cmd_paraphrase(args) -> int:
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    if not text.strip():
        print("No input text given.", file=sys.stderr)
        return 1
    engine = _engine(args)
    results = engine.paraphrase(text, method=args.method, n=args.n, strategy=args.strategy)
    if args.json:
        print(json.dumps([c.__dict__ for c in results], indent=2, ensure_ascii=False))
        return 0
    for i, c in enumerate(results, 1):
        flag = "" if c.accepted else "  [low similarity]"
        print(f"[{i}] ({c.method}, sim={c.similarity:.2f}, pinc={c.pinc:.2f}){flag}")
        print(c.text, end="\n\n")
    return 0


def cmd_eval(args) -> int:
    data = load_csv(args.input)
    sources, refs = data["sources"], data["references"]
    engine = _engine(args)
    outputs = []
    for s in sources:
        best = engine.paraphrase(s, method=args.method, n=1, strategy=args.strategy)
        outputs.append(best[0].text if best else s)
    report = evaluate_dataset(sources, outputs, refs if any(refs) else None, engine.scorer)
    print("\n== Summary ==")
    for k, v in report["summary"].items():
        print(f"{k:>18}: {v:.4f}" if isinstance(v, float) else f"{k:>18}: {v}")
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            cols = ["source", "paraphrase"] + list(report["per_example"][0])
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for s, o, row in zip(sources, outputs, report["per_example"]):
                w.writerow({"source": s, "paraphrase": o, **row})
        print(f"\nPer-example results written to {args.output}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Automatic text paraphraser")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("paraphrase", help="paraphrase a text")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text")
    src.add_argument("--file")
    p.add_argument("-n", type=int, default=3, help="number of paraphrases")
    p.add_argument("--json", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_paraphrase)

    e = sub.add_parser("eval", help="evaluate on a CSV (columns: source[,reference])")
    e.add_argument("--input", required=True)
    e.add_argument("--output")
    _add_common(e)
    e.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
