#!/usr/bin/env python3
"""Score questions with ensemble-of-ensembles uncertainty.

Input  (--input):  .jsonl / .json with a "question" field per record, or .txt with one question per line.
                   Any other fields in a record (e.g. an "id") are copied to the output unchanged.
Output (--output): .jsonl, one record per question, written as it goes. Re-running with the
                   same output skips questions that are already done.

Example:
    python scripts/run_eoe.py \
        --input examples/questions.jsonl \
        --output outputs/results.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eoe_uq import (  # noqa: E402
    EnsembleOfEnsembles,
    HFChatGenerator,
    ParaphraseGenerator,
    build_paraphrase_pool,
    set_seed,
)
from eoe_uq.paraphrase import DEFAULT_PARAPHRASE_MODEL  # noqa: E402
from eoe_uq.pipeline import DEFAULT_EMBEDDING_MODEL  # noqa: E402


def read_records(path: Path) -> List[Dict]:
    if path.suffix == ".txt":
        return [{"question": line.strip()} for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    return [{"question": r} if isinstance(r, str) else r for r in data]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True, help="Hugging Face chat model id or local path.")
    p.add_argument("--paraphrase-cache", type=Path, default=None,
                   help="JSON {question: [paraphrases]}. Default: <output dir>/paraphrase_pool.json")
    p.add_argument("--paraphrase-model", default=DEFAULT_PARAPHRASE_MODEL)
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--num-paraphrases", type=int, default=500, help="Paraphrase pool size per question.")
    p.add_argument("--num-outer", type=int, default=20, help="N: perturbed questions per question.")
    p.add_argument("--num-inner", type=int, default=20, help="M: sampled answers per perturbed question.")
    p.add_argument("--noise-level", type=float, default=0.1, help="sigma of the embedding-space noise.")
    p.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature for inner answers.")
    p.add_argument("--gen-batch-size", type=int, default=1,
                   help="Perturbed questions per generate() call; 0 = all N at once.")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--save-samples", action="store_true",
                   help="Also store perturbed questions and sampled answers in the output.")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    records = read_records(args.input)[: args.max_examples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.paraphrase_cache or args.output.parent / "paraphrase_pool.json"

    done = set()
    if args.output.exists():
        done = {json.loads(l)["question"] for l in args.output.read_text().splitlines() if l.strip()}
    todo = [r for r in records if r["question"] not in done]
    logging.info("%d questions, %d already scored, %d to do", len(records), len(done), len(todo))
    if not todo:
        return

    # Step 1: paraphrase pool (cached; the paraphraser is freed before the LLM is loaded).
    pool = {}
    if cache.exists():
        pool = json.loads(cache.read_text())
    if any(r["question"] not in pool for r in todo):
        paraphraser = ParaphraseGenerator(args.paraphrase_model)
        pool = build_paraphrase_pool(
            [r["question"] for r in todo], paraphraser, args.num_paraphrases, cache_file=cache
        )
        del paraphraser
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Steps 2-4: perturb -> sample -> decompose.
    eoe = EnsembleOfEnsembles(
        generator=HFChatGenerator(args.model),
        embedder=SentenceTransformer(args.embedding_model),
        num_outer=args.num_outer,
        num_inner=args.num_inner,
        noise_level=args.noise_level,
        temperature=args.temperature,
        gen_batch_size=args.gen_batch_size or None,
    )

    with open(args.output, "a") as f:
        for rec in tqdm(todo, desc="Scoring"):
            q = rec["question"]
            out = eoe.score(q, pool[q], return_samples=args.save_samples)
            f.write(json.dumps({**rec, **out}, ensure_ascii=False) + "\n")
            f.flush()

    logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
