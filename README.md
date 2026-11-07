# Black-Box Uncertainty Quantification for Large Language Models via Ensemble-of-Ensembles

Official implementation of the NeurIPS 2026 paper [*Black-Box Uncertainty Quantification for Large Language Models via Ensemble-of-Ensembles*](https://neurips.cc/virtual/2026/loc/sydney/poster/151082).

This is a black-box method for estimating how uncertain a large language model is about its answer to a question. It splits the uncertainty into two parts:

- **Aleatoric (AU):** how much the answers vary when the *same* question is asked repeatedly.
- **Epistemic (EU):** how much the *average* answer shifts when the question is rephrased.

The method only uses the generated text, so it works with any model that can sample answers. It does not need logits or access to model weights.

## How it works

For a question `x`:

1. **Paraphrase pool.** A paraphrasing model (default `eugenesiow/bart-paraphrase`) generates a pool `P(x)` of unique paraphrases (default 500). The original question is added to the pool.
2. **Outer ensemble (N perturbations).** For `n = 1..N`:
   - add Gaussian noise to the question embedding: `z_n = embed(x) + ε_n`, where `ε_n ~ N(0, σ²I)`;
   - snap `z_n` to the closest real question in the pool: `x'_n = argmax_{p ∈ P(x)} cos(z_n, embed(p))`.
3. **Inner ensemble (M samples).** For each `x'_n`, sample `M` answers `y_{n,1..M}` from the LLM and embed each one: `e_{n,m} = embed(y_{n,m})`.
4. **Decomposition** (sample variances with `ddof=1`, averaged over the embedding dimensions `d`):

```
AU    = mean_n  mean_d  Var_m [ e_{n,m,d} ]
EU    = mean_d  Var_n [ mean_m e_{n,m,d} ]  −  AU / M
Total = AU + EU
```

Separately, the model's greedy answer to the original question is generated. This is the answer the uncertainty scores refer to.

```
            ┌── x'_1 ──► y_{1,1} … y_{1,M} ──► mean_1, var_1 ─┐
 x ─► pool ─┼── x'_2 ──► y_{2,1} … y_{2,M} ──► mean_2, var_2 ─┼─► AU = avg(var_n)
            │   …                                             │   EU = var(mean_n) − AU/M
            └── x'_N ──► y_{N,1} … y_{N,M} ──► mean_N, var_N ─┘
```

## Installation

```bash
git clone <this-repo>
cd ensemble-of-ensembles-uq
pip install -r requirements.txt      # or: pip install -e .
```

Gated models such as Llama 3 need `huggingface-cli login` first.

## Quick start: command line

```bash
python scripts/run_eoe.py \
    --input  examples/questions.jsonl \
    --output outputs/results.jsonl \
    --model  Qwen/Qwen2.5-7B-Instruct
```

- `--input` takes `.jsonl` or `.json` with a `"question"` field, or `.txt` with one question per line. Any other fields in a record (e.g. an `id`) are copied to the output unchanged.
- Results are written to `--output` one line at a time. If a run is interrupted, re-running the same command skips questions that are already scored.
- The paraphrase pool is cached at `<output dir>/paraphrase_pool.json`, or at the path given by `--paraphrase-cache`, and reused on later runs.

Each output line looks like this:

```json
{
  "question": "What happens if you crack your knuckles a lot?",
  "answer": "Joint fluid cavitation releases a gas bubble, ...",
  "aleatoric_uncertainty": 0.0011,
  "epistemic_uncertainty": 0.0007,
  "total_uncertainty": 0.0018
}
```

Add `--save-samples` to also store the N perturbed questions and the N×M sampled answers.

## Quick start: Python

```python
from eoe_uq import (EnsembleOfEnsembles, HFChatGenerator,
                    ParaphraseGenerator, build_paraphrase_pool, set_seed)

set_seed(123)
questions = ["What is the capital of Australia?"]

# Step 1: paraphrase pool (cached to disk)
pool = build_paraphrase_pool(questions, ParaphraseGenerator(),
                             num_paraphrases=500, cache_file="paraphrase_pool.json")

# Steps 2–4: perturb → sample → decompose
eoe = EnsembleOfEnsembles(HFChatGenerator("Qwen/Qwen2.5-7B-Instruct"),
                          num_outer=20, num_inner=20, noise_level=0.1, temperature=0.9)
result = eoe.score(questions[0], pool[questions[0]])
print(result["answer"], result["epistemic_uncertainty"], result["aleatoric_uncertainty"])
```

To get AU/EU from answer embeddings you already have, with no models involved:

```python
from eoe_uq import decompose_uncertainty
decompose_uncertainty(embeddings)   # embeddings: array of shape (N, M, d)
```

## Parameters

| Argument | Default | Meaning |
|---|---|---|
| `--num-paraphrases` | 500 | Size of the paraphrase pool per question |
| `--num-outer` | 20 | N: number of perturbed questions |
| `--num-inner` | 20 | M: number of sampled answers per perturbed question |
| `--noise-level` | 0.1 | σ: standard deviation of the noise added to the question embedding |
| `--temperature` | 0.9 | Sampling temperature for the inner answers (top-k 60, top-p 0.99) |
| `--embedding-model` | `all-MiniLM-L6-v2` | Embeds both the questions (for snapping) and the answers |
| `--paraphrase-model` | `eugenesiow/bart-paraphrase` | Paraphrasing model (sampling with top-k 50, temperature 0.9) |
| `--gen-batch-size` | 1 | Perturbed questions per `generate()` call. `0` sends all N at once, which is faster and uses more memory |
| `--seed` | 123 | Random seed for the noise and the sampling |

Each question costs 1 greedy generation plus N×M sampled generations; with the defaults that is 401 generations of up to 50 tokens each.

The answer prompts are in [`eoe_uq/generation.py`](eoe_uq/generation.py). They ask for a short, direct answer.

## Using another LLM backend

`EnsembleOfEnsembles` accepts any object that has these two methods:

```python
class MyGenerator:
    def greedy(self, question: str) -> str: ...
    def sample(self, questions: list[str], num_samples: int,
               temperature: float = 0.9, batch_size=None) -> list[list[str]]: ...
```

Use this to plug in vLLM, an API model, or a different prompt format.

## Repository layout

```
eoe_uq/
  paraphrase.py     # Step 1: paraphrase pool generation and caching
  pipeline.py       # Steps 2–4: perturbation → snapping → inner sampling → scoring
  generation.py     # Hugging Face chat-model wrapper and prompts
  uncertainty.py    # AU / EU decomposition (NumPy only)
scripts/run_eoe.py  # command-line entry point
examples/           # sample input
tests/              # unit tests for the decomposition (no models needed)
```

## Citation

```bibtex
@inproceedings{ma2026bbuq,
  title     = {Black-Box Uncertainty Quantification for Large Language Models via Ensemble-of-Ensembles},
  author    = {Ma, Wang and Bhattacharjya, Debarun and Lee, Junkyu and Pham, Nhan H. and Kokel, Harsha and Ji, Qiang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
