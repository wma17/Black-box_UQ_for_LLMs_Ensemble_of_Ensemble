# Black-Box Uncertainty Quantification for LLMs via Ensemble-of-Ensembles

Official code for the NeurIPS 2026 paper [*Black-Box Uncertainty Quantification for Large Language Models via Ensemble-of-Ensembles*](https://neurips.cc/virtual/2026/loc/sydney/poster/151082).

The method estimates how uncertain an LLM is about its answer and splits the uncertainty into two parts:

- **Aleatoric (AU):** how much the answers vary when the same question is asked repeatedly.
- **Epistemic (EU):** how much the average answer shifts when the question is rephrased.

It only needs generated text, so it works with any model you can sample from (no logits or weights).

## How it works

For a question `x`:

1. **Paraphrase pool.** A paraphrasing model generates a pool `P(x)` of unique paraphrases (default 500), built once and cached.
2. **Outer ensemble.** `N` times: add Gaussian noise `N(0, σ²I)` to the embedding of `x`, then pick the paraphrase in `P(x)` with the highest cosine similarity to the noisy embedding.
3. **Inner ensemble.** For each of the `N` paraphrases, sample `M` answers from the LLM and embed them (L2-normalized), giving `e_{n,m}`.
4. **Decomposition.** With sample covariances and the trace as the scalar summary:

```
AU = tr( mean_n Cov_m[e_{n,m}] )
EU = tr( Cov_n[mean_m e_{n,m}] ) − AU / M
TU = AU + EU
```

The `− AU/M` term is the finite-`M` bias correction from the paper. Because it makes EU unbiased, EU can be slightly negative when the answers barely change across paraphrases.

The scores refer to the model's greedy answer to the original question, which is returned alongside them.

## Installation

```bash
git clone https://github.com/wma17/Black-box_UQ_for_LLMs_Ensemble_of_Ensemble.git
cd Black-box_UQ_for_LLMs_Ensemble_of_Ensemble
pip install -e .
```

Gated models such as Llama 3 need `huggingface-cli login` first.

## Usage

**Command line**

```bash
python scripts/run_eoe.py \
    --input  examples/questions.jsonl \
    --output outputs/results.jsonl \
    --model  Qwen/Qwen2.5-7B-Instruct
```

The input is `.jsonl`/`.json` with a `"question"` field (other fields such as `id` are copied to the output), or `.txt` with one question per line. Each output line has the form

```
{"question": ..., "answer": ..., "aleatoric_uncertainty": ..., "epistemic_uncertainty": ..., "total_uncertainty": ...}
```

Results are written as they finish, so re-running an interrupted command skips questions already scored. The paraphrase pool is cached in `<output dir>/paraphrase_pool.json` (or `--paraphrase-cache`). Add `--save-samples` to also keep the perturbed questions and sampled answers.

**Python**

```python
from eoe_uq import EnsembleOfEnsembles, HFChatGenerator, ParaphraseGenerator, build_paraphrase_pool, set_seed

set_seed(123)
questions = ["What is the capital of Australia?"]
pool = build_paraphrase_pool(questions, ParaphraseGenerator(), num_paraphrases=500,
                             cache_file="paraphrase_pool.json")

eoe = EnsembleOfEnsembles(HFChatGenerator("Qwen/Qwen2.5-7B-Instruct"), num_outer=20, num_inner=20)
result = eoe.score(questions[0], pool[questions[0]])
```

If you already have answer embeddings of shape `(N, M, d)`, `eoe_uq.decompose_uncertainty(embeddings)` computes AU/EU/TU directly.

To use vLLM, an API model or your own prompts, pass any object with `greedy(question) -> str` and `sample(questions, num_samples, temperature, batch_size) -> list[list[str]]` in place of `HFChatGenerator` (the default prompts are in [`eoe_uq/generation.py`](eoe_uq/generation.py)).

## Main options

| Option | Default | Meaning |
|---|---|---|
| `--num-outer` | 20 | `N`, perturbed questions per question |
| `--num-inner` | 20 | `M`, sampled answers per perturbed question |
| `--noise-level` | 0.1 | `σ`, noise scale in embedding space |
| `--num-paraphrases` | 500 | Paraphrase pool size |
| `--temperature` | 0.9 | Sampling temperature (top-k 50, top-p 0.99, max 50 new tokens) |
| `--embedding-model` | `all-MiniLM-L6-v2` | Encoder for questions and answers |
| `--paraphrase-model` | `eugenesiow/bart-paraphrase` | Paraphrasing model |
| `--gen-batch-size` | 1 | Perturbed questions per `generate()` call (`0` = all `N` at once) |
| `--seed` | 123 | Random seed |

Each question costs `N×M + 1` generations (401 with the defaults). Smaller budgets such as `N = M = 10` work well in practice (see the paper's budget ablation).

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

[MIT](LICENSE)
