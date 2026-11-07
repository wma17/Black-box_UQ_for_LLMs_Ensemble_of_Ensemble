"""Ensemble-of-ensembles uncertainty estimation.

For a question x with paraphrase pool P(x):

    outer ensemble (n = 1..N):
        z_n   = embed(x) + eps_n,          eps_n ~ N(0, sigma^2 I)
        x'_n  = argmax_{p in P(x) U {x}} cos(z_n, embed(p))      # snap to a real question
        inner ensemble (m = 1..M):
            y_{n,m} ~ LLM(. | x'_n)                               # sampled answers
            e_{n,m} = embed(y_{n,m})

    AU = mean_n mean_d Var_m[e_{n,m,d}]
    EU = mean_d Var_n[mean_m e_{n,m,d}]
"""

import random
from typing import Dict, List, Optional, Protocol

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from .uncertainty import decompose_uncertainty

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class AnswerGenerator(Protocol):
    def greedy(self, question: str) -> str: ...

    def sample(self, questions: List[str], num_samples: int, temperature: float = ..., batch_size: Optional[int] = ...) -> List[List[str]]: ...


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EnsembleOfEnsembles:
    """Scores the uncertainty of an LLM's answer to a question.

    Args:
        generator: object with ``greedy(question)`` and ``sample(questions, n, temperature, batch_size)``
            (see :class:`eoe_uq.generation.HFChatGenerator`).
        embedder: a ``SentenceTransformer`` used both to pick paraphrases and to embed answers.
        num_outer: N, number of perturbed questions.
        num_inner: M, number of sampled answers per perturbed question.
        noise_level: sigma, std of the Gaussian noise added to the question embedding.
        temperature: sampling temperature for the inner answers.
        gen_batch_size: perturbed questions per generation call (``None`` = all N at once).
    """

    def __init__(
        self,
        generator: AnswerGenerator,
        embedder: Optional[SentenceTransformer] = None,
        num_outer: int = 20,
        num_inner: int = 20,
        noise_level: float = 0.1,
        temperature: float = 0.9,
        gen_batch_size: Optional[int] = 1,
    ):
        self.generator = generator
        self.embedder = embedder or SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        self.num_outer = num_outer
        self.num_inner = num_inner
        self.noise_level = noise_level
        self.temperature = temperature
        self.gen_batch_size = gen_batch_size

    # ------------------------------------------------------------------ #
    # Outer ensemble: embedding-space perturbation snapped to a paraphrase
    # ------------------------------------------------------------------ #
    def perturb(self, question: str, paraphrases: List[str]) -> List[str]:
        """Return N perturbed questions drawn from ``paraphrases`` (plus the original)."""
        candidates = list(dict.fromkeys(list(paraphrases) + [question]))
        cand_emb = self.embedder.encode(candidates, convert_to_tensor=True)
        q_emb = self.embedder.encode(question, convert_to_tensor=True)

        chosen = []
        for _ in range(self.num_outer):
            z = q_emb + torch.randn_like(q_emb) * self.noise_level
            idx = int(torch.argmax(util.cos_sim(z, cand_emb)[0]).item())
            chosen.append(candidates[idx])
        return chosen

    # ------------------------------------------------------------------ #
    # Full pipeline for one question
    # ------------------------------------------------------------------ #
    def score(self, question: str, paraphrases: List[str], return_samples: bool = False) -> Dict:
        """Run the full ensemble of ensembles for one question.

        Returns a dict with the greedy answer, ``aleatoric_uncertainty``,
        ``epistemic_uncertainty`` and ``total_uncertainty``. With
        ``return_samples=True`` it also returns the N perturbed questions and
        the N x M sampled answers.
        """
        answer = self.generator.greedy(question)

        perturbed = self.perturb(question, paraphrases)
        inner_answers = self.generator.sample(
            perturbed, self.num_inner, temperature=self.temperature, batch_size=self.gen_batch_size
        )

        flat = [a for row in inner_answers for a in row]
        emb = np.asarray(self.embedder.encode(flat), dtype=np.float32)
        emb = emb.reshape(self.num_outer, self.num_inner, -1)

        result = {"question": question, "answer": answer, **decompose_uncertainty(emb)}
        if return_samples:
            result["perturbed_questions"] = perturbed
            result["sampled_answers"] = inner_answers
        return result
