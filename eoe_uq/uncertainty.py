"""Aleatoric / epistemic decomposition from an (N outer x M inner) answer-embedding grid."""

from typing import Dict

import numpy as np


def decompose_uncertainty(inner_embeddings: np.ndarray) -> Dict[str, float]:
    """Decompose the predictive variance of answer embeddings.

    Args:
        inner_embeddings: array of shape ``(N, M, d)``. Row ``n`` holds the ``M``
            sampled answers (embedded in ``d`` dims) for the ``n``-th perturbed
            question.

    Returns:
        A dict with

        * ``aleatoric_uncertainty`` (AU): within-perturbation spread,
          ``mean_n mean_d Var_m[e_{n,m,d}]``. How much the model's answers
          vary when the *same* question is asked repeatedly.
        * ``epistemic_uncertainty`` (EU): between-perturbation spread,
          ``mean_d Var_n[mean_m e_{n,m,d}]``. How much the model's *average*
          answer moves when the question is rephrased.
        * ``total_uncertainty``: ``AU + EU`` (law of total variance, averaged
          over embedding dimensions).

    Variances use the unbiased estimator (``ddof=1``), so ``N >= 2`` and
    ``M >= 2`` are required.
    """
    e = np.asarray(inner_embeddings, dtype=np.float64)
    if e.ndim != 3:
        raise ValueError(f"expected shape (N, M, d), got {e.shape}")
    n_outer, n_inner, _ = e.shape
    if n_outer < 2 or n_inner < 2:
        raise ValueError(f"need N >= 2 and M >= 2, got N={n_outer}, M={n_inner}")

    per_outer_var = e.var(axis=1, ddof=1).mean(axis=1)  # (N,)
    aleatoric = float(per_outer_var.mean())

    outer_means = e.mean(axis=1)  # (N, d)
    epistemic = float(outer_means.var(axis=0, ddof=1).mean())

    return {
        "aleatoric_uncertainty": aleatoric,
        "epistemic_uncertainty": epistemic,
        "total_uncertainty": aleatoric + epistemic,
    }
