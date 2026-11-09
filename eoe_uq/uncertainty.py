"""Aleatoric / epistemic decomposition from an (N outer x M inner) answer-embedding grid."""

from typing import Dict

import numpy as np


def decompose_uncertainty(inner_embeddings: np.ndarray) -> Dict[str, float]:
    """Compute AU, EU and total uncertainty.

    Args:
        inner_embeddings: array of shape ``(N, M, d)``. Row ``n`` holds the ``M``
            sampled answers (embedded in ``d`` dims) for the ``n``-th perturbed
            question.

    Returns:
        A dict with

        * ``aleatoric_uncertainty``: ``AU = tr( mean_n Cov_m[e_{n,m}] ) = mean_n sum_d Var_m[e_{n,m,d}]``
        * ``epistemic_uncertainty``: ``EU = tr( Cov_n[mean_m e_{n,m}] ) - AU / M``
        * ``total_uncertainty``: ``AU + EU``

    Variances use ``ddof=1``, so ``N >= 2`` and ``M >= 2`` are required.
    ``- AU / M`` is the finite-M bias correction; it makes EU unbiased, so EU
    can come out slightly negative when the true between-perturbation spread is near zero.
    """
    e = np.asarray(inner_embeddings, dtype=np.float64)
    if e.ndim != 3:
        raise ValueError(f"expected shape (N, M, d), got {e.shape}")
    n_outer, n_inner, _ = e.shape
    if n_outer < 2 or n_inner < 2:
        raise ValueError(f"need N >= 2 and M >= 2, got N={n_outer}, M={n_inner}")

    aleatoric = float(e.var(axis=1, ddof=1).sum(axis=-1).mean())
    epistemic = float(e.mean(axis=1).var(axis=0, ddof=1).sum()) - aleatoric / n_inner

    return {
        "aleatoric_uncertainty": aleatoric,
        "epistemic_uncertainty": epistemic,
        "total_uncertainty": aleatoric + epistemic,
    }
