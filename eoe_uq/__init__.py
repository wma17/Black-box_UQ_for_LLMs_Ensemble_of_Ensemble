"""Black-box Uncertainty Quantification for Large Language Models via Ensemble-of-Ensembles (NeurIPS 2026)."""

import importlib

from .uncertainty import decompose_uncertainty

__version__ = "0.1.0"

# Model-dependent parts (torch / transformers / sentence-transformers) are imported
# on first use, so ``decompose_uncertainty`` and the unit tests only need NumPy.
_LAZY = {
    "EnsembleOfEnsembles": ".pipeline",
    "set_seed": ".pipeline",
    "HFChatGenerator": ".generation",
    "ParaphraseGenerator": ".paraphrase",
    "build_paraphrase_pool": ".paraphrase",
    "load_paraphrase_pool": ".paraphrase",
    "save_paraphrase_pool": ".paraphrase",
}


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["decompose_uncertainty", *_LAZY]
