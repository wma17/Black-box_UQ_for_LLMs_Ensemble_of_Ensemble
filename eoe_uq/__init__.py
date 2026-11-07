"""Ensemble-of-ensembles uncertainty quantification for LLM question answering."""

from .generation import HFChatGenerator
from .paraphrase import (
    ParaphraseGenerator,
    build_paraphrase_pool,
    load_paraphrase_pool,
    save_paraphrase_pool,
)
from .pipeline import EnsembleOfEnsembles, set_seed
from .uncertainty import decompose_uncertainty

__version__ = "0.1.0"

__all__ = [
    "EnsembleOfEnsembles",
    "HFChatGenerator",
    "ParaphraseGenerator",
    "build_paraphrase_pool",
    "decompose_uncertainty",
    "load_paraphrase_pool",
    "save_paraphrase_pool",
    "set_seed",
]
