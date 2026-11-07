"""Step 1: build a pool of paraphrases for each question."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_PARAPHRASE_MODEL = "eugenesiow/bart-paraphrase"


class ParaphraseGenerator:
    """Samples paraphrases from a seq2seq paraphrasing model.

    Defaults match the paper: ``eugenesiow/bart-paraphrase`` with
    ``do_sample=True, top_k=50, temperature=0.9`` and up to 50 new tokens,
    drawn in batches of 50 until the pool holds ``num_paraphrases`` unique strings.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_PARAPHRASE_MODEL,
        device: Optional[str] = None,
        top_k: int = 50,
        temperature: float = 0.9,
        max_new_tokens: int = 50,
        batch_size: int = 50,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
        self.top_k = top_k
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    @torch.inference_mode()
    def _sample(self, question: str, n: int) -> List[str]:
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            do_sample=True,
            top_k=self.top_k,
            temperature=self.temperature,
            num_beams=1,
            num_return_sequences=n,
            max_new_tokens=self.max_new_tokens,
        )
        texts = self.tokenizer.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return [t.strip() for t in texts if t.strip()]

    def generate(self, question: str, num_paraphrases: int = 500, max_rounds: int = 200) -> List[str]:
        """Return up to ``num_paraphrases`` unique paraphrases of ``question``.

        Stops after ``max_rounds`` sampling rounds even if the pool is not full
        (short questions can have fewer distinct paraphrases than requested).
        """
        pool: set = set()
        for _ in range(max_rounds):
            needed = num_paraphrases - len(pool)
            if needed <= 0:
                break
            pool.update(self._sample(question, min(needed, self.batch_size)))
        if len(pool) < num_paraphrases:
            logger.warning(
                "Only %d/%d unique paraphrases after %d rounds for: %r",
                len(pool), num_paraphrases, max_rounds, question,
            )
        return list(pool)[:num_paraphrases]


def build_paraphrase_pool(
    questions: Iterable[str],
    generator: ParaphraseGenerator,
    num_paraphrases: int = 500,
    cache_file: Optional[Union[str, Path]] = None,
) -> Dict[str, List[str]]:
    """Build ``{question: [paraphrase, ...]}`` for every unique question.

    If ``cache_file`` is given, existing entries are reused, only missing
    questions are generated, and the merged pool is written back (JSON).
    """
    pool: Dict[str, List[str]] = {}
    if cache_file is not None and Path(cache_file).exists():
        pool = load_paraphrase_pool(cache_file)
        logger.info("Loaded %d cached paraphrase sets from %s", len(pool), cache_file)

    missing = [q for q in dict.fromkeys(questions) if q not in pool]
    for q in tqdm(missing, desc="Paraphrase pool"):
        pool[q] = generator.generate(q, num_paraphrases)

    if cache_file is not None and missing:
        save_paraphrase_pool(pool, cache_file)
    return pool


def load_paraphrase_pool(path: Union[str, Path]) -> Dict[str, List[str]]:
    with open(path) as f:
        return json.load(f)


def save_paraphrase_pool(pool: Dict[str, List[str]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)
