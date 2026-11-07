"""Answer generation with a chat LLM (Hugging Face transformers)."""

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prompt for the single greedy answer that is being scored.
ORIGINAL_PROMPT = (
    "Answer the following question with only the short, direct answer and nothing else. "
    "If you are unsure, reply with \"I don't know.\" Avoid explanations or extra conversation.\n\n"
    "Question: {question}\nAnswer:"
)

# Prompt for the sampled answers to each perturbed (paraphrased) question.
PERTURBED_PROMPT = (
    "Answer the following question with only the short, direct answer and nothing else. "
    "Do not add any explanation or further conversation.\n\n"
    "Question: {question}\nAnswer:"
)


class HFChatGenerator:
    """Wraps a Hugging Face chat model.

    Any object with the same two methods (``greedy`` and ``sample``) can be
    passed to :class:`eoe_uq.EnsembleOfEnsembles` instead, e.g. a wrapper
    around vLLM or an API model: the algorithm only needs answer text, not logits.
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 50,
        top_k: int = 60,
        top_p: float = 0.99,
        original_prompt: str = ORIGINAL_PROMPT,
        perturbed_prompt: str = PERTURBED_PROMPT,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=device_map
        ).eval()
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.original_prompt = original_prompt
        self.perturbed_prompt = perturbed_prompt
        self.eos_token_ids = self._terminators()

    def _terminators(self) -> List[int]:
        ids = []
        for tok in (self.tokenizer.eos_token, "<|eot_id|>", "<|im_end|>"):
            if tok is None:
                continue
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != self.tokenizer.unk_token_id and tid not in ids:
                ids.append(int(tid))
        return ids

    def _chat(self, template: str, question: str) -> str:
        messages = [{"role": "user", "content": template.format(question=question)}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def _generate(self, prompts: List[str], num_return_sequences: int, **kwargs) -> List[List[str]]:
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_ids,
            **kwargs,
        )
        texts = self.tokenizer.batch_decode(
            out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )
        texts = [t.strip() for t in texts]
        k = num_return_sequences
        return [texts[i * k:(i + 1) * k] for i in range(len(prompts))]

    def greedy(self, question: str) -> str:
        """Deterministic answer to the original question (the answer being scored)."""
        prompt = self._chat(self.original_prompt, question)
        return self._generate([prompt], 1, do_sample=False, num_beams=1)[0][0]

    def sample(
        self,
        questions: List[str],
        num_samples: int,
        temperature: float = 0.9,
        batch_size: Optional[int] = 1,
    ) -> List[List[str]]:
        """Draw ``num_samples`` answers for each question -> ``[len(questions)][num_samples]``.

        ``batch_size`` is the number of questions per ``generate`` call
        (``None`` = all at once). Larger is faster but needs more memory.
        """
        prompts = [self._chat(self.perturbed_prompt, q) for q in questions]
        step = batch_size or len(prompts)
        answers: List[List[str]] = []
        for i in range(0, len(prompts), step):
            answers.extend(
                self._generate(
                    prompts[i:i + step],
                    num_samples,
                    do_sample=True,
                    temperature=temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )
            )
        return answers
