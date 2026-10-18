"""Neural paraphraser built on a pretrained T5 (seq2seq) paraphrase model.

The model is loaded lazily on first use so importing the package stays cheap.
Any HuggingFace seq2seq paraphrase checkpoint that uses a "paraphrase: <text>"
style prompt works (T5, and with prefix="" also Pegasus/BART style models).
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .utils import dedupe_keep_order, split_sentences

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"
STRATEGIES = ("beam", "diverse", "sample")


class NeuralParaphraser:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        prefix: str = "paraphrase: ",
        max_length: int = 128,
    ):
        self.model_name = model_name
        self.device = device
        self.prefix = prefix
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._diverse_ok = True  # flips to False once group beam search is found unavailable

    # ----------------------------------------------------------------- loading
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    # -------------------------------------------------------------- generation
    def _generate(self, sentence: str, n: int, strategy: str, temperature: float,
                  top_p: float, diversity_penalty: float) -> List[str]:
        import torch

        enc = self._tokenizer(
            self.prefix + sentence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        max_new = min(self.max_length, int(enc["input_ids"].shape[1] * 1.6) + 8)
        base = dict(max_new_tokens=max_new, no_repeat_ngram_size=3, num_return_sequences=n)

        def run(**kw):
            with torch.no_grad():
                out = self._model.generate(**enc, **base, **kw)
            return self._tokenizer.batch_decode(
                out, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

        if strategy == "beam" or (strategy == "diverse" and n == 1):
            return run(num_beams=max(4, n * 2), early_stopping=True)
        if strategy == "diverse" and self._diverse_ok:
            try:
                return run(num_beams=n * 2, num_beam_groups=n, diversity_penalty=diversity_penalty)
            except Exception as exc:
                # Newer transformers moved group beam search to a remote add-on that needs
                # trust_remote_code; we deliberately do not enable that.
                self._diverse_ok = False
                logger.warning(
                    "Diverse beam search is unavailable in this transformers version (%s). "
                    "Using nucleus sampling instead.", str(exc).split("(")[0].strip()
                )
        return run(do_sample=True, top_k=50, top_p=top_p, temperature=temperature)

    def paraphrase_sentence(self, sentence: str, n: int = 3, strategy: str = "diverse",
                            temperature: float = 1.0, top_p: float = 0.95,
                            diversity_penalty: float = 1.5) -> List[str]:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self._load()
        outs = self._generate(sentence, n, strategy, temperature, top_p, diversity_penalty)
        outs = [o for o in dedupe_keep_order(outs) if o.strip().lower() != sentence.strip().lower()]
        return outs

    def paraphrase(self, text: str, n: int = 3, strategy: str = "diverse",
                   temperature: float = 1.0, top_p: float = 0.95,
                   diversity_penalty: float = 1.5, **_unused) -> List[str]:
        """Paraphrase a (possibly multi-sentence) text, returning up to `n` variants.

        Sentences are paraphrased independently; variant k stitches together the
        k-th candidate of each sentence (falling back to the best one / original).
        """
        sentences = split_sentences(text)
        if not sentences:
            return []
        per_sentence = []
        for s in sentences:
            cands = self.paraphrase_sentence(s, n, strategy, temperature, top_p, diversity_penalty)
            per_sentence.append(cands or [s])
        variants = []
        for k in range(n):
            parts = [c[k] if k < len(c) else c[0] for c in per_sentence]
            variants.append(" ".join(parts))
        return dedupe_keep_order(variants)