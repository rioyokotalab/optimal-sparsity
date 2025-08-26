from __future__ import annotations
import math
from typing import List, Tuple

import numpy as np
from vllm import LLM, SamplingParams


class TaskLoss:
    """
    Universal task-loss calculator (bits/token & bits/byte).

    Responsibilities
    -----------------
    • Instantiates an `LLM(model=...)` (optionally with tensor parallelism).
    • Runs inference on a batch of (prompt, answer) pairs.
    • Computes negative-log-likelihood → converts to bits/token and bits/byte.
    """

    def __init__(
        self,
        model: str,
        *,
        trust_remote_code: bool = True,
        tensor_parallel_size: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        model : str
            Hugging Face model ID or local checkpoint path.
        trust_remote_code : bool, default=True
            Pass-through flag for vLLM.
        tensor_parallel_size : int | None, default=None
            If set, enables tensor parallelism with the given degree
            (`LLM(..., tensor_parallel_size=...)`).  Leave `None`
            (or 1) for single-GPU evaluation.
        """
        self.llm = LLM(
            model=model,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.tok = self.llm.get_tokenizer()

    def eval_batch(
        self, qa_pairs: List[Tuple[str, str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """[(prompt, answer), ...] → (bits_tok_arr, bits_byte_arr)"""
        prompts_full = [q + a for q, a in qa_pairs]

        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
            stop_token_ids=[self.tok.eos_token_id],
        )
        outs = self.llm.generate(prompts_full, params)

        bits_tok, bits_byte = [], []
        for (prompt, answer), out in zip(qa_pairs, outs):
            q_len, t_len, b_len = self._lengths(prompt, answer)
            nll = self._nll_from_output(out, q_len)
            bits_tok.append(nll / math.log(2) / t_len)
            bits_byte.append(nll / math.log(2) / b_len)

        return np.array(bits_tok), np.array(bits_byte)

    def _lengths(self, prompt: str, answer: str) -> Tuple[int, int, int]:
        enc_prompt = self.tok.encode(prompt, add_special_tokens=True)
        enc_full = self.tok.encode(prompt + answer, add_special_tokens=True)
        q_len = len(enc_prompt)
        t_len = len(enc_full) - q_len
        b_len = len(answer.encode())
        return q_len, t_len, b_len

    def _nll_from_output(self, out, q_len: int) -> float:
        lp_dicts = out.prompt_logprobs[q_len:]
        toks = out.prompt_token_ids[q_len:]
        return -sum(float(d[t].logprob) for d, t in zip(lp_dicts, toks))
