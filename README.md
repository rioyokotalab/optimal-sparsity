# Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks

<p align="center">
  📄 <a href="https://arxiv.org/abs/2508.18672">[Paper]</a> |
  🤗 <a href="https://huggingface.co/collections/llm-jp/optimal-sparsity-math-68a4a5fa635fd1c1628280f1">[Hugging Face Math checkpoints]</a>
  🤗 <a href="https://huggingface.co/collections/llm-jp/optimal-sparsity-math-68a4a5fa635fd1c1628280f1">[Hugging Face Code checkpoints]</a>
  💻 <a href="https://github.com/rioyokotalab/optimal-sparsity">[Code]</a> |
  📊 <a href="(TBA)">[Log]</a> |
</p>

## TODOs

- [ ] Logs

## Quickstart

### Pre-training

Follow the environment setup instructions of [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

All training data used in this work are publicly available. Please refer to the paper for details.
We sincerely thank the contributors who made these datasets publicly accessible.

The pre-training launch scripts for the 65 released checkpoints are under
[`pre-training/scripts/optimal-sparsity-math/`](pre-training/scripts/optimal-sparsity-math/).
Each `optimal-sparsity-math-d{D}-E{E}-k{K}-{TOTAL}-A{ACTIVE}.sh` corresponds 1:1
to the Hugging Face model of the same name (`d` = hidden size, `E` = number of
experts, `k` = top-k, `TOTAL`/`ACTIVE` = total / active parameters).

### Post-training

Follow the environment setup instructions of [volcengine/verl](https://github.com/volcengine/verl)

### Evaluation

- We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

  - For code tasks, we use commit `82a9936`.
  - For other evaluations, we use commit `1044db9`.

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness/
git checkout 1044db9
pip install -e .
pip install vllm
```

You can reproduce the results by running `scripts/lm-evaluation-harness/math_eval.sh`
For code evaluation, use `scripts/lm-evaluation-harness/code_eval.sh`

- For task loss evaluation, please follow the README in the `taskloss-eval` directory:  
  [taskloss-eval/README.md](https://github.com/rioyokotalab/optimal-sparsity/blob/main/taskloss-eval/README.md)

- For test-time compute, please refer to the following script:  
  [evaluate_gsm8k.sh](https://github.com/rioyokotalab/optimal-sparsity/blob/main/test-time-compute/job_scripts/abci/evaluate_gsm8k.sh)

## Acknowledgement

We would like to express our sincere gratitude to the developers and maintainers of the following open-source libraries.
Their contributions and the fact that these codebases are publicly available have been essential for conducting this research.

- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [volcengine/verl](https://github.com/volcengine/verl)
- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [ScalingIntelligence/large_language_monkeys](https://github.com/ScalingIntelligence/large_language_monkeys)

## Citation

```bibtex
@inproceedings{
    nakamura2026optimal,
    title={Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks},
    author={Taishi Nakamura and Satoki Ishikawa and Masaki Kawamura and Takumi Okamoto and Daisuke Nohara and Jun Suzuki and Rio Yokota},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=XFw2EPRUUR}
}
```
