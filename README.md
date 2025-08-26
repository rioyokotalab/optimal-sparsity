# Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks

<p align="center">
  📄 <a href="(TBA)">[Paper]</a> |
  🤗 <a href="https://huggingface.co/collections/llm-jp/optimal-sparsity-math-68a4a5fa635fd1c1628280f1">[Hugging Face]</a>
  💻 <a href="https://github.com/rioyokotalab/optimal-sparsity">[Code]</a> |
  📊 <a href="(TBA)">[Log]</a> |
</p>

## TODOs

- [ ] Logs
- [ ] Training Scripts

## Quickstart

### Pre-training

Follow the environment setup instructions of [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

All training data used in this work are publicly available. Please refer to the paper for details. 
We sincerely thank the contributors who made these datasets publicly accessible.

### Post-training

Follow the environment setup instructions of [volcengine/verl](https://github.com/volcengine/verl)

### Evaluation

- We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

  - For code tasks, we use commit `82a9936`.
  - For other evaluations, we use commit `1044db9`.

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
  nakamura2025optimal,
  title={Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks},
  author={Taishi Nakamura and Satoki Ishikawa and Masaki Kawamura and Takumi Okamoto and Daisuke Nohara and Jun Suzuki and Rio Yokota},
  booktitle={2nd AI for Math Workshop @ ICML 2025},
  year={2025},
  url={https://openreview.net/forum?id=Ewj06opLqW}
}
```
