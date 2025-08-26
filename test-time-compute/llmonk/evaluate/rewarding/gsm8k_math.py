import csv
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydra
from lm_eval.tasks.minerva_math.utils import (
    get_unnormalized_answer,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
)
from tqdm import tqdm
from yaml.scanner import ScannerError

from llmonk.evaluate.rewarding import (
    ArmoRMPipeline,
    RMPipeline,
    SkyworkRMPipeline,
    Task,
    get_tasks,
)
from llmonk.utils import EvaluateScriptConfig, load_yaml, save_yaml


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left, f"{s}"
        return s[len(left) :]
    elif "\\boxed{" in s:
        left = "\\boxed{"
        assert s[: len(left)] == left, f"{s}"
        assert s[-1] == "}", f"{s}"
        return s[len(left) : -1]
    elif "\\fbox{" in s:
        left = "\\fbox{"
        assert s[: len(left)] == left, f"{s}"
        assert s[-1] == "}", f"{s}"
        return s[len(left) : -1]
    else:
        # raise ValueError(f"Unknown boxing format: {s}")
        print(f"Unknown boxing format: {s}")
        return "[invalidanswer]"


class ScriptConfig(EvaluateScriptConfig):
    dset = "DEFAULT"
    boxed_math: bool = False
    extract_both: bool = False
    rmp: str = "DEFAULT"

    def finalize(self):
        super().finalize()
        if self.dset == "DEFAULT":
            raise ValueError("Dataset not specified. Please set dset in the config.")
        assert self.dset in ["gsm8k", "math"], "Dataset not supported"
        if self.boxed_math:
            assert self.dset == "math", "Boxed math only supported for math dataset"
        if self.extract_both:
            assert self.dset == "math", "Extract both only supported for math dataset"
        if self.extract_both and self.boxed_math:
            raise ValueError(
                "Cannot extract both and boxed math at the same time. Please set only one."
            )
        if self.rmp == "DEFAULT":
            raise ValueError(
                "RMP model not specified. Please set rmp_model in the config."
            )
        assert self.rmp in ["Armo", "Skywork", "Skywork27B"], "RMP model not supported"


class OutputValidator:
    def __init__(self):
        pass

    def _extract_answer(self, _ans: str) -> str:
        raise NotImplementedError

    def is_correct(
        self,
        sample: str,
        gt_answer: str,
    ) -> Tuple[bool, str, str]:
        gt_answer = self._extract_answer(gt_answer)
        sample = self._extract_answer(sample)
        is_correct = (gt_answer == sample) or is_equiv(sample, gt_answer)
        return is_correct, sample, gt_answer


class GSM8kOutputValidator(OutputValidator):
    def __init__(self):
        super().__init__()
        self.ans_re = re.compile(
            r"(?:####\s*|Answer:?\s*|The answer is\s*)(-?\d+(?:\.\d+)?)", re.IGNORECASE
        )
        self.invalid_ans = "[invalid]"

    def _extract_answer(self, ans: str) -> str:
        match = self.ans_re.search(ans)
        if match:
            return match.group(1)
        else:
            return self.invalid_ans


class MathOutputValidator(OutputValidator):
    def __init__(self, boxed_math: bool, extract_both: bool):
        super().__init__()
        self.boxed_math = boxed_math
        self.extract_both = extract_both

    def _extract_answer_sample(self, ans: str) -> str:
        if self.extract_both:
            # first try to extract unnormalized answer
            unnormalized = get_unnormalized_answer(ans)
            if unnormalized == "[invalidanswer]":
                # if unnormalized answer is invalid, try to extract boxed answer
                last_boxed = last_boxed_only_string(ans)
                if last_boxed is None:
                    return "[invalidanswer]"
                return normalize_final_answer(remove_boxed(last_boxed))
            # if unnormalized answer is valid, return it
            return normalize_final_answer(unnormalized)
        if self.boxed_math:
            last_boxed = last_boxed_only_string(ans)
            if last_boxed is None:
                return "[invalidanswer]"
            return normalize_final_answer(remove_boxed(last_boxed))
        return normalize_final_answer(get_unnormalized_answer(ans))

    def _extract_answer_gt(self, ans: str) -> str:
        last_boxed = last_boxed_only_string(ans)
        assert last_boxed is not None
        return normalize_final_answer(remove_boxed(last_boxed))

    def is_correct(
        self,
        sample: str,
        gt_answer: str,
    ) -> Tuple[bool, str, str]:
        gt_answer = self._extract_answer_gt(gt_answer)
        sample = self._extract_answer_sample(sample)
        is_correct = (gt_answer == sample) or is_equiv(sample, gt_answer)
        return is_correct, sample, gt_answer


def is_correct(
    sample: str,
    gt_answer: str,
    dset: str,
    boxed_math: bool,
    extract_both: bool,
) -> Tuple[bool, str, str]:
    """
    Check if the model's answer is correct against the ground truth answer.
    returns:
        - bool: Whether the model's answer is correct.
        - str: The model's answer.
        - str: The ground truth answer.
    """
    if dset == "gsm8k":
        validator = GSM8kOutputValidator()
    elif dset == "math":
        validator = MathOutputValidator(
            boxed_math=boxed_math, extract_both=extract_both
        )
    else:
        raise ValueError(f"Dataset {dset} not supported")
    return validator.is_correct(sample, gt_answer)


def process_sample(
    task: Task,
    dset: str,
    rmp: RMPipeline,
    boxed_math: bool = False,
    extract_both: bool = False,
) -> Optional[Tuple[List[bool], List[float], List[str]]]:
    """
    Process a single sample and save the results.
        Args:
            task (Task): The task to process.
            dset (str): The dataset name.
        Returns:
            - Tuple[List[bool], List[float], List[str]]: A tuple containing:
                - List[bool]: List of correctness flags for each sample.
                - List[float]: List of rewards for each sample.
                - List[str]: List of model answers for each sample.
    """
    # if task.save_path.exists():
    #     result = load_yaml(task.save_path)
    #     return result["is_corrects"], result["rewards"], result["model_answers"]

    try:
        result = load_yaml(task.sample_path)
    except ScannerError as err:
        print(
            f"YAML error in {task.sample_path}: {err}. Marking all samples as incorrect."
        )
        result = {"samples": [], "gt_answer": "", "is_corrects": []}
        save_yaml(task.save_path, result)
        return None
    corrects: List[bool] = []
    rewards: List[float] = []
    model_answers: List[str] = []
    for sample in result.get("samples", []):
        question = result["question"]
        answer = sample
        reward = rmp(question, answer)
        correct, model_answer, _gt_answer = is_correct(
            sample,
            result["gt_answer"],
            dset,
            boxed_math,
            extract_both,
        )

        corrects.append(correct)
        rewards.append(reward)
        model_answers.append(model_answer)
    result["is_corrects"] = corrects
    result["rewards"] = rewards
    result["model_answers"] = model_answers
    save_yaml(task.save_path, result)
    return corrects, rewards, model_answers


def get_voted_is_corrects(
    corrects: List[bool],
    rewards: List[float],
    model_answers: List[str],
    k: int,
) -> Tuple[List[bool], List[bool], List[bool], List[bool]]:
    coverage = [False] * k
    majority = [False] * k
    most_rewarded = [False] * k
    reward_weighted = [False] * k
    ignore = {"[invalid]", "[invalidanswer]"}
    for i in range(k):
        n_using_samples = 2**i
        # coverage
        coverage[i] = len([tf for tf in corrects[:n_using_samples] if tf]) > 0
        # majority
        counter = Counter(model_answers[:n_using_samples])
        common = counter.most_common()
        filtered = [(val, c) for val, c in common if val not in ignore]
        if len(filtered) == 0:
            # all samples are invalid -> all methods will be False
            continue
        common_ans = filtered[0][0]
        common_ans_idx = [
            j
            for j, ans in enumerate(model_answers[:n_using_samples])
            if ans == common_ans
        ][0]
        majority[i] = corrects[common_ans_idx]
        # most_rewarded
        max_reward = max(
            [
                reward
                for reward, ans in zip(
                    rewards[:n_using_samples], model_answers[:n_using_samples]
                )
                if ans not in ignore
            ]
        )
        max_reward_idx = [
            j
            for j, reward in enumerate(rewards[:n_using_samples])
            if reward == max_reward
        ][0]
        most_rewarded[i] = corrects[max_reward_idx]
        # reward_weighted
        answer_rewards: Dict[str, float] = {}
        for j in range(n_using_samples):
            if model_answers[j] in ignore:
                continue
            answer_rewards[model_answers[j]] = (
                answer_rewards.get(model_answers[j], 0.0) + rewards[j]
            )
        max_rewarded = max(answer_rewards, key=answer_rewards.get)
        max_rewarded_idx = [
            j
            for j, ans in enumerate(model_answers[:n_using_samples])
            if ans == max_rewarded
        ][0]
        reward_weighted[i] = corrects[max_rewarded_idx]

    return coverage, majority, most_rewarded, reward_weighted


@pydra.main(base=ScriptConfig)
def main(config: Optional[ScriptConfig] = None):
    assert config is not None
    tasks = get_tasks(config.samples_dir, config.save_dir)
    tasks = sorted(tasks, key=lambda x: x.save_path)
    tasks = tasks[config.offset : config.limit : config.stride]
    print(f"Evaluating {len(tasks)} tasks")

    n_samples = len(load_yaml(tasks[0].sample_path)["samples"])
    # check if n_samples is 2**(k-1)
    if n_samples & (n_samples - 1) != 0:
        raise ValueError("Number of samples is not a power of 2")
    # extract k
    k = n_samples.bit_length()

    if config.rmp == "Armo":
        rmp = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
    elif config.rmp == "Skywork":
        rmp = SkyworkRMPipeline("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
    elif config.rmp == "Skywork27B":
        rmp = SkyworkRMPipeline(
            "Skywork/Skywork-Reward-Gemma-2-27B-v0.2", device_map="auto"
        )
    coverage_all = np.zeros(k, dtype=float)
    majority_all = np.zeros(k, dtype=float)
    most_rewarded_all = np.zeros(k, dtype=float)
    reward_weighted_all = np.zeros(k, dtype=float)
    tasks_pb = tqdm(tasks, desc="Processing tasks")
    for task in tasks_pb:
        res = process_sample(
            task, config.dset, rmp, config.boxed_math, config.extract_both
        )
        if res is None:
            tasks_pb.write(f"Error processing {task.sample_path}")
            continue
        assert (
            len(res[0]) == n_samples
            and len(res[1]) == n_samples
            and len(res[2]) == n_samples
        )
        corrects, rewards, model_answers = res
        coverage, majority, most_rewarded, reward_weighted = get_voted_is_corrects(
            corrects, rewards, model_answers, k
        )
        coverage_all += np.array(coverage, dtype=float)
        majority_all += np.array(majority, dtype=float)
        most_rewarded_all += np.array(most_rewarded, dtype=float)
        reward_weighted_all += np.array(reward_weighted, dtype=float)
    coverage_all /= len(tasks)
    majority_all /= len(tasks)
    most_rewarded_all /= len(tasks)
    reward_weighted_all /= len(tasks)
    out_file = f"{config.save_dir}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["samples"] + [2**i for i in range(k)]
        writer.writerow(header)
        row = ["coverage"] + coverage_all.tolist()
        writer.writerow(row)
        row = ["majority vote"] + majority_all.tolist()
        writer.writerow(row)
        row = ["most rewarded"] + most_rewarded_all.tolist()
        writer.writerow(row)
        row = ["reward weighted"] + reward_weighted_all.tolist()
        writer.writerow(row)

    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
