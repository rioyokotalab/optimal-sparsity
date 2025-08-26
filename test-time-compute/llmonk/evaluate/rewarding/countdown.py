import csv
import re
from collections import Counter
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydra
from tqdm import tqdm

from llmonk.evaluate.rewarding import ArmoRMPipeline, Task, get_tasks
from llmonk.utils import EvaluateScriptConfig, load_yaml, save_yaml

PROMPT_FORMAT = "Using the numbers {available_numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
FORMAT_SCORE = 0.1
CORRECT_SCORE = 1.0


class EvalEquationResultType(Enum):
    CANNOT_EXTRACT = 0
    USE_UNAVAILABLE_NUMBERS = 1
    CANNOT_EVALUATE = 2
    INCORRECT = 3
    CORRECT = 4


class EvalEquationResult:
    def __init__(self, type: EvalEquationResultType, value: Optional[int] = None):
        self.type = type
        self.value = value

    @classmethod
    def incorrect(cls, wrong_value: int):
        return cls(EvalEquationResultType.INCORRECT, wrong_value)

    @classmethod
    def correct(cls, correct_value: int):
        return cls(EvalEquationResultType.CORRECT, correct_value)

    @classmethod
    def cannot_extract(cls):
        return cls(EvalEquationResultType.CANNOT_EXTRACT)

    @classmethod
    def use_unavailable_numbers(cls):
        return cls(EvalEquationResultType.USE_UNAVAILABLE_NUMBERS)

    @classmethod
    def cannot_evaluate(cls):
        return cls(EvalEquationResultType.CANNOT_EVALUATE)


def extract_equation(sample: str) -> Optional[str]:
    answer_pattern = r"<answer>(.*?)</answer>"
    # matches = re.findall(answer_pattern, sample, re.DOTALL)
    matches = list(re.finditer(answer_pattern, sample, re.DOTALL))
    if len(matches) == 0:
        return None
    final_answer: str = matches[-1].group(1).strip()
    if "=" in final_answer:
        final_answer = final_answer.split("=", 1)[0].strip()
    return final_answer


def validate_equation(equation_str: str, available_numbers: List[int]) -> bool:
    try:
        nums = sorted([int(n) for n in re.findall(r"\d+", equation_str)])
        return nums == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str: str) -> Optional[int]:
    try:
        if not re.match(r"^[\d+\-*/().\s]+$", equation_str):
            return None
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


def get_score(
    sample: str,
    gt_target: int,
    gt_available_numbers: List[int],
) -> Tuple[float, EvalEquationResult]:
    equation = extract_equation(sample)
    if equation is None:
        return 0.0, EvalEquationResult.cannot_extract()
    if not validate_equation(equation, gt_available_numbers):
        return FORMAT_SCORE, EvalEquationResult.use_unavailable_numbers()
    res = evaluate_equation(equation)
    if res is None:
        return FORMAT_SCORE, EvalEquationResult.cannot_evaluate()
    if abs(res - gt_target) < 1e-5:
        return CORRECT_SCORE, EvalEquationResult.correct(int(res))
    return FORMAT_SCORE, EvalEquationResult.incorrect(int(res))


def process_sample(
    task: Task,
    armo_rmp: ArmoRMPipeline,
) -> Tuple[List[float], List[float], List[EvalEquationResult]]:
    result = load_yaml(task.sample_path)
    scores: List[float] = []
    rewards: List[float] = []
    ee_results: List[EvalEquationResult] = []
    gt_target: int = result["target"]
    gt_available_numbers: List[int] = result["numbers"]
    prompt = PROMPT_FORMAT.format(
        available_numbers=gt_available_numbers, target=gt_target
    )
    for sample in result.get("samples", []):
        reward = armo_rmp(prompt, sample)
        score, ee_result = get_score(sample, gt_target, gt_available_numbers)
        rewards.append(reward)
        scores.append(score)
        ee_results.append(ee_result)
    result["scores"] = scores
    result["rewards"] = rewards
    save_yaml(task.save_path, result)
    return scores, rewards, ee_results


def get_voted_is_corrects(
    scores: List[float],
    rewards: List[float],
    ee_results: List[EvalEquationResult],
    k: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    coverage = [0.0] * k
    majority = [0.0] * k
    most_rewarded = [0.0] * k
    reward_weighted = [0.0] * k
    ignore_result_type = {
        EvalEquationResultType.CANNOT_EXTRACT,
        EvalEquationResultType.USE_UNAVAILABLE_NUMBERS,
        EvalEquationResultType.CANNOT_EVALUATE,
    }
    for i in range(k):
        n_using_samples = 2**i
        if (
            len(
                [
                    res
                    for res in ee_results[:n_using_samples]
                    if res.type not in ignore_result_type
                ]
            )
            == 0
        ):
            # all result shoud be ignored -> all scores[i] = 0.0
            continue
        # coverage
        coverage[i] = max(
            [
                score
                for score, res in zip(
                    scores[:n_using_samples], ee_results[:n_using_samples]
                )
                if res.type not in ignore_result_type
            ]
        )
        # majority
        counter = Counter(
            [
                res.value
                for res in ee_results[:n_using_samples]
                if res.type not in ignore_result_type
            ]
        )
        common = counter.most_common()
        common_ans = common[0][0]
        common_idx = [res.value for res in ee_results[:n_using_samples]].index(
            common_ans
        )
        majority[i] = (
            CORRECT_SCORE
            if ee_results[common_idx].type == EvalEquationResultType.CORRECT
            else FORMAT_SCORE
        )
        # most_rewarded
        max_reward = max(
            [
                reward
                for reward, res in zip(
                    rewards[:n_using_samples], ee_results[:n_using_samples]
                )
                if res.type not in ignore_result_type
            ]
        )
        max_reward_idx = rewards[:n_using_samples].index(max_reward)
        most_rewarded[i] = (
            CORRECT_SCORE
            if ee_results[max_reward_idx].type == EvalEquationResultType.CORRECT
            else FORMAT_SCORE
        )

        # reward_weighted
        answer_rewards: Dict[int, float] = {}
        for j in range(n_using_samples):
            if ee_results[j].type in ignore_result_type:
                continue
            v = ee_results[j].value
            assert v is not None
            answer_rewards[v] = answer_rewards.get(v, 0.0) + rewards[j]
        max_rewarded = max(answer_rewards, key=answer_rewards.get)
        max_rewarded_idx = [res.value for res in ee_results[:n_using_samples]].index(
            max_rewarded
        )
        reward_weighted[i] = (
            CORRECT_SCORE
            if ee_results[max_rewarded_idx].type == EvalEquationResultType.CORRECT
            else FORMAT_SCORE
        )

    return coverage, majority, most_rewarded, reward_weighted


@pydra.main(base=EvaluateScriptConfig)
def main(config: Optional[EvaluateScriptConfig] = None):
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

    armo_rmp = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
    coverage_all = np.zeros(k, dtype=float)
    majority_all = np.zeros(k, dtype=float)
    most_rewarded_all = np.zeros(k, dtype=float)
    reward_weighted_all = np.zeros(k, dtype=float)
    tasks_pb = tqdm(tasks, desc="Processing tasks")
    for task in tasks_pb:
        res = process_sample(task, armo_rmp)
        assert (
            len(res[0]) == n_samples
            and len(res[1]) == n_samples
            and len(res[2]) == n_samples
        )
        scores, rewards, ee_results = res
        coverage, majority, most_rewarded, reward_weighted = get_voted_is_corrects(
            scores, rewards, ee_results, k
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
