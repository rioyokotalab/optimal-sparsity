import csv
import re
import signal
from typing import Any, List, Optional, Tuple

import numpy as np
import pydra
from math_verify import parse, verify
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


class ScriptConfig(EvaluateScriptConfig):
    rmp: str = "DEFAULT"

    def finalize(self):
        super().finalize()
        if self.rmp == "DEFAULT":
            raise ValueError(
                "RMP model not specified. Please set rmp_model in the config."
            )
        assert self.rmp in ["Armo", "Skywork"], "RMP model not supported"


def handler(_signum, _frame):
    raise TimeoutError("Execution timed out!")


def execute_function(code: str, timeout=3) -> Optional[str]:
    try:
        # Set the alarm handler
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)  # Start the alarm
        local_namespace = {}
        exec(code, {}, local_namespace)
        return str(local_namespace["simple_math_problem"]())
    except TimeoutError:
        return None
    except Exception:
        return None
    finally:
        # Always disable the alarm after execution
        signal.alarm(0)


def execute_tinygsm_code(text) -> Optional[str]:
    code = text.split("\ndef")[-1]
    code = "def" + code
    try:
        return execute_function(code)
    except:
        return None


def execute_llm_code(text) -> Optional[str]:
    try:
        # Extract code inside <llm-code> tags
        code_match = re.search(r"<llm-code>(.*?)</llm-code>", text, re.DOTALL)
        if not code_match:
            return None

        code = code_match.group(1).strip()

        # Create a dictionary for execution context
        exec_globals = {}

        # Split the code into lines and execute it
        lines = code.split("\n")
        last_expr = lines[-1]  # The last line of code
        timeout = 3

        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)  # Start the alarm
            exec(code, exec_globals)
        except TimeoutError:
            return None
        except Exception:
            return None
        finally:
            # Always disable the alarm after execution
            signal.alarm(0)

        return str(eval(last_expr, exec_globals))
    except:
        return None


def get_llm_answer(
    text,
):
    response_type = "text"
    if "<llm-code>" in text:
        code_out = execute_llm_code(text)
        response_type = "llm-code"
        if code_out is not None:
            return parse(code_out), "llm-code"
    if "def" in text:
        code_out = execute_tinygsm_code(text)
        response_type = "tinygsm-code"
        if code_out is not None:
            return parse(code_out), "tinygsm-code"

    return parse(text), response_type


def compute_score(
    solution_str,
    ground_truth,
    _method="strict",
    score=1.0,
) -> Tuple[float, Any, Any]:
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    llm_answer, _ = get_llm_answer(solution_str)
    correct_answer = parse(ground_truth)
    ret = 0.0
    if verify(llm_answer, correct_answer) == True:
        ret = score
    return ret, llm_answer, ground_truth


def is_correct(
    sample: str,
    gt_answer: str,
) -> Tuple[bool, str, str]:
    """
    Check if the model's answer is correct against the ground truth answer.
    returns:
        - bool: Whether the model's answer is correct.
        - str: The model's answer.
        - str: The ground truth answer.
    """
    ret, model_answer, gt_answer = compute_score(
        sample,
        gt_answer,
    )
    is_correct = ret > 0.0
    return is_correct, model_answer, gt_answer


def process_sample(
    task: Task,
    rmp: RMPipeline,
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
        )

        corrects.append(correct)
        rewards.append(reward)
        model_answers.append(model_answer)
    result["is_corrects"] = corrects
    result["rewards"] = rewards
    result["model_answers"] = model_answers
    save_yaml(task.save_path, result)
    return corrects, rewards, model_answers


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

    if n_samples != 1:
        raise NotImplementedError("TTC evaluation is not implemented for n_samples > 1")

    if config.rmp == "Armo":
        rmp = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
    elif config.rmp == "Skywork":
        rmp = SkyworkRMPipeline("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
    coverage_all = np.zeros(k, dtype=float)
    tasks_pb = tqdm(tasks, desc="Processing tasks")
    for task in tasks_pb:
        res = process_sample(task, rmp)
        if res is None:
            tasks_pb.write(f"Error processing {task.sample_path}")
            continue
        assert (
            len(res[0]) == n_samples
            and len(res[1]) == n_samples
            and len(res[2]) == n_samples
        )
        # corrects, rewards, model_answers = res
        corrects = res[0]
        coverage = np.array([corrects[0]], dtype=float)
        coverage_all += np.array(coverage, dtype=float)
    coverage_all /= len(tasks)
    out_file = f"{config.save_dir}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["samples"] + [2**i for i in range(k)]
        writer.writerow(header)
        row = ["coverage"] + coverage_all.tolist()
        writer.writerow(row)

    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()

