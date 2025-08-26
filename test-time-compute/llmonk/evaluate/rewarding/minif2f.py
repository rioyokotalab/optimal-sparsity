import csv
import functools
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pydra
from lean_dojo import (
    Dojo,
    DojoCrashError,
    DojoHardTimeoutError,
    DojoInitError,
    LeanGitRepo,
    ProofFinished,
    TacticState,
    Theorem,
)
from loguru import logger
from tqdm import tqdm

from llmonk.evaluate.rewarding import ArmoRMPipeline, Task, get_tasks
from llmonk.utils import (
    EvaluateScriptConfig,
    Timeout,
    get_theorem_name,
    load_yaml,
    save_yaml,
)


class ScriptConfig(EvaluateScriptConfig):
    repo_url: str = "https://github.com/rah4927/lean-dojo-mew"
    commit: str = "d00c776260c77de7e70125ef0cd119de6c0ff1de"
    file_path: str = "MiniF2F/Test.lean"


class DojoWrapper:
    def __init__(self, theorem):
        self.theorem = theorem
        self.reset_dojo()

    def reset_dojo(self):
        if hasattr(self, "dojo"):
            self.dojo._cleanup_tmp_dir()
        # set hard_timeout to 60 min. timeout is controled by out of dojo timeout
        dojo, init_state = Dojo(self.theorem, hard_timeout=60 * 60).__enter__()
        # dojo, init_state = Dojo(self.theorem).__enter__()
        self.dojo = dojo
        self.init_state = init_state


def get_proof_steps(completion: str) -> List[str]:
    """
    Split a lean proof completion into individual proof steps.
    """
    # First, split by newlines
    newline_steps = completion.split("\n")

    steps = []
    for newline_step in newline_steps:
        # Use regex to split on semicolons not surrounded by angle brackets
        # since <;> is a lean operator
        current_steps = re.split(r"(?<!<);(?!>)", newline_step)
        steps.extend(current_steps)

    # Remove Lean indentation and strip whitespace
    steps = [step.replace("·", "").strip() for step in steps]

    return steps


def check_proof(
    dojo_wrapper: DojoWrapper,
    proof_steps: List[str],
) -> Tuple[str, int]:
    state = dojo_wrapper.init_state
    dojo = dojo_wrapper.dojo
    step_cnt = 0

    try:
        for step in proof_steps:
            step_cnt += 1
            state_type = None

            try:
                with Timeout(10):
                    state = dojo.run_tac(state, step)
            except Timeout.Timeout:
                dojo_wrapper.reset_dojo()
                state_type = "Timeout"
                break

            if isinstance(state, ProofFinished) or not isinstance(state, TacticState):
                break

    except (DojoInitError, DojoHardTimeoutError, DojoCrashError):
        state_type = "Exception"

    if isinstance(state, ProofFinished):
        state_type = "Finished"
    else:
        if state_type is None:
            state_type = "TacticError/Incomplete"

    return state_type, step_cnt


def check_completion(
    theorem: Theorem,
    completions: List[str],
) -> Tuple[List[bool], List[str], List[int]]:
    dojo_wrapper = DojoWrapper(theorem)
    corrects: List[bool] = []
    states: List[str] = []
    num_steps: List[int] = []
    for completion in completions:
        proof_steps = get_proof_steps(completion)
        state, step_cnt = check_proof(dojo_wrapper, proof_steps)
        corrects.append(state == "Finished")
        states.append(state)
        num_steps.append(step_cnt)
    dojo_wrapper.dojo._cleanup_tmp_dir()
    return corrects, states, num_steps


def get_rewards(
    armo_rmp: ArmoRMPipeline,
    question: str,
    completions: List[str],
) -> List[float]:
    rewards: List[float] = []
    for completion in completions:
        answer = completion
        reward = armo_rmp(question, answer)
        rewards.append(reward)
    return rewards


def process_theorem(
    task: Task,
    repo_url: str,
    commit: str,
    file_path: Path,
):
    # print(f"Processing {task.sample_path} to {task.save_path}")
    if task.save_path.exists():
        # print(f"Already processed {task.sample_path} to {task.save_path}")
        return
    result = load_yaml(task.sample_path)
    repo = LeanGitRepo(repo_url, commit)
    theorem = Theorem(
        repo,
        file_path,
        get_theorem_name(result["theorem_name"]),
    )
    corrects, _, _ = check_completion(theorem, result["samples"])
    result["is_corrects"] = corrects
    del repo
    del theorem
    save_yaml(task.save_path, result)


def get_voted_is_corrects(
    corrects: List[bool],
    rewards: List[float],
    k: int,
) -> Tuple[List[bool], List[bool]]:
    coverage = [False] * k
    most_rewarded = [False] * k
    for i in range(k):
        n_using_samples = 2**i
        coverage[i] = len([tf for tf in corrects[:n_using_samples] if tf]) > 0
        max_reward = max(rewards[:n_using_samples])
        max_reward_is_correct: bool = [
            tf
            for tf, r in zip(corrects[:n_using_samples], rewards[:n_using_samples])
            if r == max_reward
        ][0]
        most_rewarded[i] = max_reward_is_correct
    return coverage, most_rewarded


@pydra.main(ScriptConfig)
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

    # use 8 workers for gpu_1
    # print("checking completions with 8 workers")
    process_theorem_partial = functools.partial(
        process_theorem,
        repo_url=config.repo_url,
        commit=config.commit,
        file_path=Path(config.file_path),
    )

    # remove annoying warning about resource limit unregarded
    def lean_dojo_filter(record):
        if (
            record["name"].startswith("lean_dojo.container")
            and record["level"].name == "WARNING"
        ):
            return False
        return True

    logger.remove()
    logger.add(sys.stderr, filter=lean_dojo_filter, level="WARNING")
    # with multiprocessing.Pool(processes=8) as pool:
    #     _ = list(
    #         tqdm(
    #             pool.map(process_theorem_partial, tasks),
    #             total=len(tasks),
    #             desc="Processing proofs",
    #         )
    #     )
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_theorem_partial, task) for task in tasks]
        for future in tqdm(futures, desc="Processing proofs"):
            future.result()
    # for task in tqdm(tasks, desc="Processing proofs"):
    #     process_theorem_partial(task)

    coverage_all = np.zeros(k, dtype=float)
    most_rewarded_all = np.zeros(k, dtype=float)
    armo_rmp = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
    for task in tqdm(tasks, desc="Getting rewards and voted is_corrects"):
        # checked proof in multiple workers, so read the results from the file
        result = load_yaml(task.save_path)
        rewards = get_rewards(
            armo_rmp,
            result["question"],
            result["samples"],
        )
        coverage, most_rewarded = get_voted_is_corrects(
            result["is_corrects"],
            rewards,
            k,
        )
        result["rewards"] = rewards
        save_yaml(task.save_path, result)
        coverage_all += np.array(coverage, dtype=float)
        most_rewarded_all += np.array(most_rewarded, dtype=float)

    coverage_all /= len(tasks)
    most_rewarded_all /= len(tasks)
    out_file = f"{config.save_dir}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["samples"] + [2**i for i in range(k)]
        writer.writerow(header)
        row = ["coverage"] + coverage_all.tolist()
        writer.writerow(row)
        row = ["most rewarded"] + most_rewarded_all.tolist()
        writer.writerow(row)

    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
