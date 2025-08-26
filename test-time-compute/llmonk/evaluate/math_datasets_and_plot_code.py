#!/usr/bin/env python
import csv
import re
from pathlib import Path
from collections import Counter
from copy import deepcopy
import multiprocessing
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pydra
from tqdm import tqdm
import signal
from time import monotonic

from lm_eval.tasks.minerva_math.utils import (
    last_boxed_only_string,
    normalize_final_answer,
    get_unnormalized_answer,
    remove_boxed,
    is_equiv,
)

from llmonk.utils import load_yaml, save_yaml, EvaluateScriptConfig
import ast

ANS_RE_GSM8k = re.compile(r"#### (\-?[\$0-9\.\,]+)")
INVALID_ANS_GSM8k = "[invalid]"
GSM8K_IGNORE_REGEXES = [",", "\\$", "\\.$"]

def filter_ignores(st, regexes_to_ignore):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            st = re.sub(s, "", st)
    return st

def extract_answer_gsm8k(completion):
    match = ANS_RE_GSM8k.search(completion)
    if match:
        return match.group(1)
    else:
        return INVALID_ANS_GSM8k

class TimeoutError(Exception):
    pass


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = (
                final_answer.split("#### ")[1].replace(",", "").replace("$", "")
            )
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def try_float(s):
    try:
        f = float(s)
    except Exception:
        f = None
    return f

def extract_up_to_return(code_str: str) -> str:
    m = re.search(r'^(.*?^.*?return[^\n]*\n)', code_str, re.MULTILINE | re.DOTALL)
    return m.group(1) if m else code_str

def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.0, score=1.0
):
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution exceeded the time limit!")
    solution_str = extract_up_to_return(solution_str)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)
    try:
        if "import" not in solution_str:
            namespace = {}
            exec(
                f"def simple_math_problem():\n{solution_str}",
                namespace
            )
            answer = float(namespace["simple_math_problem"]())
        else:
            answer = 0
    except Exception as e:
        print(e)
        answer = 0
    finally:
        signal.alarm(0)
    if answer == 0:
        answer1 = extract_solution(solution_str=solution_str, method="strict")
        answer2 = extract_solution(solution_str=solution_str, method="flexible")
        if answer1 is None and answer2 is None:
            answer = 0
        else:
            float_answer = try_float(answer1)
            float_answer2 = try_float(answer2)
            float_ground_truth = try_float(ground_truth)
            if answer1 == ground_truth:
                answer = answer1
            elif answer2 == ground_truth:
                answer = answer2
            elif (
                float_answer is not None
                and float_ground_truth is not None
                and abs(float_answer - float_ground_truth) < 1e-5
            ):
                answer = answer1
            elif (
                float_answer2 is not None
                and float_ground_truth is not None
                and abs(float_answer2 - float_ground_truth) < 1e-5
            ):
                answer = answer2
            else:
                answer = 0
    return answer

def compute_score_MATH(
    solution_str
):
    import subprocess, tempfile, os, json
    solution_str = extract_up_to_return(solution_str)
    tmp_py = tempfile.NamedTemporaryFile(suffix=".py", delete=False, dir=os.getcwd())
    tmp_py.write((
        "def simple_math_problem():\n" +
        solution_str +
        "print(simple_math_problem())\n"
    ).encode("utf-8"))
    tmp_py.flush()
    tmp_py.close()

    cmd = [
        "singularity", "exec",
        "--bind", f"{os.getcwd()}:/work",
        "math-runner.sif",
        "python", f"/work/{os.path.basename(tmp_py.name)}"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode != 0:
            raise RuntimeError(f"Container exec failed: {res.stderr}")
        answer = res.stdout.strip()
    except Exception as e:
        answer = "ERROR"
    os.remove(tmp_py.name)
    if answer == "ERROR":
        answer = get_unnormalized_answer(solution_str)
        if answer  == '[invalidanswer]':
            answer = "ERROR"
    return answer


def is_correct_gsm8k_code(model_completion, gt_example):
    gt_answer = extract_answer_gsm8k(gt_example)
    assert gt_answer != INVALID_ANS_GSM8k
    answer = compute_score(model_completion, gt_answer)
    return (abs(float(answer) - float(gt_answer)) < 1e-5), answer, gt_answer

def is_correct_minerva_code(og_pred, gt):
    pred_raw = compute_score_MATH(og_pred)
    pred = normalize_final_answer(pred_raw)
    gt_val = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    if not (pred == gt_val or is_equiv(pred, gt_val)):
        pred = pred_raw
    
    return pred == gt_val or is_equiv(pred, gt_val), pred, gt_val

class ScriptConfig(EvaluateScriptConfig):
    dset: str = "gsm8k"

def is_correct(sample: str, gt_answer: str, dset: str):
    if dset == "gsm8k":
        return is_correct_gsm8k_code(sample, gt_answer)
    elif dset == "math":
        return is_correct_minerva_code(sample, gt_answer)
    else:
        raise ValueError(f"Dataset {dset} not supported")

def process_sample(config: EvaluateScriptConfig):
    try:
        result = load_yaml(config.sample_path)
    except yaml.scanner.ScannerError as err:
        print(f"YAML error in {config.sample_path}: {err}. Marking all samples as incorrect.")
        result = {
            "samples": [],
            "gt_answer": "",
            "is_corrects": []  
        }
        save_yaml(config.save_path, result)
        return

    corrects = []
    answers = []
    for sample in result.get("samples", []):
        try:
            correct, answer, truth_answer = is_correct(sample, result["gt_answer"], config.dset)
            if correct:
                print("あってる！")
            print("回答",answer,"答え", truth_answer)
        except Exception as e:
            print(e)
            correct = False
            answer = 0
        corrects.append(bool(correct))
        answers.append(answer)
    if config.dset == "math":
        truth_answer = extract_answer_gsm8k(result["gt_answer"])
    result["is_corrects"] = corrects
    save_yaml(config.save_path, result)
    coverage = []
    majority = []
    IGNORE = {'[invalid]'}
    for i in range(8):
        counter = Counter(answers[:2**i])
        common = counter.most_common()
        filtered = [(val, c) for val, c in common if val not in IGNORE]

        if filtered:
            candidates = [filtered[0][0]]
        else:
            candidates = []

        hit = False
        for c in candidates:
            if config.dset == "gsm8k":
                try:
                    if abs(int(truth_answer) - float(c)) < 1e-5:
                        hit = True
                except ValueError:
                    hit = False
            else:  
                if c == truth_answer or is_equiv(c, truth_answer):
                    hit = True
        majority.append(hit)
        coverage.append(max(corrects[:2**i]))
    print("カバレッジ",coverage)
    return np.array(coverage), majority

def get_tasks(config):
    sample_paths = Path(config.samples_dir).glob("*.yaml")
    tasks = []
    for sample_path in tqdm(sample_paths, desc="Loading generations"):
        save_path = config.save_dir / sample_path.name
        task_config = deepcopy(config)
        task_config.sample_path = sample_path
        task_config.save_path = save_path
        tasks.append(task_config)
    return tasks

@pydra.main(base=ScriptConfig)
def main(config: ScriptConfig):
    tasks = get_tasks(config)
    tasks = sorted(tasks, key=lambda x: x.save_path)
    tasks = tasks[config.offset : config.limit : config.stride]
    print(f"Evaling on {len(tasks)} problems.")
    eval_dir = config.save_dir 
    coverage_sum = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    majority_sum = np.zeros(8, dtype=int)
    
    i = 0
    for task in tqdm(tasks):
        coverage, majority = process_sample(task)
        coverage_sum += coverage
        majority_sum += np.array(majority, dtype=int)
        i += 1
    coverage_all = coverage_sum / i
    majority_all = majority_sum / i
    out_file=f"{eval_dir}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["samples"] + [1,2,4,8,16,32,64,128]
        writer.writerow(header)
        row = ["coverage"] + coverage_all.tolist()
        writer.writerow(row)
        row = ["majority vote"] + majority_all.tolist()
        writer.writerow(row)

    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
