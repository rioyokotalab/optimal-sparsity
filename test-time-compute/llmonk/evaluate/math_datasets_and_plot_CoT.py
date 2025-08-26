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

from lm_eval.tasks.minerva_math.utils import (
    last_boxed_only_string,
    normalize_final_answer,
    get_unnormalized_answer,
    remove_boxed,
    is_equiv,
)

from llmonk.utils import load_yaml, save_yaml, EvaluateScriptConfig

ANS_RE_GSM8k = re.compile(r"(?:####\s*|Answer:?\s*|The answer is\s*)(-?\d+(?:\.\d+)?)",re.IGNORECASE)
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

def is_correct_gsm8k(model_completion, gt_example):
    gt_answer = extract_answer_gsm8k(gt_example)
    assert gt_answer != INVALID_ANS_GSM8k
    model_answer = extract_answer_gsm8k(model_completion)
    return model_answer == gt_answer or is_equiv(model_answer, gt_answer), model_answer, gt_answer

def is_correct_minerva(og_pred, gt):
    pred = normalize_final_answer(get_unnormalized_answer(og_pred))
    gt_val = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    return (pred == gt_val or is_equiv(pred, gt_val)), pred, gt_val

class ScriptConfig(EvaluateScriptConfig):
    dset: str = "gsm8k"

def is_correct(sample: str, gt_answer: str, dset: str):
    if dset == "gsm8k":
        return is_correct_gsm8k(sample, gt_answer)
    elif dset == "math":
        return is_correct_minerva(sample, gt_answer)
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
        except Exception as e:
            correct = False
            answer = 0
        corrects.append(bool(correct))
        answers.append(answer)
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
