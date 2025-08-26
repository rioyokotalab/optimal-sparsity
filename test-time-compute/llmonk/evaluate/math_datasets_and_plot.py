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
        match_str = match.group(1).strip()
        match_str = filter_ignores(match_str, GSM8K_IGNORE_REGEXES)
        return match_str
    else:
        return INVALID_ANS_GSM8k

def is_correct_gsm8k(model_completion, gt_example):
    gt_answer = extract_answer_gsm8k(gt_example)
    assert gt_answer != INVALID_ANS_GSM8k
    model_answer = extract_answer_gsm8k(model_completion)
    return model_answer == gt_answer or is_equiv(model_answer, gt_answer)

def is_correct_minerva(og_pred, gt):
    pred = normalize_final_answer(get_unnormalized_answer(og_pred))
    gt_val = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    return pred == gt_val or is_equiv(pred, gt_val)

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
    if config.save_path.exists():
        return

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
    for sample in result.get("samples", []):
        try:
            correct = is_correct(sample, result["gt_answer"], config.dset)
        except Exception as e:
            correct = False
        corrects.append(correct)
    result["is_corrects"] = corrects
    save_yaml(config.save_path, result)

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

def check_individual_results(eval_dir):
    eval_dir = Path(eval_dir)
    for result_file in eval_dir.glob("*.yaml"):
        with open(result_file) as f:
            result = yaml.safe_load(f)
            correct_answers = sum(result["is_corrects"])
            total_samples = len(result["is_corrects"])
            print(f"Problem: {result_file.name}")
            print(f"Total samples: {total_samples}")
            print(f"Correct answers: {correct_answers}")
            accuracy = correct_answers / total_samples if total_samples > 0 else 0
            print(f"Accuracy: {accuracy:.2%}")
            print("-" * 50)

def extract_answer_sample(completion, dset):
    if dset == "math":
        return normalize_final_answer(get_unnormalized_answer(completion))
    elif dset == "gsm8k":
        return extract_answer_gsm8k(completion)
    else:
        raise ValueError(f"Dataset {dset} not supported")

def extract_answer_gt(completion, dset):
    if dset == "math":
        return normalize_final_answer(remove_boxed(last_boxed_only_string(completion)))
    elif dset == "gsm8k":
        return extract_answer_gsm8k(completion)
    else:
        raise ValueError(f"Dataset {dset} not supported")

def analyze_results(eval_dir, sample_sizes, dset):
    all_results = []
    for result_file in Path(eval_dir).glob("*.yaml"):
        with open(result_file) as f:
            result = yaml.safe_load(f)
            all_results.append(result)

    def get_majority_vote(samples):
        if not samples:
            return ""
        answers = [extract_answer_sample(s, dset) for s in samples]
        majority_votes = Counter(answers).most_common(1)
        if majority_votes:
            return majority_votes[0][0]
        else:
            return ""

    coverage_overall = np.mean([any(r["is_corrects"]) for r in all_results])
    majority_correct_overall = 0
    for result in all_results:
        majority_ans = get_majority_vote(result["samples"])
        gt_ans = extract_answer_gt(result["gt_answer"], dset)
        if is_equiv(majority_ans, gt_ans):
            majority_correct_overall += 1
    majority_accuracy_overall = majority_correct_overall / len(all_results) if len(all_results) > 0 else 0

    print(f"Total problems evaluated: {len(all_results)}")
    print(f"Overall Coverage (any correct): {coverage_overall:.2%}")
    print(f"Overall Majority voting accuracy: {majority_accuracy_overall:.2%}")

    results = []
    for k in sample_sizes:
        coverage_k = np.mean([any(r["is_corrects"][:k]) for r in all_results])
        majority_correct_k = 0
        for result in all_results:
            samples_k = result["samples"][:k]
            majority_ans = get_majority_vote(samples_k)
            gt_ans = extract_answer_gt(result["gt_answer"], dset)
            if is_equiv(majority_ans, gt_ans):
                majority_correct_k += 1
        majority_accuracy_k = majority_correct_k / len(all_results) if len(all_results) > 0 else 0
        print(f"Coverage@{k}: {coverage_k:.2%}")
        print(f"Majority Voting Accuracy@{k}: {majority_accuracy_k:.2%}")
        results.append({
            "sample_size": k,
            "coverage": coverage_k,
            "majority_voting_accuracy": majority_accuracy_k
        })
    return results

def plot_results(result_stats, output_png):
    sample_sizes = [stat["sample_size"] for stat in result_stats]
    coverages = [stat["coverage"] for stat in result_stats]
    majority_accuracies = [stat["majority_voting_accuracy"] for stat in result_stats]

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, coverages, "b-", label="Coverage", marker="o")
    plt.plot(sample_sizes, majority_accuracies, "r-", label="Majority Voting", marker="s")
    plt.xscale("log")
    plt.xlabel("Number of Samples")
    plt.ylabel("Accuracy")
    plt.title("Performance vs Number of Samples")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_png)
    plt.close()
    print(f"Graph has been saved to {output_png}")

def output_csv(result_stats, output_csv_path):
    sample_sizes = [stat["sample_size"] for stat in result_stats]
    coverages = [stat["coverage"] for stat in result_stats]
    majority_accuracies = [stat["majority_voting_accuracy"] for stat in result_stats]

    header = ["Metric"] + [str(s) for s in sample_sizes]
    rows = [
        ["coverage"] + coverages,
        ["majority_voting_accuracy"] + majority_accuracies,
    ]
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"CSV file has been saved to {output_csv_path}")

@pydra.main(base=ScriptConfig)
def main(config: ScriptConfig):
    tasks = get_tasks(config)
    tasks = sorted(tasks, key=lambda x: x.save_path)
    tasks = tasks[config.offset : config.limit : config.stride]
    print(f"Evaling on {len(tasks)} problems.")

    if config.num_workers not in [0, None]:
        with multiprocessing.Pool(processes=config.num_workers) as pool:
            list(tqdm(pool.map(process_sample, tasks), total=len(tasks)))
    else:
        for task in tqdm(tasks):
            process_sample(task)

    eval_dir = config.save_dir  
    print("\nChecking individual results:")
    check_individual_results(eval_dir)

    print("\nAnalyzing overall results...")
    sample_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    result_stats = analyze_results(eval_dir, sample_sizes, config.dset)

    eval_dir_path = Path(eval_dir)
    output_csv_file = f"{eval_dir_path}.csv"
    output_png_file = f"{eval_dir_path}.png"

    output_csv(result_stats, output_csv_file)
    plot_results(result_stats, output_png_file)

if __name__ == "__main__":
    main()
