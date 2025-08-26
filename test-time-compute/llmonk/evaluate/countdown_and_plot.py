#!/usr/bin/env pytho#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import re
import yaml
import numpy as np
import csv
from collections import Counter

def extract_solution(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None

    final_answer = matches[-1].group(1).strip()
    if '=' in final_answer:
        final_answer = final_answer.split('=', 1)[0].strip()
    return final_answer

def validate_equation(equation_str, available_numbers):
    try:
        nums = sorted([int(n) for n in re.findall(r'\d+', equation_str)])
        return nums == sorted(available_numbers)
    except:
        return False

def evaluate_equation(equation_str):
    try:
        if not re.match(r'^[\d+\-*/().\s]+$', equation_str):
            return None
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None

def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    t, nums = ground_truth['target'], ground_truth['numbers']
    eq = extract_solution(solution_str)
    if eq is None:
        return 0.0, 0.0
    if not validate_equation(eq, nums):
        return format_score, 0.5
    res = evaluate_equation(eq)
    if res is None:
        res = 0
    return (score, res) if (res is not None and abs(res - t) < 1e-5) \
       else (format_score, res)

def process_file(in_path: Path, out_path: Path):
    data = yaml.safe_load(in_path.read_text(encoding='utf-8'))
    gt = {'target': data['target'], 'numbers': data['numbers']}
    returned = [compute_score(s, gt) for s in data.get('samples', [])]
    is_corrects, answers = zip(*returned)
    coverage = []
    majority = []
    IGNORE = {0, 0.5}
    for i in range(8):
        counter = Counter(answers[:2**i])
        common = counter.most_common()
        filtered = [(val, c) for val, c in common if val not in IGNORE]

        if filtered:
            max_count = max(c for _, c in filtered)
            candidates = [val for val, c in filtered if c == max_count]
        else:
            candidates = [common[0][0]]

        hit = any(abs(gt['target'] - c) < 1e-5 for c in candidates)
        majority.append(hit)
        coverage.append(max(is_corrects[:2**i]))
    data['is_corrects'] = is_corrects

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return np.array(coverage), majority # [0.1, 1.0]

def main():
    p = argparse.ArgumentParser(description="Add is_corrects to YAMLs")
    p.add_argument('--input_dir',  required=True)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    inp, outp = Path(args.input_dir), Path(args.output_dir)
    coverage_sum = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    majority_sum = np.zeros(8, dtype=int)
    
    i = 0
    for file in inp.glob("*.yaml"):
        rel = file.relative_to(inp)
        coverage, majority = process_file(file, outp/rel)
        coverage_sum += coverage
        majority_sum += np.array(majority, dtype=int)
        print(f"→ {rel}")
        i += 1
    coverage_all = coverage_sum / i
    majority_all = majority_sum / i
    out_file=f"{args.output_dir}.csv"
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
