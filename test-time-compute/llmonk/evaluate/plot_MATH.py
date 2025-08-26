import sys
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# MATH 用の正規化・抽出ユーティリティ関数をインポート
from lm_eval.tasks.minerva_math.utils import (
    last_boxed_only_string,
    normalize_final_answer,
    get_unnormalized_answer,
    remove_boxed,
    is_equiv
)

# サンプル側の回答抽出関数（出力から箱付きでなく生の形式を正規化）
def extract_answer_math_sample(completion):
    # 生の予測から不要な部分を取り除いて正規化
    return normalize_final_answer(get_unnormalized_answer(completion))

# 正解側の抽出関数（gt_answer から、箱付きの表現を取り出して正規化）
def extract_answer_math_gt(completion):
    return normalize_final_answer(remove_boxed(last_boxed_only_string(completion)))

def check_individual_results(eval_dir):
    eval_dir = Path(eval_dir)
    for result_file in eval_dir.glob("*.yaml"):
        with open(result_file) as f:
            result = yaml.safe_load(f)
            correct_answers = sum(x if isinstance(x, int) else 0 for x in result["is_corrects"])
            total_samples = len(result["is_corrects"])
            print(f"Problem: {result_file.name}")
            print(f"Total samples: {total_samples}")
            print(f"Correct answers: {correct_answers}")
            accuracy = correct_answers / total_samples if total_samples > 0 else 0
            print(f"Accuracy: {accuracy:.2%}")
            print("-" * 50)

def analyze_results(eval_dir, sample_sizes):
    """
    読み込んだすべての結果に対して、各サンプルサイズでのカバレッジと多数決正解率を算出する。
    """
    all_results = []

    for result_file in Path(eval_dir).glob("*.yaml"):
        with open(result_file) as f:
            result = yaml.safe_load(f)
            all_results.append(result)

    def get_majority_vote(samples):
        # サンプルごとの回答抽出には、MATH用のサンプル抽出関数を利用
        answers = [extract_answer_math_sample(s) for s in samples]
        return Counter(answers).most_common(1)[0][0]

    # 全体の集計（デバッグ用）
    coverage_overall = np.mean([any(r["is_corrects"]) for r in all_results])
    majority_correct_overall = 0
    for result in all_results:
        majority_ans = get_majority_vote(result["samples"])
        # gt_answer の抽出には MATH 用の gt 抽出関数を利用
        gt_ans = extract_answer_math_gt(result["gt_answer"])
        if is_equiv(majority_ans, gt_ans):
            majority_correct_overall += 1
    majority_accuracy_overall = majority_correct_overall / len(all_results) if len(all_results) > 0 else 0

    print(f"Total problems evaluated: {len(all_results)}")
    print(f"Overall Coverage (any correct): {coverage_overall:.2%}")
    print(f"Overall Majority voting accuracy: {majority_accuracy_overall:.2%}")

    # 各サンプルサイズごとに評価を計算
    results = []
    for k in sample_sizes:
        coverage_k = np.mean([any(r["is_corrects"][:k]) for r in all_results])
        majority_correct_k = 0
        for result in all_results:
            samples_k = result["samples"][:k]
            majority_ans = get_majority_vote(samples_k)
            gt_ans = extract_answer_math_gt(result["gt_answer"])
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
    print(f"Transposed CSV file has been saved to {output_csv_path}")

def main(eval_dir):
    eval_path = Path(eval_dir)
    # 出力ファイルは eval_dir のディレクトリ名をもとに作成します
    output_base = eval_path
    output_csv_file = f"{output_base}.csv"
    output_png_file = f"{output_base}.png"

    print("Checking individual results:")
    check_individual_results(eval_dir)
    print("\nAnalyzing overall results...")

    # 指定されたサンプルサイズ
    sample_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    result_stats = analyze_results(eval_dir, sample_sizes)

    # CSV 出力（縦横反転した形式）
    output_csv(result_stats, output_csv_file)
    # PNG 出力
    plot_results(result_stats, output_png_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <eval_dir>")
        sys.exit(1)
    eval_dir_arg = sys.argv[1]
    main(eval_dir_arg)
