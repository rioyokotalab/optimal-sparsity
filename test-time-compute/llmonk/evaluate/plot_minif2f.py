import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml

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
    all_results = []

    for result_file in Path(eval_dir).glob("*.yaml"):
        with open(result_file) as f:
            result = yaml.safe_load(f)
            all_results.append(result)

    coverage_overall = np.mean([any(r["is_corrects"]) for r in all_results])
    print(f"Total problems evaluated: {len(all_results)}")
    print(f"Overall Coverage (any correct): {coverage_overall:.2%}")

    results = []
    for k in sample_sizes:
        coverage_k = np.mean([any(r["is_corrects"][:k]) for r in all_results])
        print(f"Coverage@{k}: {coverage_k:.2%}")
        results.append({
            "sample_size": k,
            "coverage": coverage_k,
        })
    return results

def plot_results(result_stats, output_png):
    sample_sizes = [stat["sample_size"] for stat in result_stats]
    coverages = [stat["coverage"] for stat in result_stats]

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, coverages, "b-", label="Coverage", marker="o")
    plt.xscale("log")
    plt.xlabel("Number of Samples")
    plt.ylabel("Coverage")
    plt.title("Coverage vs Number of Samples")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_png)
    plt.close()
    print(f"Graph has been saved to {output_png}")

def output_csv(result_stats, output_csv_path):
    sample_sizes = [stat["sample_size"] for stat in result_stats]
    coverages = [stat["coverage"] for stat in result_stats]

    header = ["Metric"] + [str(s) for s in sample_sizes]
    rows = [
        ["coverage"] + coverages,
    ]

    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"Transposed CSV file has been saved to {output_csv_path}")

def main(eval_dir):
    eval_path = Path(eval_dir)
    output_base = eval_path
    output_csv_file = f"{output_base}.csv"
    output_png_file = f"{output_base}.png"

    print("Checking individual results:")
    check_individual_results(eval_dir)
    print("\nAnalyzing overall results...")

    sample_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    result_stats = analyze_results(eval_dir, sample_sizes)

    output_csv(result_stats, output_csv_file)
    plot_results(result_stats, output_png_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <eval_dir>")
        sys.exit(1)
    eval_dir_arg = sys.argv[1]
    main(eval_dir_arg)
