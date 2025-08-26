from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from taskloss_eval.task_loss import TaskLoss


from pathlib import Path


def rel_model_path(model_str: str) -> Path:
    p = Path(model_str)
    return Path(*p.parts[1:]) if p.is_absolute() else p


def make_out_path(model: str, task_path: Path) -> Path:
    rel_task = task_path.relative_to("datasets").with_suffix(".csv")
    return Path("results") / rel_model_path(model) / rel_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name / path")
    ap.add_argument(
        "--task", type=Path, required=True, help="JSON file [{prompt,answer}, ...]"
    )
    ap.add_argument(
        "--tp", type=int, default=1, help="tensor_parallel_size for vLLM (default=1)"
    )
    args = ap.parse_args()

    data = json.loads(args.task.read_text())
    qa_pairs = [(d["prompt"], d["answer"]) for d in data]

    tl = TaskLoss(args.model, tensor_parallel_size=args.tp)
    bits_tok, bits_byte = tl.eval_batch(qa_pairs)

    out_path = make_out_path(args.model, args.task)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "bits_token", "bits_byte"])
        for i, (bt, bb) in enumerate(zip(bits_tok, bits_byte)):
            w.writerow([i, f"{bt:.4f}", f"{bb:.4f}"])
        w.writerow(["AVG", f"{bits_tok.mean():.4f}", f"{bits_byte.mean():.4f}"])

    print(f"✓ wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
