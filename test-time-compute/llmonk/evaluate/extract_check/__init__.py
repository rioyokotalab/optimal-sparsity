from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from tqdm import tqdm

from llmonk.utils import EvaluateScriptConfig


@dataclass
class Task:
    sample_path: Path
    save_path: Path


def get_tasks(
    sample_dir: Path,
    save_dir: Path,
    save_path_suffix: str = "",
) -> List[Task]:
    tasks = []
    for sample_path in sample_dir.glob("*.yaml"):
        # save_path = save_dir / sample_path.name
        save_path = save_dir / sample_path.name.replace(
            ".yaml", f"{save_path_suffix}.yaml"
        )
        tasks.append(Task(sample_path=sample_path, save_path=save_path))
    return tasks


def check(
    config: EvaluateScriptConfig,
    process_task: Callable[[Task], List[bool]],
):
    tasks = get_tasks(config.samples_dir, config.save_dir, save_path_suffix="extract")
    tasks = sorted(tasks, key=lambda x: x.save_path)
    tasks = tasks[config.offset : config.limit : config.stride]
    print(f"Checking {len(tasks)} tasks")

    n_total = 0
    n_invalid = 0
    for task in tqdm(tasks):
        is_invalid = process_task(task)
        n_total += len(is_invalid)
        n_invalid += sum(is_invalid)
    print(f"Total samples        : {n_total}")
    print(f"Total invalid samples: {n_invalid}")
    print(f"Total valid samples  : {n_total - n_invalid}")
    print(f"Invalid ratio        : {n_invalid / n_total:.2%}")
    print(f"Valid ratio          : {(n_total - n_invalid) / n_total:.2%}")

    with open(f"{config.save_dir}_extract_summary.txt", "w") as f:
        f.write(f"Total samples        : {n_total}\n")
        f.write(f"Total invalid samples: {n_invalid}\n")
        f.write(f"Total valid samples  : {n_total - n_invalid}\n")
        f.write(f"Invalid ratio        : {n_invalid / n_total:.2%}\n")
        f.write(f"Valid ratio          : {(n_total - n_invalid) / n_total:.2%}\n")
