import re
from typing import List, Optional

import pydra

from llmonk.evaluate.extract_check import Task, check
from llmonk.utils import EvaluateScriptConfig, load_yaml, save_yaml


def process_task(
    task: Task,
) -> List[bool]:
    result = load_yaml(task.sample_path)
    invalid_ans = "[invalid]"
    is_invalid: List[bool] = []
    for sample in result["samples"]:
        extracted_ans = extract_ans(sample)
        is_invalid.append(extracted_ans == invalid_ans)
    save_yaml(task.save_path, {"invalid": is_invalid})
    return is_invalid


def extract_ans(
    sample: str,
) -> str:
    ans_re = re.compile(
        r"(?:####\s*|Answer:?\s*|The answer is\s*)(-?\d+(?:\.\d+)?)", re.IGNORECASE
    )
    match = ans_re.search(sample)
    if match:
        return match.group(1)
    else:
        return "[invalid]"


@pydra.main(base=EvaluateScriptConfig)
def main(config: Optional[EvaluateScriptConfig] = None):
    assert config is not None
    check(
        config,
        process_task=process_task,
    )


if __name__ == "__main__":
    main()
