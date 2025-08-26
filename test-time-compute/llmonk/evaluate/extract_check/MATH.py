import functools
from typing import List, Optional

import pydra
from lm_eval.tasks.minerva_math.utils import (
    get_unnormalized_answer,
    last_boxed_only_string,
    normalize_final_answer,
)

from llmonk.evaluate.extract_check import Task, check
from llmonk.utils import EvaluateScriptConfig, load_yaml, save_yaml

# def remove_boxed(s: str) -> str:
#     if "\\boxed " in s:
#         left = "\\boxed "
#         assert s[: len(left)] == left
#         return s[len(left) :]
#
#     left = "\\boxed{"
#
#     assert s[: len(left)] == left, f"{s}"
#     assert s[-1] == "}"
#
#     return s[len(left) : -1]


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left, f"{s}"
        return s[len(left) :]
    elif "\\boxed{" in s:
        left = "\\boxed{"
        assert s[: len(left)] == left, f"{s}"
        assert s[-1] == "}", f"{s}"
        return s[len(left) : -1]
    elif "\\fbox{" in s:
        left = "\\fbox{"
        assert s[: len(left)] == left, f"{s}"
        assert s[-1] == "}", f"{s}"
        return s[len(left) : -1]
    else:
        # raise ValueError(f"Unknown boxing format: {s}")
        print(f"Unknown boxing format: {s}")
        return "[invalidanswer]"


def process_task(
    task: Task,
    boxed_math: bool,
    extract_both: bool,
) -> List[bool]:
    result = load_yaml(task.sample_path)
    invalid_ans = "[invalidanswer]"
    is_invalid: List[bool] = []
    for sample in result["samples"]:
        extracted_ans = extract_ans(sample, boxed_math, extract_both)
        is_invalid.append(extracted_ans == invalid_ans)
    save_yaml(task.save_path, {"invalid": is_invalid})
    return is_invalid


def extract_ans(
    sample: str,
    boxed_math: bool,
    extract_both: bool,
) -> str:
    if extract_both:
        unnormalized = get_unnormalized_answer(sample)
        if unnormalized == "[invalidanswer]":
            last_boxed = last_boxed_only_string(sample)
            if last_boxed is None:
                return "[invalidanswer]"
            return normalize_final_answer(remove_boxed(last_boxed))
        return normalize_final_answer(unnormalized)
    if boxed_math:
        last_boxed = last_boxed_only_string(sample)
        if last_boxed is None:
            return "[invalidanswer]"
        return normalize_final_answer(remove_boxed(last_boxed))
    return normalize_final_answer(get_unnormalized_answer(sample))


class ScriptConfig(EvaluateScriptConfig):
    boxed_math: bool = False
    extract_both: bool = False


@pydra.main(base=ScriptConfig)
def main(config: Optional[ScriptConfig] = None):
    assert config is not None
    process_task_partial = functools.partial(
        process_task,
        boxed_math=config.boxed_math,
        extract_both=config.extract_both,
    )
    check(
        config,
        process_task=process_task_partial,
    )


if __name__ == "__main__":
    main()
