import json
from datasets import load_dataset
import re
from pathlib import Path
import torch
from tqdm import tqdm
import pydra
import random
from functools import partial
import vllm

from llmonk.generate.prompts import MINIF2F_FEW_SHOT_PROMPT
from llmonk.utils import save_yaml, GenerateScriptConfig


def replace_theorem_name(lean_code, new_name):
    """
    Replace dataset's theorem name with a generic name
    to avoid leaking information about how to solve the problem.
    """
    pattern = r"theorem\s+\w+\s*\n"
    replacement = f"theorem{new_name}\n"
    modified_code = re.sub(pattern, replacement, lean_code)
    return modified_code


def get_lean_prompt(data, theorem_name: str, add_solution: bool = False):
    header = "Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.\n"
    header += "```" + data["header"]
    stmt = data["formal_statement"].replace(" sorry", "").replace("sorry", "")
    if add_solution:
        prompt = header + "\n" + stmt + data["solution"] + "```"
    else:
        prompt = header + "\n" + stmt + "\nby (\n"

    prompt = replace_theorem_name(prompt, theorem_name)
    return prompt


def run_inference(item, config: GenerateScriptConfig, model):
    outpath = config.save_dir / f"{item['id']}.yaml"
    if outpath.exists():
        return

    # we use five few-shot examples
    prompt = MINIF2F_FEW_SHOT_PROMPT + get_lean_prompt(item, theorem_name="6")

    sampling_params = vllm.SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=config.stop_strings,
    )

    num_samples = config.num_samples
    batch_size = config.batch_size
    assert num_samples % batch_size == 0

    samples = []
    for _ in tqdm(range(num_samples // batch_size), desc=f"Item {item['id']}"):
        requests_list = [prompt] * batch_size
        responses = model.generate(requests_list, sampling_params=sampling_params, use_tqdm=False)
        for response in responses:
            samples.append(response.outputs[0].text)

    out = {
        "prompt": prompt,
        "question": item["formal_statement"],
        "samples": samples,
        "theorem_name": item["id"],
    }
    save_yaml(outpath, out)


@pydra.main(GenerateScriptConfig)
def main(config: GenerateScriptConfig):
    model = vllm.LLM(
        model=config.model,
        dtype="auto",
        trust_remote_code=True,
        pipeline_parallel_size=1,
        tensor_parallel_size=config.tensor_parallel_size,
    )

    dataset = load_dataset("cat-searcher/minif2f-lean4")
    math_problems = [p for p in dataset["test"] if "mathd" in p["id"]]
    assert len(math_problems) == 130

    if config.limit is not None:
        limit = config.limit
    else:
        limit = len(math_problems)
    stride = config.stride if config.stride is not None else 1
    offset = config.offset if config.offset is not None else 0
    math_problems = math_problems[offset:limit:stride]
    print(f"Total number of items to process: {len(math_problems)}")

    for item in tqdm(math_problems, desc="Processing all items"):
        run_inference(item, config=config, model=model)


if __name__ == "__main__":
    main()
