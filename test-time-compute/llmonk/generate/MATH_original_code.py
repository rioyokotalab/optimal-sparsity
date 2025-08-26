import json
from pathlib import Path
import torch
from tqdm import tqdm
import pydra
import random
from functools import partial
import vllm

from llmonk.generate.prompts import MATH_CODE_COT_PROMPT
from llmonk.utils import save_yaml, GenerateScriptConfig


def load_jsonl(file_path):
    dataset = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            data["id"] = i  
            dataset.append(data)
    return dataset


def run_inference(item, config: GenerateScriptConfig, model):
    outpath = config.save_dir / f"{item['id']}.yaml"
    if outpath.exists():
        return

    prompt = MATH_CODE_COT_PROMPT + f"def simple_math_problem():\n    '''\n    {item['problem']}\n    '''\n"

    sampling_params = vllm.SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=config.stop_strings,
        logprobs=1,  
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
        "question": item["problem"],
        "samples": samples,
        "gt_answer": item["solution"],
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

    train_path = Path("dataset/math_splits/train.jsonl")
    test_path = Path("dataset/math_splits/test.jsonl")

    train_dataset = load_jsonl(train_path)
    test_dataset = load_jsonl(test_path)

    print(f"Number of test items: {len(test_dataset)}")
    print(f"Number of train items: {len(train_dataset)}")

    random.seed(config.seed)
    for i, data in enumerate(test_dataset):
        few_shot_items = random.sample(train_dataset, min(config.num_few_shot, len(train_dataset)))
        data["few_shot_items"] = few_shot_items

    random.shuffle(test_dataset)

    if config.limit is not None:
        limit = config.limit
    else:
        limit = len(test_dataset)
    stride = config.stride if config.stride is not None else 1
    offset = config.offset if config.offset is not None else 0

    test_dataset = test_dataset[offset:limit:stride]

    print(f"Total number of items to process: {len(test_dataset)}")

    for item in tqdm(test_dataset, desc="Processing all items"):
        run_inference(item, config=config, model=model)


if __name__ == "__main__":
    main()
