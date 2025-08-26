import time
from pathlib import Path
import torch
from tqdm import tqdm
import pydra
import random
from functools import partial
import vllm
import json

from llmonk.generate.prompts import Countdown_COT_PROMPT
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
    

    prompt = Countdown_COT_PROMPT + f"\n\nQuestion:\nUsing the numbers {item['nums']}, create an equation that equals {item['target']}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>.\n\nAnswer:"

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
        "numbers": item["nums"],
        "samples": samples,
        "target": item["target"],
    }
    save_yaml(outpath, out)


@pydra.main(GenerateScriptConfig)
def main(config: GenerateScriptConfig):
    model = vllm.LLM(
        model=config.model,
        dtype="auto",
        trust_remote_code=True,
    )

    test_path = Path("dataset/countdown/test.jsonl")

    test_dataset = load_jsonl(test_path)

    print(f"Number of train items: {len(test_dataset)}")

    random.seed(config.seed)
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
