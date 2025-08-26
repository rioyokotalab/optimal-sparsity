import time
import torch
from datasets import load_dataset
from tqdm import tqdm
import pydra
import random
from functools import partial
import vllm
from llmonk.generate.prompts import GSM8K_CODE_COT_PROMPT
from llmonk.utils import save_yaml, GenerateScriptConfig
import gc


def get_few_shot_prompt(item):
    few_shot_items = item["few_shot_items"]
    few_shot_pieces = []
    for f in few_shot_items:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/568af943e315100af3f00937bfd6947844769ab8/lm_eval/tasks/gsm8k/gsm8k.yaml
        few_shot_prompt = f"Question: {f['question']}\nAnswer: {f['answer']}\n\n"
        few_shot_pieces.append(few_shot_prompt)
    return "".join(few_shot_pieces)


def run_inference(item, config: GenerateScriptConfig, model, tokenizer):
    outpath = config.save_dir / f"{item['id']}.yaml"
    if outpath.exists():
        return

    prompt = GSM8K_CODE_COT_PROMPT + f"def simple_math_problem():\n    '''\n    {item['question']}\n    '''\n"

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
        "question": item["question"],
        "samples": samples,
        "gt_answer": item["answer"],
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
    tokenizer = model.get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    test_dataset = list(load_dataset("gsm8k", "main", split="test"))
    train_dataset = list(load_dataset("gsm8k", "main", split="train"))

    print(f"Number of test items: {len(test_dataset)}")
    print(f"Number of train items: {len(train_dataset)}")

    random.seed(config.seed)
    for i, data in enumerate(train_dataset):
        data["id"] = i

    for i, data in enumerate(test_dataset):
        few_shot_items = random.sample(train_dataset, config.num_few_shot)
        data["id"] = i
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
        run_inference(item, config=config, model=model, tokenizer=tokenizer)
    try:
        model._engine.shutdown_background_loop()  #  [oai_citation_attribution:0‡vLLM](https://docs.vllm.ai/en/latest/api/engine/async_llm_engine.html?utm_source=chatgpt.com)
    except Exception:
        pass

    # モデル／executor をクリーンアップ
    try:
        model.shutdown()       # LLM クラスの shutdown があれば呼び出し
    except Exception:
        pass
    del model
    gc.collect()


if __name__ == "__main__":
    main()
