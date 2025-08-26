from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer


@dataclass
class Task:
    sample_path: Path
    save_path: Path


class RMPipeline:
    def __init__(
        self,
        model_id,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        truncation=True,
        trust_remote_code=False,
        max_length=4096,
    ):
        raise NotImplementedError(
            "RMPipeline is an abstract class, please use ArmoRMPipeline or SkyworkRMPipeline instead."
        )

    def __call__(
        self,
        question: str,
        answer: str,
    ) -> float:
        raise NotImplementedError(
            "RMPipeline is an abstract class, please use ArmoRMPipeline or SkyworkRMPipeline instead."
        )


class ArmoRMPipeline(RMPipeline):
    def __init__(
        self,
        model_id,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        truncation=True,
        trust_remote_code=False,
        max_length=4096,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(
        self,
        question: str,
        answer: str,
    ) -> float:
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
            # multi_obj_rewards = output.rewards.cpu().float()
            # helpsteer_correctness = multi_obj_rewards[0][1].item()
            # helpsteer_coherence = multi_obj_rewards[0][2].item()
            # score = helpsteer_correctness * 0.5 + helpsteer_coherence * 0.5
        return score


class SkyworkRMPipeline(RMPipeline):
    def __init__(
        self,
        model_id="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        truncation=True,
        trust_remote_code=False,
        max_length=4096,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            num_labels=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(
        self,
        question: str,
        answer: str,
    ) -> float:
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            tokenize=True,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.logits[0][0].float().item()
        score = torch.sigmoid(torch.tensor(score)).item()
        return score


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
