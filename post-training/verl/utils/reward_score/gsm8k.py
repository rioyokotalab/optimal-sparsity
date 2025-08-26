# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import os
import re
import signal
from argparse import ArgumentParser
from pathlib import Path

import jsonlines
import numpy as np
import torch
from math_verify import parse, verify
from tqdm import tqdm


def handler(signum, frame):
    raise TimeoutError("Execution timed out!")


def execute_function(code: str, timeout=3):
    try:
        # Set the alarm handler
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)  # Start the alarm
        local_namespace = {}
        exec(code, {}, local_namespace)
        return str(local_namespace["simple_math_problem"]())
    except TimeoutError as e:
        return None
    except Exception:
        return None
    finally:
        # Always disable the alarm after execution
        signal.alarm(0)


def execute_tinygsm_code(text):
    code = text.split("\ndef")[-1]
    code = "def" + code
    try:
        return execute_function(code)
    except:
        return None


def execute_llm_code(text):
    try:
        # Extract code inside <llm-code> tags
        code_match = re.search(r"<llm-code>(.*?)</llm-code>", text, re.DOTALL)
        if not code_match:
            return None

        code = code_match.group(1).strip()

        # Create a dictionary for execution context
        exec_globals = {}

        # Split the code into lines and execute it
        lines = code.split("\n")
        last_expr = lines[-1]  # The last line of code
        timeout = 3

        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)  # Start the alarm
            exec(code, exec_globals)
        except TimeoutError as e:

            return None
        except Exception:
            return None
        finally:
            # Always disable the alarm after execution
            signal.alarm(0)

        return str(eval(last_expr, exec_globals))
    except:
        return None


def execute_code(text):
    if "<llm-code>" in text:
        code_out = execute_llm_code(text)
        return code_out
    else:
        return execute_tinygsm_code(text)


def parse_text_answer(text):
    answer = parse(text)


def get_llm_answer(text):
    response_type = "text"
    if "<llm-code>" in text:
        code_out = execute_llm_code(text)
        response_type = "llm-code"
        if code_out is not None:
            return parse(code_out), "llm-code"
    if "def" in text:
        code_out = execute_tinygsm_code(text)
        response_type = "tinygsm-code"
        if code_out is not None:
            return parse(code_out), "tinygsm-code"

    return parse(text), response_type


def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.0, score=1.0
):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    llm_answer, _ = get_llm_answer(solution_str)
    correct_answer = parse(ground_truth)
    ret = 0.0
    if verify(llm_answer, correct_answer) == True:
        ret = score
    return ret
