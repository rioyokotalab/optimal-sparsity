# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

import argparse
import json
import os
import re
import sys
import types
from collections import OrderedDict
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from huggingface_hub import save_torch_state_dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, MixtralConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME


@torch.inference_mode()
def clone_state_dict(elem):
    """clone all tensors in the elem to cpu device."""
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        elem = elem.clone()
    elif isinstance(elem, (np.ndarray, str)):
        pass
    elif isinstance(elem, Mapping):
        elem = dict(elem)
        for k, v in elem.items():
            elem[k] = clone_state_dict(v)
        elem = elem_type(elem)
    elif isinstance(elem, Sequence):
        elem = list(elem)
        for i in range(len(elem)):
            elem[i] = clone_state_dict(elem[i])
        elem = elem_type(elem)
    return elem


def add_args(parser):
    parser.add_argument(
        '--megatron-path', type=str, default=None, help='Base directory of Megatron repository'
    )

    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_path", type=str, required=True, help="Path to the checkpoint to convert."
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to the converted checkpoint."
    )

    parser.add_argument("--world_size", type=int, default=1, help=("world_size"))

    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument(
        "--target_expert_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    parser.add_argument("--print-checkpoint-structure", action="store_true")

    return parser


megatron_to_transformers = {
    "self_attention.linear_proj": "self_attn.o_proj",
    "mlp.router": "block_sparse_moe.gate",
}

tensor_parallel_params_mg = [
    # megatron-lm layers to merge across tp ranks
    'self_attention.linear_proj.weight',
    'self_attention.linear_qkv.weight',
]

column_split_tensor_parallel_params_mg = ['self_attention.linear_proj']


def get_checkpoint_sub_dir_name(tp_rank, pp_rank, pp_size, ep_rank, ep_size):
    sub_dir_name = f"mp_rank_{tp_rank:02d}"
    if pp_size > 1:
        sub_dir_name = f"{sub_dir_name}_{pp_rank:03d}"
    if ep_size > 1:
        sub_dir_name = f"{sub_dir_name}_{ep_rank:03d}"
    return sub_dir_name


def get_megatron_sharded_states(args, tp_size, pp_size, ep_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.
    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = [{'model': {}} for i in range(tp_size)]
    for tp_index, i in enumerate(range(tp_size)):
        for ep_index, j in enumerate(range(ep_size)):
            sub_dir_name = get_checkpoint_sub_dir_name(i, pp_rank, pp_size, j, ep_size)
            print(f"Loading {sub_dir_name}...")
            # Since distrib_optim.pt is unnecessary, explicitly specify model_optim_rng.pt instead.
            checkpoint_path = os.path.join(args.load_path, sub_dir_name, 'model_optim_rng.pt')
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Could not find model_optim_rng.pt in {os.path.join(args.load_path, sub_dir_name)}. "
                    f"Available files: {os.listdir(os.path.join(args.load_path, sub_dir_name))}"
                )

            print(f"Loading checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            print("checkpoint_path", checkpoint_path)
            ep_length = len(
                [
                    i
                    for i in state_dict['model']
                    if 'mlp.experts.linear_fc2.weight' in i and 'decoder.layers.0' in i
                ]
            )
            # combine experts within a tensor_parallel
            for key, value in list(state_dict['model'].items()):
                if 'linear_fc' in key and 'weight' in key and not '_extra_state' in key:
                    key_list = key.split('.')
                    local_ep_index = int(key_list[-1][6:])
                    key_list[-1] = f"weight{ep_index * ep_length + local_ep_index}"
                    new_key = '.'.join(key_list)
                    del state_dict['model'][key]
                    state_dict['model'][new_key] = value
            tp_state_dicts[tp_index]['model'].update(state_dict['model'])
    return tp_state_dicts


def megatron_to_transformers_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :] for compatibility with later versions
    of NVIDIA Megatron-LM. The inverse operation is performed inside Megatron-LM to read checkpoints:
    https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209 If param is the weight tensor of the
    self-attention block, the returned tensor will have to be transposed one more time to be read by HuggingFace GPT2.
    This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def transformers_to_megatron_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to the one compatible with respective NVIDIA Megatron-LM chekpoint versions. Input
    is [num_splits * num_heads * hidden_size, :] and output is [num_heads * hidden_size * num_splits, :] for version
    1.0 and [num_heads * num_splits * hidden_size, :] for version 2.0 and later. If param is the weight tensor of the
    self-attention block, the param needs to be already transposed before calling this function.
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # Input is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`
    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def get_element_from_dict_by_path(d, path):
    if path not in d:
        d[path] = {}
    d = d[path]
    return d


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert NVIDIA Megatron-LM checkpoint to HuggingFace Transformers checkpoint. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality. This greatly extends the functionality of
    `convert_megatron_gpt2_checkpoint.py`

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    # Load Megatron-LM checkpoint arguments from the state dict
    possible_state_paths: list[str] = [os.path.join(args.load_path)]
    print(f"DEBUG: possible_state_paths: {possible_state_paths}")
    state_path = None
    for p in possible_state_paths:
        if os.path.exists(p):
            state_path = p
            print(f"Loading Megatron-LM checkpoint arguments from: {state_path}")
            break
    assert state_path is not None, f"Cannot find state path in {possible_state_paths}"
    possible_sub_dirs = [
        "mp_rank_00",
        "mp_rank_00_000",
        "mp_rank_00_dp_000",
        "mp_rank_00_000_dp_000",
        "mp_rank_00_000_000",
    ]

    state_dirs = os.listdir(state_path)
    for sub_dir in possible_sub_dirs:
        if sub_dir in state_dirs:
            rank0_checkpoint_path = os.path.join(state_path, sub_dir, 'model_optim_rng.pt')
            break
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")  # type: ignore
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")  # type: ignore
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint instead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # Create Transformers GPT2 config from Megatron-LM arguments
    vocab_size = megatron_args.padded_vocab_size

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    config = MixtralConfig(
        architectures=["MixtralForCausalLM"],
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=megatron_args.hidden_size,
        initializer_range=megatron_args.init_method_std,
        intermediate_size=megatron_args.ffn_hidden_size,
        max_position_embeddings=megatron_args.seq_length,
        mlp_bias=False,
        model_type='mixtral',
        num_attention_heads=megatron_args.num_attention_heads,
        num_experts_per_tok=megatron_args.moe_router_topk,
        num_hidden_layers=megatron_args.num_layers,
        num_key_value_heads=megatron_args.num_query_groups,
        num_local_experts=megatron_args.num_experts,
        output_router_logits=False,
        rms_norm_eps=megatron_args.norm_epsilon,
        rope_theta=megatron_args.rotary_base,
        router_aux_loss_coef=0.01,
        tie_word_embeddings=False,
        torch_dtype=dtype,
        use_cache=True,
    )

    output_state_dict = {}

    # checkpoint_version = state_dict.get("checkpoint_version", 3.0)
    tp_size = args.target_tensor_model_parallel_size
    pp_size = args.target_pipeline_model_parallel_size
    ep_size = args.target_expert_model_parallel_size

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Convert.
    print("Converting")

    # Embeddings
    print("Converting embeddings")
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, ep_size, 0)

    # import pdb
    # pdb.set_trace()

    # Convert and store the word embeddings.
    word_embeddings = []

    # import pdb
    # pdb.set_trace()
    embeddings = tp_state_dicts[0]["model"]["embedding.word_embeddings.weight"]
    for tp_rank in range(tp_size):
        embeddings = tp_state_dicts[tp_rank]["model"]["embedding.word_embeddings.weight"]
        word_embeddings.append(embeddings)

    word_embeddings = torch.cat(word_embeddings, dim=0)
    word_embeddings = word_embeddings.to(dtype)
    output_state_dict["model.embed_tokens.weight"] = word_embeddings.clone()
    # Reset the vocab size
    config.vocab_size = word_embeddings.shape[0]

    # Transformer Layers
    print("Converting transformer layers")
    # The number of heads.
    heads = config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers // pp_size

    hidden_size = config.hidden_size
    num_groups = config.num_key_value_heads

    for pp_rank in range(pp_size):
        if pp_size > 0:
            print(f"Converting pipeline parallel rank {pp_rank}")
            tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, ep_size, pp_rank)

        # The transformer.

        path = 'model'

        # Extract the layers.
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            if key.endswith('_extra_state'):
                continue
            # deal with experts
            if 'linear_fc' in key:
                print(f"Processing expert key: {key}")
                key_list = key.split('.')
                layer_id = int(key_list[2]) + pp_rank * num_layers
                expert_id = int(key_list[-1][6:])  # Extracts number from 'weightN'
                print(f"Layer ID: {layer_id}, Expert ID: {expert_id}")
                dim = 1 if 'linear_fc2' in key else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

                if 'linear_fc2' in key:
                    output_state_dict[
                        f'model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w2.weight'
                    ] = params
                else:
                    params_split = [torch.chunk(i, 2, 0) for i in torch.chunk(params, tp_size, 0)]
                    output_state_dict[
                        f'model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w1.weight'
                    ] = torch.cat([i[0] for i in params_split])
                    output_state_dict[
                        f'model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w3.weight'
                    ] = torch.cat([i[1] for i in params_split])

                continue

            new_key = key.replace('decoder.', '')
            if 'layer_norm_weight' in new_key:
                new_key += '.weight'
            # Match the name.
            m = layer_re.match(new_key)
            # Stop if that's not a layer
            if m is None:
                continue

            # The index of the layer.
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)

            # The name of the layer.
            layer_name = f"model.layers.{layer_idx}"

            print(layer_name, op_name, weight_or_bias)

            if op_name + "." + weight_or_bias not in tensor_parallel_params_mg:
                params = val.to(dtype)
            else:
                dim = 1 if op_name in column_split_tensor_parallel_params_mg else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("layer_norm_weight") or op_name.endswith("layernorm"):
                ln_name = (
                    "input_layernorm"
                    if op_name.endswith("layer_norm_weight")
                    else "post_attention_layernorm"
                )
                output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = (
                    params.clone()
                )
                continue

            # Transpose the QKV matrix.
            elif (
                op_name == "attention.linear_qkv" or op_name == "self_attention.linear_qkv"
            ) and weight_or_bias == "weight":

                all_qkvs = [
                    i.reshape(
                        num_groups // args.target_tensor_model_parallel_size,
                        (heads // num_groups * hidden_size_per_head + 2 * hidden_size_per_head),
                        hidden_size,
                    )
                    for i in torch.chunk(params, args.target_tensor_model_parallel_size, 0)
                ]
                split_size = heads // num_groups * hidden_size_per_head
                all_qs = torch.cat(
                    [i[:, :split_size, :].reshape(-1, hidden_size) for i in all_qkvs]
                )
                all_kvs = torch.cat(
                    [i[:, split_size:, :].reshape(-1, hidden_size) for i in all_qkvs]
                )

                checkpoint_version = 3.0
                out_q = megatron_to_transformers_fix_query_key_value_ordering(
                    all_qs, checkpoint_version, 1, heads, hidden_size_per_head
                )

                out_kv = megatron_to_transformers_fix_query_key_value_ordering(
                    all_kvs, checkpoint_version, 2, num_groups, hidden_size_per_head
                )
                out_kv = torch.chunk(out_kv, 2)

                output_state_dict[layer_name + f".self_attn.q_proj.weight"] = out_q.clone()
                output_state_dict[layer_name + f".self_attn.k_proj.weight"] = out_kv[0].clone()
                output_state_dict[layer_name + f".self_attn.v_proj.weight"] = out_kv[1].clone()

            # Transpose the weights.
            elif weight_or_bias == "weight":
                out_name = megatron_to_transformers[op_name]
                output_state_dict[layer_name + '.' + out_name + '.' + "weight"] = params.clone()

    if config.num_hidden_layers != (layer_idx + 1):
        raise ValueError(f"Expected {config.num_hidden_layers} layers but found {layer_idx + 1}")

    # The final layernorm.
    print("Converting final layernorm")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    try:
        output_state_dict["model.norm.weight"] = (
            params["decoder.final_layernorm.weight"].to(dtype).clone()
        )
    except:
        output_state_dict["model.norm.weight"] = (
            params["decoder.final_norm.weight"].to(dtype).clone()
        )

    # For LM head, transformers' wants the matrix to weight embeddings.
    print("Converting LM head")
    params = torch.cat(
        [
            get_element_from_dict_by_path(tp_state_dicts[i]['model'], 'output_layer.weight')
            for i in range(tp_size)
        ]
    )
    output_state_dict["lm_head.weight"] = params.to(dtype).clone()

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    print("Saving checkpoint...")
    config.save_pretrained(args.save_path)
    save_torch_state_dict(
        state_dict=output_state_dict, save_directory=args.save_path, safe_serialization=True
    )
    print(f"Model weights saved in {args.save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
