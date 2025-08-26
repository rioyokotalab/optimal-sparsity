# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    ##############
    # General
    ##############
    optimizer: str = 'adam'
    """Optimizer to use (one of Adam or SGD)."""

    lr: Optional[float] = None
    """Initial learning rate. Depending on decay style and initial warmup, the learning rate at each
       iteration would be different.
    """

    min_lr: Optional[float] = None
    """Minumum value for learning rate. The scheduler clip values below this threshold."""

    decoupled_lr: Optional[float] = None
    """Separate learning rate for the input and output layer."""

    decoupled_min_lr: Optional[float] = None
    """Minimum value for learning rate for the input and output layer. The scheduler clip values
       below this threshold.
    """

    weight_decay: float = 0.01
    """Weight decay coefficient for L2 regularization."""

    ##############
    # Component-wise Learning Rate Scaling
    ##############
    lr_scale_final_layernorm: float = 1.0
    """Learning rate scaling factor for final layer normalization."""

    lr_scale_expert: float = 1.0
    """Learning rate scaling factor for expert layers."""

    lr_scale_shared_expert: float = 1.0
    """Learning rate scaling factor for shared expert layers."""

    lr_scale_gate: float = 1.0
    """Learning rate scaling factor for gating/router layers."""

    lr_scale_pre_layernorm: float = 1.0
    """Learning rate scaling factor for pre-MLP layer normalization."""

    lr_scale_attention_proj: float = 1.0
    """Learning rate scaling factor for attention projection layer."""

    lr_scale_qkv_layernorm: float = 1.0
    """Learning rate scaling factor for QKV layer normalization."""

    lr_scale_qkv: float = 1.0
    """Learning rate scaling factor for QKV transformation layer."""

    lr_scale_embedding: float = 1.0
    """Learning rate scaling factor for embedding layer."""

    lr_scale_final_linear: float = 1.0
    """Learning rate scaling factor for final output layer."""

    ##############
    # Precision
    ##############
    fp16: bool = False
    """If true, train with fp16 mixed precision training. Defaults to False."""

    bf16: bool = False
    """If true, train with bf16 mixed precision training. Defaults to False."""

    params_dtype: torch.dtype = torch.float32
    """dtype used when intializing the weights. Defaults to torch.float32."""

    ###############
    # Loss scaling
    ###############
    loss_scale: Optional[float] = None
    """Static loss scaling, positive power of 2 values can improve fp16 convergence. If None,
       dynamic loss scaling is used.
    """

    initial_loss_scale: float = 2**32
    """Initial loss-scale for dynamic loss scaling."""

    min_loss_scale: float = 1.0
    """Minimum loss scale for dynamic loss scaling."""

    loss_scale_window: float = 1000
    """Window over which to raise/lower dynamic scale."""

    hysteresis: int = 2
    """Hysteresis for dynamic loss scaling."""

    ##############
    # Optimizer
    ##############
    # Adam
    adam_beta1: float = 0.9
    """First coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_beta2: float = 0.999
    """Second coefficient for computing running averages of gradient and its square in Adam
    optimizer.
    """

    adam_eps: float = 1e-08
    """Term added to the denominator to improve numerical stability in Adam optimizer."""

    # SGD.
    sgd_momentum: float = 0.9
    """Momentum factor for SGD optimizer."""

    #######################
    # Distributed optimizer
    #######################
    use_distributed_optimizer: bool = False
    """Distribute optimizer state over data-parallel replicas."""

    overlap_param_gather_with_optimizer_step: bool = False
    """If true, overlap param all-gather of first bucket with optimizer step."""

    ################
    # Miscellaneous
    ################
    clip_grad: float = 1.0
    """Gradient clipping based on global L2 norm."""

    log_num_zeros_in_grad: bool = False
    """If true, calculate and log the number of zeros in gradient."""

    barrier_with_L1_time: bool = False
    """If true, use barrier with level 1 time measurements."""

    timers: Callable = None
    """Function to get timers."""

    config_logger_dir: str = ""
    """When non-empty, dumps entry-point configs to config_logger_dir"""
