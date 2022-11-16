# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.cuda import Event as CudaEvent
from torch.nn import Module, ModuleList

from fairseq import distributed_utils
from fairseq.modules.linear import Linear

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

logger = logging.getLogger(__name__)

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe

    has_tutel = True
except (ModuleNotFoundError, AssertionError):
    # import raises AssertionError without CUDA
    has_tutel = False



class MOETaskLayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(
        self,
        experts: Union[Module, ModuleList],
        args,
    ) -> None:
        super().__init__()
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])


    def forward(self, *input: Tensor, input_padding_mask=None, prefix_tokens=None, 
        encoder_embeddings: Optional[Tensor]=None, **kwargs: Any) -> Tensor:
        expert_output_1 = self.experts[0](
            input
        )
        expert_output_2 = self.experts[1](
            input
        )
        return expert_output_1 + expert_output_2 , 0