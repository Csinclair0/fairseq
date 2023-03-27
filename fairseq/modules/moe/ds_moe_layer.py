import torch 
import typing
import logging 
from torch import Tensor
from deepspeed.moe.layer import MoE as DsMoE
from typing import Callable, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)
class MoE(DsMoE):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 ep_size=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 noisy_gate_policy: typing.Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False,
                 enable_expert_tensor_parallelism: bool = False):
        """Initialize an MoE layer.

        Arguments:
            hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
            expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
            num_experts (int, optional): default=1, the total number of experts per layer.
            ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
            k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
            capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
            eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
            min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
            use_residual (bool, optional): default=False, make this MoE layer a Residual MoE (https://arxiv.org/abs/2201.05596) layer.
            noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
            drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).
            use_rts (bool, optional): default=True, whether to use Random Token Selection.
            use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
            enable_expert_tensor_parallelism (bool, optional): default=False, whether to use tensor parallelism for experts
        """

        super(MoE, self).__init__(hidden_size, 
                                  expert, 
                                  num_experts,
                                   ep_size, 
                                   k, 
                                   capacity_factor,
                                   eval_capacity_factor, 
                                   min_capacity, 
                                   use_residual, 
                                   noisy_gate_policy, 
                                   drop_tokens, 
                                   use_rts, 
                                   use_tutel, 
                                   enable_expert_tensor_parallelism)

    def forward(self, *input: Tensor, input_padding_mask=None, used_token = None, prefix_tokens=None, 
        encoder_embeddings: Optional[Tensor]=None, **kwargs: Any):
        input = input[0]
        input_shape = list(input.shape)
        if input_padding_mask is not None:
            assert (
                len(input_padding_mask.shape) == 2
            ), "input Tensor must have dimensions: (s)equence, (t)oken"
            assert input_padding_mask.shape[0] == input.shape[0]
            assert input_padding_mask.shape[1] == input.shape[1]
    
        expected_bsz = (
            getattr(self.args, "batch_size", 0)
            if self.training
            else getattr(self.args, "batch_size_valid", 0)
        )
        if expected_bsz is None:
            expected_bsz = 0
            
        if (
            not self.in_generation
            and expected_bsz != 0
            and input_shape[0] != expected_bsz
        ):
            logger.warning(
                f"padding batch with unexpected size {input_shape[0]} (expected: {expected_bsz})"
            )
            assert input_shape[0] < expected_bsz, f"{input_shape[0]} < {expected_bsz}"
            padded_input = torch.zeros(
                (expected_bsz, input_shape[1], input_shape[2]),
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            padded_input[: input_shape[0], :, :] = input
            input = padded_input

        output = self.deepspeed_moe(input, used_token)
        if self.use_residual:
            # Residual MoE
            output_mlp = self.mlp(input[0])
            if type(output_mlp) is tuple:
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(input[0])
            coef = torch.nn.functional.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        self.metadata =  { "moe_gate_loss" : self.deepspeed_moe.l_aux }
        return output, { "moe_gate_loss" : self.deepspeed_moe.l_aux }