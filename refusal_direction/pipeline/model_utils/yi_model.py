
import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from jaxtyping import Float

from refusal_direction.pipeline.utils.utils import get_orthogonalized_matrix
from refusal_direction.pipeline.model_utils.model_base import ModelBase

# Yi chat templates are based on
# - Official tokenizer config: https://huggingface.co/01-ai/Yi-6B-Chat/blob/main/tokenizer_config.json
# - Replicate default prompt template: https://replicate.com/01-ai/yi-6b-chat

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

YI_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

YI_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|> 
<|im_start|>assistant
"""

YI_REFUSAL_TOKS = [59597] # ['I']

# Noting some other top refusal tokens. But really a vast majority of the probability is placed on the first.
YI_REFUSAL_TOKS_EXTRA = [59597, 2301, 4786] # ['I', 'It', 'As']

def format_instruction_yi_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        formatted_instruction = YI_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = YI_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_yi_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_yi_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_yi_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_yi_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_yi_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer-1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer-1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.model.layers[layer-1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class YiModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()

        model.requires_grad_(False)
        for param in model.parameters():
            param.requires_grad = False

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        tokenizer.padding_side = 'left'

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_yi_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        """获取 EOI tokens。
        
        通过实际格式化一个示例指令来获取 marker tokens，确保返回的是实际 tokenize 后的格式。
        这样可以处理 tokenizer 编码/解码不完全可逆的情况。
        """
        # 使用实际的格式化函数来获取 marker
        test_instruction = "test"
        formatted = format_instruction_yi_chat(instruction=test_instruction)
        tokens = self.tokenizer.encode(formatted, add_special_tokens=False)
        
        # 找到 instruction 部分之后的 tokens（即 marker 部分）
        # instruction tokens
        instruction_tokens = self.tokenizer.encode(test_instruction, add_special_tokens=False)
        
        # 在 tokens 中找到 instruction 的结束位置
        instruction_end_idx = None
        for i in range(len(tokens) - len(instruction_tokens) + 1):
            if tokens[i:i+len(instruction_tokens)] == instruction_tokens:
                instruction_end_idx = i + len(instruction_tokens)
                break
        
        if instruction_end_idx is not None:
            # 返回 instruction 之后的所有 tokens（即 marker tokens）
            return tokens[instruction_end_idx:]
        else:
            # 如果找不到，回退到从模板提取
            return self.tokenizer.encode(YI_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return YI_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_o_proj_modules(self):
        return torch.nn.ModuleList([block_module.self_attn.o_proj for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_yi_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_yi_weights, direction=direction, coeff=coeff, layer=layer)