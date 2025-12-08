import torch
import functools
import warnings

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Union
from jaxtyping import Int, Float

# 假设这些是你项目中的原有工具引用
from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# --- 常量定义 ---

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# --- 辅助函数 ---

def format_instruction_qwen_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = QWEN_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def _get_qwen_model_attr(model):
    """获取 Qwen 模型的属性路径（兼容 Qwen1 和 Qwen2）"""
    if hasattr(model, 'transformer'):
        # Qwen1 架构
        return model.transformer, 'h', 'wte'
    elif hasattr(model, 'model'):
        # Qwen2 架构 (Llama-like)
        return model.model, 'layers', 'embed_tokens'
    else:
        raise ValueError(f"无法识别 Qwen 模型架构: {type(model)}")

def _get_layer_modules(layer) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    输入一个 transformer layer block，返回 (attn_output_module, mlp_output_module)
    兼容 Qwen1 和 Qwen2
    """
    # 1. 识别 Attention 输出层
    if hasattr(layer, 'attn'):
        # Qwen1
        attn_out = layer.attn.c_proj
    elif hasattr(layer, 'self_attn'):
        # Qwen2
        attn_out = layer.self_attn.o_proj
    else:
        raise ValueError("Unknown attention mechanism")

    # 2. 识别 MLP 输出层
    if hasattr(layer, 'mlp'):
        mlp_module = layer.mlp
        if hasattr(mlp_module, 'c_proj'):
            # Qwen1
            mlp_out = mlp_module.c_proj
        elif hasattr(mlp_module, 'down_proj'):
            # Qwen2
            mlp_out = mlp_module.down_proj
        else:
            raise ValueError("Unknown MLP output projection")
    else:
        raise ValueError("Layer has no MLP")

    return attn_out, mlp_out

# --- 核心操作函数 ---

def orthogonalize_qwen_weights(model, direction: Float[Tensor, "d_model"]):
    model_base, layers_attr, embed_attr = _get_qwen_model_attr(model)
    
    # 1. 处理 Embedding
    embed = getattr(model_base, embed_attr)
    embed.weight.data = get_orthogonalized_matrix(embed.weight.data, direction)
    
    # 2. 处理各层 (Attention Out 和 MLP Out)
    layers = getattr(model_base, layers_attr)
    for block in layers:
        attn_out, mlp_out = _get_layer_modules(block)
        
        # 注意：通常 PyTorch Linear 是 xW^T，所以这里转置处理是正确的
        attn_out.weight.data = get_orthogonalized_matrix(attn_out.weight.data.T, direction).T
        mlp_out.weight.data = get_orthogonalized_matrix(mlp_out.weight.data.T, direction).T

def act_add_qwen_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    model_base, layers_attr, _ = _get_qwen_model_attr(model)
    layers = getattr(model_base, layers_attr)
    
    # 获取目标层
    target_layer = layers[layer-1]
    _, mlp_out = _get_layer_modules(target_layer)
    
    dtype = mlp_out.weight.dtype
    device = mlp_out.weight.device

    bias_vec = (coeff * direction).to(dtype=dtype, device=device)

    # Qwen2 的 Linear 层可能没有 bias (bias=None)
    if mlp_out.bias is None:
        # 如果模型原始就没有 bias，我们需要手动注册一个 Parameter
        # 注意：这改变了模型结构，可能会在保存/加载时产生 strict 警告，但在运行时是有效的
        mlp_out.bias = torch.nn.Parameter(bias_vec)
    else:
        mlp_out.bias = torch.nn.Parameter(mlp_out.bias + bias_vec) # 叠加

# --- 模型类 ---

class QwenModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.float16):
        model_kwargs = {}
        
        # Flash Attention 逻辑
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            try:
                model_kwargs["attn_implementation"] = "sdpa"
            except:
                model_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            **model_kwargs,
        ).eval()

        model.requires_grad_(False) 
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        tokenizer.padding_side = 'left'
        
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # 兼容不同词表的特殊 token
                for token in ['<|endoftext|>', '<|extra_0|>', '<|im_end|>']:
                    ids = tokenizer.convert_tokens_to_ids(token)
                    if ids != tokenizer.unk_token_id:
                        tokenizer.pad_token = token
                        tokenizer.pad_token_id = ids
                        break
                
                # 如果还是找不到，就用 0
                if tokenizer.pad_token is None:
                    tokenizer.pad_token_id = 0
                    
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_qwen_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        # 动态获取 instruction 结束符
        part = QWEN_CHAT_TEMPLATE.split("{instruction}")[-1]
        return self.tokenizer.encode(part, add_special_tokens=False)

    def _get_refusal_toks(self):
        # 原始代码硬编码了 Qwen1 的 ID。这里改为动态获取。
        # 常见的 Refusal 开头: "I cannot", "As an AI", "Sorry"
        # 注意：Qwen2 的 token ID 与 Qwen1 完全不同。
        
        # 尝试编码常用的拒绝词首 Token
        refusal_candidates = ["I", "As", "Sorry", "I'm", "However"]
        toks = []
        for word in refusal_candidates:
            # add_special_tokens=False 防止加入 bos/eos
            t = self.tokenizer.encode(word, add_special_tokens=False)
            if len(t) > 0:
                toks.append(t[0])
        return list(set(toks))

    def _get_model_block_modules(self):
        # 兼容 Qwen1 和 Qwen2
        if hasattr(self.model, 'transformer'):
            return self.model.transformer.h
        elif hasattr(self.model, 'model'):
            return self.model.model.layers
        else:
            raise ValueError(f"无法识别 Qwen 模型架构: {type(self.model)}")

    def _get_attn_modules(self):
        # Qwen1: .attn, Qwen2: .self_attn
        modules = []
        for block in self.model_block_modules:
            if hasattr(block, 'attn'):
                modules.append(block.attn)
            elif hasattr(block, 'self_attn'):
                modules.append(block.self_attn)
        return torch.nn.ModuleList(modules)
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_o_proj_modules(self):
        # Qwen1: attn.c_proj, Qwen2: self_attn.o_proj
        modules = []
        for block in self.model_block_modules:
            attn_out, _ = _get_layer_modules(block)
            modules.append(attn_out)
        return torch.nn.ModuleList(modules)

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_qwen_weights, direction=direction, coeff=coeff, layer=layer)