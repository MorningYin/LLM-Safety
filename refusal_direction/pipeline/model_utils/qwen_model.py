import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Optional
from jaxtyping import Int, Float

from refusal_direction.pipeline.utils.utils import get_orthogonalized_matrix
from refusal_direction.pipeline.model_utils.model_base import ModelBase

# --- 常量定义 ---

# 默认的英文 system prompt，可以根据需要改成别的语言
DEFAULT_SYSTEM_PROMPT_EN = (
    "You are a helpful, honest and safe assistant. "
    "Always answer in English."
)

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE_NO_SYSTEM = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# --- 辅助函数 ---

def format_instruction_qwen_chat(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
) -> str:
    """
    构造 Qwen Chat 风格的 prompt 文本。
    """
    if system is not None:
        formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(
            instruction=instruction,
            system=system,
        )
    else:
        formatted_instruction = QWEN_CHAT_TEMPLATE_NO_SYSTEM.format(
            instruction=instruction,
        )

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    """
    将一批 instruction 按 Qwen Chat 模板拼成 prompt 并 tokenize。
    """
    if outputs is not None:
        prompts = [
            format_instruction_qwen_chat(
                instruction=instruction,
                output=output,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen_chat(
                instruction=instruction,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
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
    if hasattr(model, "transformer"):
        # Qwen1 架构
        return model.transformer, "h", "wte"
    elif hasattr(model, "model"):
        # Qwen2 架构 (Llama-like)
        return model.model, "layers", "embed_tokens"
    else:
        raise ValueError(f"无法识别 Qwen 模型架构: {type(model)}")


def _get_layer_modules(layer) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    输入一个 transformer layer block，返回 (attn_output_module, mlp_output_module)
    兼容 Qwen1 和 Qwen2
    """
    # 1. 识别 Attention 输出层
    if hasattr(layer, "attn"):
        # Qwen1
        attn_out = layer.attn.c_proj
    elif hasattr(layer, "self_attn"):
        # Qwen2
        attn_out = layer.self_attn.o_proj
    else:
        raise ValueError("Unknown attention mechanism")

    # 2. 识别 MLP 输出层
    if hasattr(layer, "mlp"):
        mlp_module = layer.mlp
        if hasattr(mlp_module, "c_proj"):
            # Qwen1
            mlp_out = mlp_module.c_proj
        elif hasattr(mlp_module, "down_proj"):
            # Qwen2
            mlp_out = mlp_module.down_proj
        else:
            raise ValueError("Unknown MLP output projection")
    else:
        raise ValueError("Layer has no MLP")

    return attn_out, mlp_out


# --- 核心操作函数 ---

def orthogonalize_qwen_weights(
    model,
    direction: Float[Tensor, "d_model"],
):
    model_base, layers_attr, embed_attr = _get_qwen_model_attr(model)

    # 1. 处理 Embedding
    embed = getattr(model_base, embed_attr)
    embed.weight.data = get_orthogonalized_matrix(embed.weight.data, direction)

    # 2. 处理各层 (Attention Out 和 MLP Out)
    layers = getattr(model_base, layers_attr)
    for block in layers:
        attn_out, mlp_out = _get_layer_modules(block)

        # 注意：通常 PyTorch Linear 是 xW^T，所以这里转置处理是正确的
        attn_out.weight.data = get_orthogonalized_matrix(
            attn_out.weight.data.T, direction
        ).T
        mlp_out.weight.data = get_orthogonalized_matrix(
            mlp_out.weight.data.T, direction
        ).T


def act_add_qwen_weights(
    model,
    direction: Float[Tensor, "d_model"],
    coeff: float,
    layer: int,
):
    model_base, layers_attr, _ = _get_qwen_model_attr(model)
    layers = getattr(model_base, layers_attr)

    # 获取目标层（1-based 索引）
    target_layer = layers[layer - 1]
    _, mlp_out = _get_layer_modules(target_layer)

    dtype = mlp_out.weight.dtype
    device = mlp_out.weight.device

    bias_vec = (coeff * direction).to(dtype=dtype, device=device)

    # Qwen2 的 Linear 层可能没有 bias (bias=None)
    if mlp_out.bias is None:
        # 如果模型原始就没有 bias，我们需要手动注册一个 Parameter
        mlp_out.bias = torch.nn.Parameter(bias_vec)
    else:
        mlp_out.bias = torch.nn.Parameter(mlp_out.bias + bias_vec)  # 叠加


# --- 模型类 ---

class QwenModel(ModelBase):
    """
    更鲁棒版本：
    - 使用 system prompt 强制指定语言（默认英文）
    - 兼容 Qwen1 / Qwen2 架构
    - 在生成阶段禁用外部 attention_mask，以避免 scaled_dot_product_attention 的维度错误
    """

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.float16,
        target_language: str = "English",
        system_prompt: Optional[str] = None,
    ):
        """
        参数:
            model_path: 模型路径或名称
            dtype: 模型 dtype（默认 fp16，可以换成 bfloat16）
            target_language: 目标回答语言，用于构造默认 system prompt
            system_prompt: 自定义 system prompt，若不为 None 则优先使用
        """
        self._dtype = dtype
        self._target_language = target_language
        self._custom_system_prompt = system_prompt
        super().__init__(model_path)

    # ------- 模型 / tokenizer 加载 -------

    def _load_model(self, model_path, dtype: torch.dtype = None):
        if dtype is None:
            dtype = self._dtype

        model_kwargs = {}

        # Flash Attention 逻辑
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            model_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            **model_kwargs,
        ).eval()

        model.requires_grad_(False)

        # ---- 关键修复：覆盖 prepare_inputs_for_generation，禁用外部 attention_mask ----
        import types

        original_prepare = model.prepare_inputs_for_generation

        def fixed_prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            inputs_embeds=None,
            **kwargs,
        ):
            # 有 past_key_values 时，只喂最后一个 token（标准增量生成）
            if past_key_values is not None:
                input_ids = input_ids[:, -1].unsqueeze(-1)

            # 关键：生成阶段不再传 attention_mask，避免传入给 sdpa 出现维度不兼容
            attention_mask = None

            if inputs_embeds is not None and past_key_values is None:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids}

            model_inputs.update(
                {
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache", True),
                    "attention_mask": attention_mask,
                }
            )
            return model_inputs

        model.prepare_inputs_for_generation = types.MethodType(
            fixed_prepare_inputs_for_generation, model
        )

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # 兼容不同词表的特殊 token
                for token in ["<|endoftext|>", "<|extra_0|>", "<|im_end|>"]:
                    ids = tokenizer.convert_tokens_to_ids(token)
                    if ids != tokenizer.unk_token_id:
                        tokenizer.pad_token = token
                        tokenizer.pad_token_id = ids
                        break

                # 如果还是找不到，就用 0
                if tokenizer.pad_token is None:
                    tokenizer.pad_token_id = 0

        return tokenizer

    # ------- system prompt & tokenization -------

    def _get_system_prompt(self) -> str:
        """
        返回最终使用的 system prompt：
        - 若用户传了自定义 system_prompt，就用它；
        - 否则根据 target_language 生成一个简单的英文模板。
        """
        if self._custom_system_prompt is not None:
            return self._custom_system_prompt

        return (
            "You are a helpful, honest and safe assistant. "
            f"Always answer in {self._target_language}."
        )

    def _get_tokenize_instructions_fn(self):
        # 使用带 system prompt 的 Qwen Chat 模板
        system_prompt = self._get_system_prompt()
        return functools.partial(
            tokenize_instructions_qwen_chat,
            tokenizer=self.tokenizer,
            system=system_prompt,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        # 动态获取 instruction 结束符（基于带 system 的模板）
        part = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.split("{instruction}")[-1]
        return self.tokenizer.encode(part, add_special_tokens=False)

    def _get_refusal_toks(self):
        # 动态获取拒绝起始 token，如 "I", "As", "Sorry" 等
        refusal_candidates = ["I", "As", "Sorry", "I'm", "However"]
        toks = []
        for word in refusal_candidates:
            t = self.tokenizer.encode(word, add_special_tokens=False)
            if len(t) > 0:
                toks.append(t[0])
        return list(set(toks))

    # ------- 模型结构访问（用于 hook / 干预） -------

    def _get_model_block_modules(self):
        # 兼容 Qwen1 和 Qwen2
        if hasattr(self.model, "transformer"):
            return self.model.transformer.h
        elif hasattr(self.model, "model"):
            return self.model.model.layers
        else:
            raise ValueError(f"无法识别 Qwen 模型架构: {type(self.model)}")

    def _get_attn_modules(self):
        # Qwen1: .attn, Qwen2: .self_attn
        modules = []
        for block in self.model_block_modules:
            if hasattr(block, "attn"):
                modules.append(block.attn)
            elif hasattr(block, "self_attn"):
                modules.append(block.self_attn)
        return torch.nn.ModuleList(modules)

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            [block_module.mlp for block_module in self.model_block_modules]
        )

    def _get_o_proj_modules(self):
        # Qwen1: attn.c_proj, Qwen2: self_attn.o_proj
        modules = []
        for block in self.model_block_modules:
            attn_out, _ = _get_layer_modules(block)
            modules.append(attn_out)
        return torch.nn.ModuleList(modules)

    # ------- 干预操作接口 -------

    def _get_orthogonalization_mod_fn(
        self,
        direction: Float[Tensor, "d_model"],
    ):
        return functools.partial(orthogonalize_qwen_weights, direction=direction)

    def _get_act_add_mod_fn(
        self,
        direction: Float[Tensor, "d_model"],
        coeff,
        layer,
    ):
        return functools.partial(
            act_add_qwen_weights,
            direction=direction,
            coeff=coeff,
            layer=layer,
        )
