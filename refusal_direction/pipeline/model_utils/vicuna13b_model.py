import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import Tensor
from jaxtyping import Float
from pipeline.model_utils.model_base import ModelBase

class Vicuna13BModel(ModelBase):
    """
    针对 lmsys/vicuna-13b-v1.5 的具体实现。

    - 底座：Llama 2 架构（Hugging Face 上用 LlamaForCausalLM）
    - tokenizer: LlamaTokenizer（无 chat_template，自行加简单的 [INST] ... [/INST] 模板）
    """

    def __init__(self, model_name_or_path: str = "lmsys/vicuna-13b-v1.5"):
        super().__init__(model_name_or_path)

        # Vicuna tokenizer 默认没有 pad_token，这里把 pad_token 对齐到 eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # 一般做生成建议用 left padding，方便批量推理
        self.tokenizer.padding_side = "left"

    # ========== 模型 / tokenizer 加载 ==========

    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        """
        加载 Vicuna-13B v1.5 模型（Llama 架构）。
        这里用 device_map="auto" + torch_dtype="auto"，你可以按需改成 CPU / 单卡等。
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()
        return model

    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        """
        加载 Vicuna 的 Llama tokenizer。
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,  # 官方 tokenizer_config.json 就是 LlamaTokenizer（非 fast）
        )
        return tokenizer

    # ========== 指令 tokenization & 特殊 token ==========

    def _get_tokenize_instructions_fn(self):
        """
        Vicuna 没有在 tokenizer 里内置 chat_template，这里手动加一个简化版 Llama2/Vicuna 模板：

            <s>[INST] {instruction} [/INST]

        多轮对话如果你需要，可以自己在上层拼好整段 string，这里就只当成一条“用户指令”。
        """

        def tokenize_instructions(instructions: list[str]):
            prompts = [
                f"[INST] {inst.strip()} [/INST]"
                for inst in instructions
            ]
            return self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            )

        return tokenize_instructions

    def _get_eoi_toks(self):
        """
        End-of-instruction：这里用 [/INST] 的 token 序列。
        如果后面你只想用一个 ID，当成“看到这个就算 prompt 结束”，
        可以取 encode 结果的最后一个。
        """
        toks = self.tokenizer.encode("[/INST]", add_special_tokens=False)
        return toks

    def _get_refusal_toks(self):
        """
        Refusal tokens：Vicuna(Llama2) 常见拒绝起始是 "I'm sorry", "I cannot" 等。
        为简单起见，用 "I" 的 token id 作为拒绝起始 token。
        """
        toks = self.tokenizer.encode("I", add_special_tokens=False)
        if len(toks) == 0:
            return []
        return [toks[0]]

    # ========== 模型结构访问接口（blocks / attn / mlp / o_proj） ==========

    def _get_model_block_modules(self):
        """
        Llama 架构：
            self.model  -> LlamaForCausalLM
            self.model.model -> LlamaModel
            self.model.model.layers -> ModuleList[LlamaDecoderLayer]
        """
        return self.model.model.layers

    def _get_attn_modules(self):
        """
        每层的 self_attn 模块列表。
        """
        return [block.self_attn for block in self.model.model.layers]

    def _get_mlp_modules(self):
        """
        每层的 mlp 模块列表（LlamaMLP）。
        """
        return [block.mlp for block in self.model.model.layers]

    def _get_o_proj_modules(self):
        """
        每层 attention 输出投影 o_proj。
        """
        return [block.self_attn.o_proj for block in self.model.model.layers]

    # ========== 权重正交化（用于“从权重中移除拒绝方向”类实验） ==========

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        """
        思路：对每一层的 mlp.down_proj 做线性变换，使得输出在给定 direction 上没有分量。

        对于 Llama：
            y = W x     (W: [d_model, d_intermediate])
        希望对任意 x 都有：
            direction^T y = 0
        等价于：
            direction^T W = 0
        做法：
            W' = (I - u u^T) W，其中 u = direction / ||direction||
        """

        # 先归一化到 unit vector
        unit_dir = direction / (direction.norm() + 1e-8)

        def orthogonalize_fn(model: AutoModelForCausalLM | None = None):
            target_model = model if model is not None else self.model
            d_vec = unit_dir.to(target_model.device)

            for block in target_model.model.layers:
                mlp = block.mlp
                if not hasattr(mlp, "down_proj"):
                    continue

                W = mlp.down_proj.weight  # [d_model, d_intermediate]
                # comp = u (u^T W)
                # u: [d_model, 1], (u^T W): [1, d_intermediate]
                u = d_vec.view(-1, 1)
                comp = u @ (u.T @ W)
                with torch.no_grad():
                    W -= comp

        return orthogonalize_fn

    # ========== 激活加法（activation intervention） ==========

    def _get_act_add_mod_fn(
        self,
        direction: Float[Tensor, "d_model"],
        coeff: float,
        layer: int,
    ):
        """
        返回一个 forward hook，用于在指定层的 MLP 输出上添加方向向量：
            output = output + coeff * direction

        用法示例（外部）：
            hook_fn = model._get_act_add_mod_fn(direction, coeff=-1.0, layer=16)
            mlp_module = model.model_mlp_modules[15]  # layer 从 1 开始，这里是第 16 层
            fwd_hooks = [(mlp_module, hook_fn)]
        """
        # 预先处理好方向向量
        d_vec = (coeff * direction / (direction.norm() + 1e-8)).to(self.model.device)

        # 注意：这里返回的是 *forward hook*，签名为 (module, input, output)
        def hook_fn(module, input, output):
            # output: [batch, seq, d_model]
            if not isinstance(output, torch.Tensor):
                return output
            # 广播到 batch & seq
            return output + d_vec.view(1, 1, -1)

        return hook_fn
