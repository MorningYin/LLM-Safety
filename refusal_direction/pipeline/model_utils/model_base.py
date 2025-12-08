from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.hook_utils import add_hooks

class ModelBase(ABC):
    """
    模型基类，为不同模型家族（Llama、Qwen、Gemma等）提供统一的接口。
    
    这个抽象基类定义了所有模型必须实现的接口，用于：
    1. 统一不同模型的结构差异（路径、命名等）
    2. 提供统一的文本生成和激活干预接口
    3. 简化上层代码，无需关心具体模型类型
    
    所有具体模型类（如 Llama3Model、QwenModel）都必须继承此类并实现所有抽象方法。
    """
    def __init__(self, model_name_or_path: str):
        """
        初始化模型基类。
        
        参数:
            model_name_or_path: 模型路径或名称
        """
        # 模型基本信息
        self.model_name_or_path = model_name_or_path  # 模型路径或名称
        
        # 核心模型组件（由子类实现加载）
        self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)  # Transformers模型实例
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)  # Tokenizer实例
        
        # 指令处理相关属性
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()  # 指令tokenization函数，自动应用模型特定的聊天模板
        self.eoi_toks = self._get_eoi_toks()  # End of Instruction tokens，指令结束标记的token ID列表
        self.refusal_toks = self._get_refusal_toks()  # Refusal tokens，模型拒绝回答时常用的起始token ID列表（如 'I'）

        # 模型结构访问接口（用于激活干预）
        self.model_block_modules = self._get_model_block_modules()  # Transformer blocks列表，用于访问各层
        self.model_attn_modules = self._get_attn_modules()  # Attention模块列表，用于访问注意力层
        self.model_mlp_modules = self._get_mlp_modules()  # MLP模块列表，用于访问前馈网络层

    def del_model(self):
        """
        删除模型以释放内存。
        当不再需要模型时调用此方法可以释放GPU/CPU内存。
        """
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    # ========== 抽象方法：子类必须实现 ==========
    
    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        """
        加载Transformers模型。

        参数:
            model_name_or_path: 模型路径或名称
            
        返回:
            AutoModelForCausalLM: 加载的模型实例
            
        注意:
            子类需要实现模型特定的加载逻辑（如数据类型、设备映射等）
        """
        pass
    

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        """
        加载Tokenizer。
        
        参数:
            model_name_or_path: 模型路径或名称
            
        返回:
            AutoTokenizer: 加载的tokenizer实例
            
        注意:
            子类需要设置tokenizer的特定配置（如padding_side、pad_token等）
        """
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        """
        获取指令tokenization函数。
        
        返回:
            Callable: 一个函数，接受instructions列表，返回tokenized结果
            
        注意:
            该函数应该自动应用模型特定的聊天模板（如Llama-3的特殊标记、Qwen的im_start/im_end等）
            返回的函数签名应为: fn(instructions: List[str]) -> tokenized_output
        """
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        """
        获取End of Instruction tokens（指令结束标记）。
        
        返回:
            List[int]: 指令结束标记的token ID列表
            
        用途:
            用于识别指令部分的结束位置，在激活干预等场景中可能需要
        """
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        """
        获取Refusal tokens（拒绝标记）。
        
        返回:
            List[int]: 模型拒绝回答时常用的起始token ID列表
            
        说明:
            不同模型的拒绝标记可能不同，例如：
            - Llama-3: [40] (对应 'I')
            - Qwen: [40, 2121] (对应 'I', 'As')
            - Llama-2: [306] (对应 'I')
            
        用途:
            用于计算拒绝分数，判断模型是否拒绝回答
        """
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        """
        获取Transformer blocks列表。
        
        返回:
            ModuleList或类似结构: 所有Transformer层的列表
            
        说明:
            不同模型的路径不同：
            - Llama系列: model.model.layers
            - Qwen: model.transformer.h
            
        用途:
            用于访问各层进行激活干预、hook注册等操作
        """
        pass

    @abstractmethod
    def _get_attn_modules(self):
        """
        获取Attention模块列表。
        
        返回:
            ModuleList: 所有Attention层的列表
            
        说明:
            不同模型的Attention模块命名可能不同：
            - Llama系列: block.self_attn
            - Qwen: block.attn
            
        用途:
            用于访问注意力层进行激活干预
        """
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        """
        获取MLP（前馈网络）模块列表。
        
        返回:
            ModuleList: 所有MLP层的列表
            
        用途:
            用于访问MLP层进行激活干预（拒绝方向通常作用于MLP层）
        """
        pass

    @abstractmethod
    def _get_o_proj_modules(self):
        """
        获取Attention输出投影模块列表。
        
        返回:
            ModuleList: 所有层的输出投影模块列表
            
        说明:
            不同模型的输出投影层命名可能不同：
            - Llama系列、Gemma、Yi: self_attn.o_proj
            - Qwen: attn.c_proj
            
        用途:
            用于访问attention输出投影层，获取每个head的输出
        """
        pass

    @abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        """
        获取权重正交化函数。
        
        参数:
            direction: 拒绝方向向量，形状为 (d_model,)
            
        返回:
            Callable: 一个函数，接受model参数，将模型权重正交化到指定方向
            
        说明:
            正交化操作用于从模型权重中移除拒绝方向，用于消融实验
            不同模型的权重路径不同（如mlp.down_proj vs mlp.c_proj）
        """
        pass

    @abstractmethod
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        """
        获取激活加法函数（用于激活干预）。
        
        参数:
            direction: 拒绝方向向量，形状为 (d_model,)
            coeff: 系数，正数表示增强拒绝，负数表示减弱拒绝
            layer: 目标层索引（从1开始）
            
        返回:
            Callable: 一个函数，接受model参数，在指定层的MLP输出添加方向向量
            
        说明:
            激活加法用于干预模型行为，通过添加方向向量来增强或减弱拒绝行为
            不同模型的实现方式可能不同（如通过bias或直接修改激活）
        """
        pass

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64):
        """
        批量生成文本补全，支持激活干预（通过hooks）。
        
        这是模型的核心生成接口，用于：
        1. 对数据集中的指令生成响应
        2. 支持通过hooks进行激活干预（如添加拒绝方向、消融方向等）
        3. 自动应用模型特定的聊天模板
        
        参数:
            dataset: 数据集列表，每个元素应包含 'instruction' 和 'category' 字段
            fwd_pre_hooks: 前向传播前的hook列表，格式为 [(module, hook_fn), ...]
                          用于在模型前向传播前修改激活（如激活加法）
            fwd_hooks: 前向传播后的hook列表，格式为 [(module, hook_fn), ...]
                      用于在模型前向传播后修改激活
            batch_size: 批处理大小，默认8
            max_new_tokens: 最大生成token数，默认64
            
        返回:
            List[Dict]: 补全结果列表，每个元素包含：
                - 'category': 指令类别
                - 'prompt': 原始指令
                - 'response': 模型生成的响应文本
                
        工作流程:
            1. 从数据集中提取指令和类别
            2. 按批次处理（提高效率）
            3. 对每批指令应用模型特定的聊天模板并tokenize
            4. 在hook上下文内生成文本（支持激活干预）
            5. 提取新生成的token（去除输入部分）
            6. 解码为文本并返回结构化结果
            
        使用示例:
            # 无干预生成
            completions = model_base.generate_completions(dataset)
            
            # 带激活干预生成（减弱拒绝）
            direction = torch.load("direction.pt")
            hook = get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0)
            hooks = [(model_base.model_block_modules[15], hook)]
            completions = model_base.generate_completions(dataset, fwd_pre_hooks=hooks)
        """
        # 配置生成参数：贪婪解码（do_sample=False），不采样
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        instructions = [x['instruction'] for x in dataset]  # 提取所有指令
        categories = [x['category'] for x in dataset]  # 提取所有类别

        # 按批次处理数据集
        for i in tqdm(range(0, len(dataset), batch_size)):
            # 对当前批次的指令进行tokenization（自动应用聊天模板）
            tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i + batch_size])

            # 在hook上下文内生成文本（hook用于激活干预）
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(self.model.device),
                    generation_config=generation_config,
                )

            # 提取新生成的部分（去除输入token）
            generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

            # 解码并构建结果
            for generation_idx, generation in enumerate(generation_toks):
                completions.append({
                    'category': categories[i + generation_idx],
                    'prompt': instructions[i + generation_idx],
                    'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                })

        return completions
