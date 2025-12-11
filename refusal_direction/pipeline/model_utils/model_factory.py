from refusal_direction.pipeline.model_utils.model_base import ModelBase

MODEL_ALIASES = {"gemma-2-2b-it": "google/gemma-2-2b-it",
                 "llama3" : "/root/autodl-tmp/Projects/LLaMA-3-8b-IT",
                 "llama2" : "/root/autodl-tmp/Projects/llama-2-7b-chat-hf",
                 "gemma-2-9b-it": "google/gemma-2-9b-it",
                 "qwen-7b": "Qwen/Qwen-7B-Chat",
                 "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
                 "yi-6b-chat": "01-ai/Yi-6B-Chat"
                 }

def construct_model_base(model_alias: str) -> ModelBase:
    model_path = MODEL_ALIASES.get(model_alias, model_alias)
    if 'qwen' in model_path.lower():
        from refusal_direction.pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    if 'llama-3' in model_path.lower():
        from refusal_direction.pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif 'llama' in model_path.lower():
        from refusal_direction.pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    elif 'gemma' in model_path.lower():
        from refusal_direction.pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path) 
    elif 'yi' in model_path.lower():
        from refusal_direction.pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)
    elif 'vicuna' in model_path.lower():
        from refusal_direction.pipeline.model_utils.vicuna13b_model import Vicuna13BModel
        return Vicuna13BModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
