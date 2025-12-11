import argparse
from easyjailbreak.attacker.GCG_Zou_2023 import GCG
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.huggingface_model import from_pretrained
from easyjailbreak.models.openai_model import OpenaiModel
from refusal_direction.pipeline.model_utils.model_factory import MODEL_ALIASES
from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template

def register_model_template(model_alias: str):
    """根据 model_alias 注册对应的 FastChat 对话模板，确保格式与 model_utils 中的定义对齐。"""
    model_path = MODEL_ALIASES.get(model_alias, model_alias)
    
    # Llama3 模板
    if 'llama-3' in model_path.lower() or model_alias == 'llama3':
        register_conv_template(
            Conversation(
                name="llama3",
                system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
                roles=("<|start_header_id|>user<|end_header_id|>", "<|start_header_id|>assistant<|end_header_id|>"),
                sep_style=SeparatorStyle.NO_COLON_TWO,
                sep="<|eot_id|>",
                sep2="<|eot_id|>",
            ),
            override=True
        )
    
    # Llama2 模板（FastChat已有，但确保注册）
    elif model_alias == 'llama2' or ('llama' in model_path.lower() and 'llama-3' not in model_path.lower()):
        register_conv_template(
            Conversation(
                name="llama2",
                system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
                roles=("[INST]", "[/INST]"),
                sep_style=SeparatorStyle.LLAMA2,
                sep=" ",
                sep2=" </s><s>",
            ),
            override=True
        )
    
    # Qwen 模板
    elif 'qwen' in model_path.lower():
        register_conv_template(
            Conversation(
                name=model_alias,
                system_template="<|im_start|>system\n{system_message}",
                system_message="You are a helpful assistant.",
                roles=("<|im_start|>user", "<|im_start|>assistant"),
                sep_style=SeparatorStyle.CHATML,
                sep="<|im_end|>",
                stop_str="<|endoftext|>",
            ),
            override=True
        )
    
    # Gemma 模板
    elif 'gemma' in model_path.lower():
        register_conv_template(
            Conversation(
                name=model_alias,
                roles=("<start_of_turn>user", "<start_of_turn>model"),
                sep_style=SeparatorStyle.NO_COLON_TWO,
                sep="<end_of_turn>",
                sep2="<end_of_turn>",
            ),
            override=True
        )
    
    # Yi 模板
    elif 'yi' in model_path.lower():
        register_conv_template(
            Conversation(
                name=model_alias,
                system_template="<|im_start|>system\n{system_message}",
                system_message="You are a helpful assistant.",
                roles=("<|im_start|>user", "<|im_start|>assistant"),
                sep_style=SeparatorStyle.CHATML,
                sep="<|im_end|>",
                stop_str="<|endoftext|>",
            ),
            override=True
        )


def get_template_name(model_alias: str) -> str:
    """将 model_alias 映射到对应的 FastChat 模板名称。"""
    model_path = MODEL_ALIASES.get(model_alias, model_alias)
    
    # 检查 model_alias 或 model_path 来确定模型类型
    if model_alias == 'llama3' or 'llama-3' in model_path.lower():
        return "llama3"
    elif model_alias == 'llama2' or ('llama' in model_path.lower() and 'llama-3' not in model_path.lower()):
        return "llama2"
    else:
        # 对于其他模型，直接使用 model_alias 作为模板名称
        return model_alias


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_alias', type=str, default='vicuna-13b-v1.5')
    return parser.parse_args()

def main(model_alias: str):
    # Register template for the model before creating models
    register_model_template(model_alias)
    template_name = get_template_name(model_alias)
    
    # First, prepare models and datasets.
    attack_model = from_pretrained(model_name_or_path=MODEL_ALIASES[model_alias], model_name=template_name)
    target_model = from_pretrained(model_name_or_path=MODEL_ALIASES[model_alias], model_name=template_name)
    dataset = JailbreakDataset('AdvBench')

    # Then instantiate the recipe.
    attacker = GCG(
        attack_model=attack_model,
        target_model=target_model,
        jailbreak_datasets=dataset,
        jailbreak_prompt_length=10,
        num_turb_sample=128,  # 从 512 减少到 256
        max_num_iter=200,     # 从 500 减少到 100
    )

    # Finally, start jailbreaking.
    attacker.attack(save_path=f'/root/autodl-tmp/Safety_analyze/{model_alias}/dataset/jailbreak/jailbreak_result.jsonl')

if __name__ == '__main__':
    args = parse_args()
    main(args.model_alias)