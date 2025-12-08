
import os

from dataclasses import dataclass
from typing import Optional, Tuple

MODEL_ALIASES = {"gemma-2-2b-it": "google/gemma-2-2b-it",
                 "llama3" : "/root/autodl-tmp/Projects/LLaMA-3-8B-IT",
                 "llama2" : "/root/autodl-tmp/Projects/llama-2-7b-chat-hf",
                 "gemma-2-9b-it": "google/gemma-2-9b-it",
                 "qwen-7b": "Qwen/Qwen-7B-Chat",
                 "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
                 "yi-6b-chat": "01-ai/Yi-6B-Chat"
                 }


@dataclass
class Config:
    model_alias: str
    model_path: Optional[str] = None
    n_test: int = 128
    splits: str = "saladbench"
    sample: bool = False
    filter_train: bool = True
    filter_val: bool = True
    # evaluation_datasets: Tuple[str] = ("jailbreakbench", "strongreject", "sorrybench", "xstest")
    evaluation_datasets: Tuple[str] = ("jailbreakbench",)
    evaluate_ablation: bool = True
    evaluate_actadd: bool = False
    evaluate_harmless: bool = False
    evaluate_loss: bool = False
    
    subspace_n_samples: int = 64
    max_new_tokens: int = 512
    completions_batch_size: int = 128

    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching", "strongreject")
    refusal_eval_methodologies: Tuple[str] = ("substring_matching",)
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048

    def __post_init__(self):
        # Allow callers to specify either an explicit model_path or a known model_alias.
        # If model_path is omitted, resolve it from the alias map; if alias not found,
        # fall back to using the alias itself as a path string.
        if not self.model_path:
            self.model_path = MODEL_ALIASES.get(self.model_alias, self.model_alias)

    def artifact_path(self) -> str:
        # Default output base; allow override via env SAVE_DIR
        save_dir = os.getenv("SAVE_DIR", "/root/autodl-tmp/rdo_res")
        # Dim dir can be used to separate experiments; default to model alias to avoid None
        dim_dir = os.getenv("DIM_DIR", self.model_alias)
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../", save_dir, dim_dir, self.model_alias)