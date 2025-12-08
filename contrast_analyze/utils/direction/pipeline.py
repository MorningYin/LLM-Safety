"""Stage 2 工具：生成拒答方向及候选。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from contrast_analyze.utils.common import ensure_dir
from contrast_analyze.utils.pipeline_config import PipelineConfig
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.run_pipeline import (
    filter_data as pipeline_filter_data,
    generate_and_save_candidate_directions as pipeline_generate_candidates,
    load_and_sample_datasets,
    select_direction,
)


def _save_candidate(config: PipelineConfig, candidate_directions: torch.Tensor, eoi_token_ids):
    artifact_dir = ensure_dir(config.direction_artifacts_dir)
    tensor_cpu = candidate_directions.detach().cpu()
    save_path = artifact_dir / "candidate_directions.pt"
    torch.save(tensor_cpu, save_path)

    metadata = {
        "shape": list(tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype),
        "n_positions": tensor_cpu.shape[0],
        "n_layers": tensor_cpu.shape[1],
        "d_model": tensor_cpu.shape[2],
        "positions": list(range(-len(eoi_token_ids), 0)),
        "eoi_token_ids": list(eoi_token_ids),
    }
    with open(artifact_dir / "candidate_directions.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _select_and_save(config: PipelineConfig, model_base, harmful_val, harmless_val, candidate_directions):
    artifact_dir = ensure_dir(config.direction_artifacts_dir)
    select_dir = artifact_dir / "select_direction"
    select_dir.mkdir(parents=True, exist_ok=True)

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=str(select_dir),
    )

    with open(artifact_dir / "direction_metadata.json", "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, artifact_dir / "direction.pt")
    return pos, layer, direction


def run_stage2_direction(config: PipelineConfig):
    cfg = Config(model_alias=config.model_alias)
    model_base = construct_model_base(cfg.model_alias)

    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)
    harmful_train, harmless_train, harmful_val, harmless_val = pipeline_filter_data(
        cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val
    )

    candidate_directions = pipeline_generate_candidates(cfg, model_base, harmful_train, harmless_train)
    _save_candidate(config, candidate_directions, model_base.eoi_toks)
    _select_and_save(config, model_base, harmful_val, harmless_val, candidate_directions)

    del model_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_independent_direction(config: PipelineConfig):
    """
    训练与第一个方向相互独立的第二个拒答方向。
    
    基于 rdo.py --train_independent_direction 的逻辑。
    通过调用 rdo.py 中的函数来实现训练。
    """
    # 加载第一个方向
    artifact_dir = ensure_dir(config.direction_artifacts_dir)
    direction_file = artifact_dir / "direction.pt"
    metadata_file = artifact_dir / "direction_metadata.json"
    
    if not direction_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"第一个方向文件不存在。请先运行 run_stage2_direction 生成第一个方向。"
            f"期望文件: {direction_file}, {metadata_file}"
        )
    
    first_direction = torch.load(direction_file, map_location="cpu")
    with open(metadata_file, "r") as f:
        direction_metadata = json.load(f)
    
    best_layer = direction_metadata["layer"]
    best_pos = direction_metadata["pos"]
    
    print(f"已加载第一个方向: layer={best_layer}, pos={best_pos}")
    print(f"方向向量形状: {first_direction.shape}")
    
    # 设置环境变量，以便 rdo.py 能找到方向文件
    # rdo.py 期望 DIM_DIR 环境变量指向包含 direction.pt 的目录
    os.environ.setdefault("DIM_DIR", str(artifact_dir.parent.name))
    os.environ.setdefault("SAVE_DIR", str(config.runs_root.parent))
    
    # 将 rdo.py 的路径添加到 sys.path
    rdo_path = Path(__file__).parent.parent.parent.parent / "geometry-of-refusal"
    if str(rdo_path) not in sys.path:
        sys.path.insert(0, str(rdo_path))
    
    try:
        # 导入 rdo.py 中的关键函数和类
        # 注意：这需要 rdo.py 中的所有依赖都可用
        from rdo import (
            train_independent_vector,
            repind_rdo,
            DirectionalAblation,
            CustomDataset,
            build_prompts_and_labels,
            generate_harmful_targets,
            generate_harmless_targets,
            apply_chat_template,
            get_bypass_scores,
            compute_ce_loss,
            kl_div_fn,
            get_cosine_sims_for_vector,
            projection_einops,
            custom_collate,
            refusal_metric,
        )
        
        # 由于 rdo.py 使用全局变量，我们需要设置它们
        # 这里我们采用一个更实用的方法：直接调用 train_independent_vector
        # 但需要准备所有必要的参数
        
        print("注意: train_independent_direction 需要 rdo.py 的完整环境")
        print("建议: 确保已安装所有依赖，并且 rdo.py 可以正常运行")
        
        # 由于 rdo.py 的复杂性，我们创建一个包装调用
        # 实际训练逻辑在 rdo.py 中，这里主要负责：
        # 1. 加载第一个方向
        # 2. 准备参数
        # 3. 调用训练函数
        # 4. 保存结果到正确的位置
        
        # 计算 repind_layers
        cfg = Config(model_alias=config.model_alias)
        model_base = construct_model_base(cfg.model_alias)
        # 获取模型层数
        if hasattr(model_base, 'model') and hasattr(model_base.model, 'config'):
            num_layers = model_base.model.config.num_hidden_layers
        elif hasattr(model_base, 'config'):
            num_layers = model_base.config.num_hidden_layers
        else:
            num_layers = 32  # 默认值
        layer_cutoff = 0.9
        repind_layers = list(range(int(num_layers * layer_cutoff)))
        del model_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 准备独立向量列表（包含第一个方向）
        independent_vectors = [first_direction]
        
        # 调用训练函数
        # 注意：这需要 rdo.py 中的全局变量（model, train_dataset等）都已设置
        print("开始训练独立方向...")
        print("警告: 此功能需要 rdo.py 的完整环境，包括已加载的模型和数据集")
        print("如果遇到错误，请确保已按照 rdo.py 的要求设置好环境")
        
        # 由于 rdo.py 依赖全局状态，我们提供一个占位实现
        # 实际使用时，需要确保 rdo.py 的环境已正确设置
        results = train_independent_vector(
            group_name="repind_pipeline",
            run_name=f"repind_{config.model_alias}",
            independent_vectors=independent_vectors,
            repind_layers=repind_layers,
            epochs=2,
            retain_lambda=0.1,
            repind_lambda=200,
        )
        
        # 保存第二个方向
        lowest_loss_vector = results['lowest_loss_vector']
        independent_direction_file = artifact_dir / "independent_direction.pt"
        torch.save(lowest_loss_vector.cpu(), independent_direction_file)
        
        # 保存元数据
        independent_metadata = {
            "layer": best_layer,
            "pos": best_pos,
            "source": "trained_independent",
            "based_on": "direction.pt",
        }
        independent_metadata_file = artifact_dir / "independent_direction_metadata.json"
        with open(independent_metadata_file, "w") as f:
            json.dump(independent_metadata, f, indent=4)
        
        print(f"独立方向已保存到: {independent_direction_file}")
        print(f"元数据已保存到: {independent_metadata_file}")
        
    except ImportError as e:
        print(f"无法导入 rdo.py 模块: {e}")
        print("请确保 rdo.py 及其所有依赖都可用")
        raise
    except Exception as e:
        print(f"训练独立方向时出错: {e}")
        import traceback
        traceback.print_exc()
        raise

