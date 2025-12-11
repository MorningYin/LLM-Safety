"""JSONL 文件读写工具。"""

import json
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: Path) -> List[Dict]:
    """从 JSONL 文件加载数据。
    
    Args:
        path: JSONL 文件路径
        
    Returns:
        数据列表，每个元素是一个字典
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件为空或格式错误
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件：{path}")

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if not items:
        raise ValueError(f"{path} 为空")
    
    return items


def save_jsonl(path: Path, items: List[Dict]) -> None:
    """保存数据到 JSONL 文件。
    
    Args:
        path: 输出文件路径
        items: 要保存的数据列表
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

