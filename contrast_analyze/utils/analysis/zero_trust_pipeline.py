"""Stage 5 工具：零信任轨迹 orchestrator。"""

from __future__ import annotations

from contrast_analyze.sequential_zero_trust import main as zero_trust_main


def run_zero_trust(args):
    """
    显式入口，复用 sequential_zero_trust.main 中的 run_zero_trust。
    """
    zero_trust_main.run_zero_trust(args)

