"""最小可运行示例：在 quick CIFAR-10 配置上跑通若干轮联邦训练。

用法（在项目根目录下）：

    python examples/minimal_quick_cifar10.py
"""

from pathlib import Path

from scripts.demo import run_baseline


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "quick_cifar10.yaml"
    run_baseline(str(cfg_path), run_ga=False)
