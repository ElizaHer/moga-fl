import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ga_opt.optimizer import run_nsga2


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 pymoo-NSGA-II 优化联邦学习超参数的示例")
    parser.add_argument("--pop", type=int, default=12, help="种群大小")
    parser.add_argument("--gen", type=int, default=5, help="迭代代数")
    args = parser.parse_args()

    X, F = run_nsga2(pop_size=args.pop, n_gen=args.gen)

    print("=== GA 优化完成（展示前 5 个解） ===")
    for i in range(min(5, len(X))):
        x = X[i]
        f1, f2, f3 = F[i]
        print(f"解 {i}: x={x}, 目标=(-acc, time, -fairness)={f1:.4f}, {f2:.4f}, {f3:.4f}")


if __name__ == "__main__":
    main()
