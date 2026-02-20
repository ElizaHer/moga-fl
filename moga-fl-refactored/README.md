# MOGA-FL-Refactored: 基于 FedLab 1.3.0 与 Pymoo 的多目标联邦学习调度框架

本项目是对 [ElizaHer/moga-fl](https://github.com/ElizaHer/moga-fl) 的一次深度工程化重构，旨在提供一个更现代化、更易于扩展、基于 FedLab 1.3.0 API 的联邦学习与多目标优化（Multi-Objective Genetic Algorithm, MOGA）研究框架。

核心改动点：
- **联邦学习核心**：从原仓库自定义的、类似 `torch.distributed` 的底层实现，全面迁移至 FedLab 1.3.0 的标准 API，特别是 `ClientTrainer` 与 `ModelMaintainer`，去除了原有的复杂手写 Server/Client 通信逻辑，使得 FL 流程更清晰、更易于与 FedLab 生态集成。
- **多目标遗传算法**：将原仓库中简化的 NSGA-III/MOEA/D 实现，替换为业界广泛使用且功能强大的 `pymoo` 库。这不仅提升了算法的稳定性和可复现性，也为未来扩展更复杂的约束、多样的遗传算子（交叉、变异）与更先进的多目标算法（如 R-NSGA-II, U-NSGA-III）打下基础。
- **工程结构**：采用更清晰的模块化分层，将联邦学习核心 (`fl_core`)、遗传算法优化 (`ga_opt`)、配置 (`configs`)、运行脚本 (`scripts`)、示例 (`examples`) 与文档 (`docs`) 完全解耦。
- **配置与依赖**：全面采用 YAML 进行配置，并提供 `requirements.txt` 以固定 `fedlab==1.3.0` 与 `pymoo` 等关键依赖，保证环境可复现。
- **示例与文档**：提供最小可运行示例与详细的中文文档，降低上手门槛。

## 目录结构

```
moga-fl-refactored/
├── fl_core/                    # 联邦学习核心，适配 FedLab 1.3.0
│   ├── data.py                 # 数据处理：合成数据生成、Non-IID 划分
│   ├── fedlab_wrappers.py      # 对 FedLab 1.3.0 API 的封装（如 ClientTrainer）
│   ├── models.py               # 示例模型（简单 CNN）
│   └── trainer.py              # 单机联邦学习仿真器
├── ga_opt/                     # 多目标遗传算法（基于 pymoo）
│   ├── optimizer.py            # NSGA-II 优化器入口
│   └── problem.py              # 将 FL 超参数搜索定义为一个 pymoo Problem
├── configs/                    # 配置文件
│   └── demo_fl.yaml
├── scripts/                    # 运行脚本
│   ├── demo_fl.py              # 运行联邦学习最小示例
│   └── demo_ga.py              # 运行 GA-FL 优化示例
├── examples/                   # 最小可运行示例
│   └── minimal_run.py
├── docs/                       # 中文文档
│   ├── migration_guide.md      # 从原仓库到本仓库的迁移指南
│   └── fedlab_api_mapping.md   # FedLab 1.3.0 API 对应关系
├── requirements.txt            # Python 依赖
└── README.md                   # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行联邦学习最小示例

该示例将在一个合成数据集上，运行一个简单的 FedAvg 流程。

```bash
python scripts/demo_fl.py --config configs/demo_fl.yaml
```

预期输出：
```
=== 联邦学习最小示例结果 ===
平均准确率: 0.1895
平均时间 proxy: 5.0000
平均 Jain 公平指数: 0.7344
```
*注：由于随机性，您的结果可能与此略有不同。*

### 3. 运行遗传算法优化联邦学习超参数示例

该示例将使用 `pymoo` 中的 NSGA-II 算法，对联邦学习的超参数（如客户端采样率、本地训练轮数等）进行多目标优化。

优化目标：
1. **最大化** 平均准确率
2. **最小化** 训练时间（此处用“参与客户端数 × 本地 epoch”作为简单代理）
3. **最大化** Jain 公平指数

```bash
python scripts/demo_ga.py --pop 8 --gen 3
```

预期输出：
```
=== GA 优化完成（展示前 5 个解） ===
解 0: x=[...], 目标=(-acc, time, -fairness)=-0.3385, 2.0000, -0.9882
解 1: x=[...], 目标=(-acc, time, -fairness)=-0.4025, 10.0000, -0.9883
解 2: x=[...], 目标=(-acc, time, -fairness)=-0.3590, 7.0000, -0.9893
解 3: x=[...], 目标=(-acc, time, -fairness)=-0.3670, 14.0000, -0.9932
解 4: x=[...], 目标=(-acc, time, -fairness)=-0.3495, 7.0000, -0.9937
```
输出展示了 Pareto 前沿上的部分候选解及其对应的三个目标值。

## GA 库选择：Pymoo vs. Geatpy2

根据任务要求，我们优先评估了 `pymoo` 作为遗传算法的核心库，并将其成功集成。

**选择 Pymoo 的理由：**
- **功能全面与社区活跃**：`pymoo` 是学术界和工业界广泛使用的多目标优化库，提供了包括 NSGA-II、NSGA-III、MOEA/D 在内的丰富算法实现。其社区活跃，文档详尽，更新迭代快。
- **强大的可扩展性**：`pymoo` 在定义问题（Problem）、算子（Operator）、终止条件（Termination）等方面提供了高度灵活的接口。将联邦学习超参数搜索封装为一个 `pymoo.Problem` 子类非常自然，便于后续添加更复杂的约束或自定义遗传算子。
- **与原仓库功能匹配度高**：原仓库的核心是多目标优化（准确率、时间、公平性、能耗），这正是 `pymoo` 的核心应用场景。NSGA-II 作为 `pymoo` 的经典算法，足以等价替换原仓库中的简化版 NSGA-III，同时提供了更可靠的性能保证。
- **决策支持**：`pymoo` 提供了丰富的可视化和决策工具（如 `Scatter` 图、`pcp` 图），便于对 Pareto 前沿进行分析。

**替代方案：Geatpy2**
- `geatpy2` 是另一个优秀的国产遗传和进化算法库，同样功能强大，尤其在国内有着广泛的用户基础，中文文档和社区支持非常友好。
- 在本次重构中，`pymoo` 已能很好地满足需求。若未来遇到 `pymoo` 无法解决的特定问题，或团队更熟悉 `geatpy2` 的 API，将其作为替代方案是完全可行的。迁移成本主要在于将 `ga_opt/problem.py` 中的 `FLHyperparamProblem` 重新封装为 `geatpy2` 的 `Problem` 类，核心的联邦学习评估逻辑可以复用。

## 详细文档

- **[迁移指南](./docs/migration_guide.md)**：详细说明了从原 `moga-fl` 仓库到本仓库的核心改动、代码结构映射与功能对应关系。
- **[FedLab 1.3.0 API 适配说明](./docs/fedlab_api_mapping.md)**：解释了如何将原有的自定义 FL 框架适配到 FedLab 1.3.0 的核心 API。

---

*本项目由 Aime 完成，旨在提供一个清晰、健壮的联邦学习多目标优化研究起点。*
