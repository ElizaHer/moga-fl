# 迁移指南：从 moga-fl 到 moga-fl-refactored

本文档为熟悉原 `moga-fl` 仓库的开发者提供一份简明迁移指南，说明本次重构的核心变化、代码结构映射与功能对应关系。

## 核心设计理念变更

| 方面 | 原仓库 (`moga-fl`) | 本仓库 (`moga-fl-refactored`) | 理由 |
| --- | --- | --- | --- |
| **联邦学习框架** | 自定义实现的 Server/Client 通信与聚合逻辑，类似于 `torch.distributed` 的底层封装。 | 基于 **`fedlab==1.3.0`** 的标准 API，特别是 `ClientTrainer`。 | **标准化与解耦**：拥抱社区标准，避免重复造轮子，使代码更易于理解和维护。将 FL 核心逻辑与上层调度/GA 解耦。 |
| **多目标遗传算法** | 简化版的 NSGA-III / MOEA/D 手写实现。 | 采用成熟的 **`pymoo`** 库，使用其内置的 NSGA-II 算法。 | **健壮性与可扩展性**：`pymoo` 提供更可靠、经过广泛验证的算法实现。便于未来扩展到更复杂的约束处理和更多样的进化算法。 |
| **项目结构** | 所有源码位于 `src/` 下，按功能划分模块。 | 更清晰的分层结构：`fl_core`, `ga_opt`, `configs`, `scripts` 等。 | **模块化与关注点分离**：使开发者能快速定位到自己关心的部分，如只想修改 FL 训练逻辑，只需关注 `fl_core`。 |
| **运行与评估** | `run_baselines.py` 和 `run_ga_optimization.py` 强耦合，GA 评估直接调用 `Server`。 | GA 优化 (`demo_ga.py`) 通过 `FLHyperparamProblem` 调用一个简化的 `FLSimulation`，与主 FL 流程解耦。 | **灵活性与测试便利性**：GA 优化器不再依赖于一个完整的、复杂的 `Server` 实例，而是依赖于一个更轻量的 FL 仿真器，便于快速迭代和单元测试。 |

## 代码文件与功能映射

| 原仓库路径 (`moga-fl/src/`) | 功能描述 | 新仓库对应路径 (`moga-fl-refactored/`) | 主要变化 |
| --- | --- | --- | --- |
| `training/server.py`, `training/client.py`, `training/aggregator.py` | 自定义联邦学习 Server/Client 架构、聚合逻辑 | `fl_core/trainer.py`, `fl_core/fedlab_wrappers.py` | - **重写**：`Server` 被 `FLSimulation` 替代，后者是一个更轻量的单机仿真器。<br>- **适配 FedLab**：`Client` 的本地训练逻辑被封装到 `SimpleClientTrainer`，继承自 `fedlab.core.client.trainer.ClientTrainer`。<br>- **简化**：移除了复杂的网络通信和多线程逻辑，聚焦于算法本身。 |
| `training/algorithms/fedavg.py`, `fedprox.py`, `scaffold.py` | 各联邦学习算法实现 | `fl_core/trainer.py` (FedAvg), （其他算法暂未迁移） | - **集成**：FedAvg 逻辑直接集成在 `FLSimulation._fedavg_aggregate` 中。<br>- **待办**：FedProx/SCAFFOLD 等高级算法可基于 `SimpleClientTrainer` 扩展实现。 |
| `ga/*` | 手写 NSGA-III, MOEA/D, 目标/约束函数 | `ga_opt/problem.py`, `ga_opt/optimizer.py` | - **替换为 Pymoo**：遗传算法逻辑完全由 `pymoo` 驱动。<br>- `FLHyperparamProblem` 定义了优化问题（决策变量、目标函数）。<br>- `run_nsga2` 调用 `pymoo.minimize` 执行优化。 |
| `scripts/run_ga_optimization.py` | GA 优化入口脚本 | `scripts/demo_ga.py` | - **简化**：入口脚本只负责调用 `run_nsga2`，不再包含复杂的评估器构造逻辑。 |
| `scripts/run_baselines.py` | 联邦学习基线运行脚本 | `scripts/demo_fl.py` | - **简化**：功能不变，但其调用的 `FLSimulation` 已是基于 FedLab 适配的新版本。 |
| `data/*` | 数据加载与 Non-IID 划分 | `fl_core/data.py` | - **保留与增强**：保留了 Dirichlet 划分，并提供了可复现的合成数据生成函数 `make_synthetic_dataset`，便于无数据依赖的快速测试。 |
| `configs/*` | YAML 配置文件 | `configs/` | - **简化**：提供了更简洁的示例配置文件 `demo_fl.yaml`。 |

## 如何迁移自定义功能？

#### 迁移一个自定义的联邦学习算法

1.  **创建新的 ClientTrainer**：在 `fl_core/` 下创建一个新的 Python 文件，定义一个类继承自 `fl_core.fedlab_wrappers.SimpleClientTrainer`。
2.  **重写 `train()` 方法**：在你的新 Trainer 中，重写 `train()` 方法以实现你算法的本地训练逻辑（例如，若要实现 FedProx，在此处添加近端项）。
3.  **修改 `FLSimulation`**：调整 `FLSimulation` 的 `__init__`，使其可以接收并使用你自定义的 Trainer。

#### 迁移一个自定义的遗传算法

1.  **修改 `ga_opt/problem.py`**：
    -   调整 `FLHyperparamProblem` 的 `n_var`（决策变量数量）以及 `xl`, `xu`（边界）。
    -   在 `_evaluate` 方法中，解析新的决策变量 `x`，并将其应用到 `self.fl_sim.cfg` 中。
2.  **选择 Pymoo 算法**：在 `ga_opt/optimizer.py` 中，你可以将 `NSGA2` 更换为 `pymoo` 提供的任何其他多目标优化算法。
