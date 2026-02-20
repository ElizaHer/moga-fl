# FedLab 1.3.0 API 适配说明

本文档旨在阐明 `moga-fl-refactored` 项目是如何将原仓库自定义的联邦学习框架适配到 `fedlab==1.3.0` 标准 API 的，为希望基于 FedLab 进行二次开发的开发者提供参考。

## 核心 API 映射

FedLab 1.3.0 的核心在于其模块化的设计，主要包括 `ClientTrainer`（客户端训练逻辑）和 `ServerHandler`（服务器端聚合与控制逻辑）。在我们的重构中，由于目标是单机可运行的仿真环境，我们主要利用了 FedLab 的 **客户端** 相关抽象，而将服务器端的复杂网络管理和多进程协调逻辑简化为单机版的循环。

| 原仓库概念 | FedLab 1.3.0 对应 API | 本项目封装 (`moga-fl-refactored`) | 说明 |
| --- | --- | --- | --- |
| **客户端本地训练** (`Client.local_train`) | `fedlab.core.client.trainer.ClientTrainer` | `fl_core.fedlab_wrappers.SimpleClientTrainer` | `SimpleClientTrainer` 继承自 `ClientTrainer`，并实现了其核心的 `train()` 和 `local_process()` 方法。它封装了模型训练的标准 PyTorch 循环，使其符合 FedLab 的接口规范。 |
| **模型维护与序列化** (`Server.global_state`, `Client` 内部模型) | `fedlab.core.model_maintainer.ModelMaintainer` | `fl_core.fedlab_wrappers.GlobalModel` (继承 `ModelMaintainer`) | FedLab 使用 `ModelMaintainer` 来统一管理模型的参数获取 (`model_parameters`) 和设置 (`set_model`)。我们的 `GlobalModel` 和 `SimpleClientTrainer` 都利用了这一基类来处理模型的序列化与反序列化，确保与 FedLab 的工具链（如 `SerializationTool`）兼容。 |
| **客户端上传内容** | 自定义的 `state_dict` 字典 | `ClientTrainer.uplink_package` 属性 | FedLab 约定客户端通过 `uplink_package` 属性定义需要上传给服务器的数据。在 `SimpleClientTrainer` 中，我们简单地将其实现为 `return [self.model_parameters]`，即只上传模型参数。这为未来扩展（如上传梯度、loss 值等）提供了标准接口。 |
| **服务器聚合** (`Aggregator.aggregate_fedavg`) | `fedlab.utils.aggregator.Aggregators` | `fl_core.trainer.FLSimulation._fedavg_aggregate` | FedLab 提供了多种聚合函数（如 `fedavg`）。在我们的简化仿真器 `FLSimulation` 中，我们自行实现了一个与 FedLab `fedavg` 等价的加权平均聚合 `_fedavg_aggregate`，以避免引入完整的 `ServerHandler` 依赖，保持单机仿真的轻量性。 |
| **服务器到客户端的数据下发** | 自定义的数据包 | `ClientTrainer.local_process(payload: ...)` | FedLab 中，服务器通过调用客户端的 `local_process` 方法并传入 `payload`（一个张量列表）来下发数据。在 `SimpleClientTrainer` 中，我们约定 `payload[0]` 为全局模型参数，并在 `local_process` 的开头调用 `self.set_model(payload[0])` 来更新本地模型。 |

## 重构后的工作流（单机仿真）

相较于 FedLab 标准的跨进程（`Manager` + `Handler`）或分布式（`torch.distributed`）模式，`moga-fl-refactored` 采用了一种更轻量、更适合与遗传算法快速集成的 **单机串行仿真** 模式：

1.  **初始化** (`FLSimulation.__init__`):
    -   创建一个 `GlobalModel` 实例，作为全局模型状态的维护者。
    -   为每个虚拟客户端 `cid` 创建一个 `SimpleClientTrainer` 实例，每个实例持有自己的本地数据加载器 (`DataLoader`)。

2.  **仿真循环** (`FLSimulation.run`):
    -   **客户端采样**: 在每轮开始时，随机选择一部分客户端参与训练。
    -   **模型下发**: 获取全局模型的参数 (`self.global_model.model_parameters`)，并将其作为 `payload`。
    -   **本地训练**: 遍历被选中的客户端，依次调用其 `client.local_process(payload)` 方法。该方法内部会：
        1.  用 `payload` 更新自己的模型。
        2.  执行 `train()` 方法，完成多个本地 epoch 的训练。
    -   **模型上传**: 本地训练结束后，从每个参与的客户端获取其更新后的模型参数 (`client.uplink_package`)。
    -   **全局聚合**: 调用 `_fedavg_aggregate`，对收集到的所有本地模型参数进行加权平均。
    -   **更新全局模型**: 将聚合后的参数设置回 `self.global_model`。
    -   **评估**: 使用测试集评估新全局模型的性能。

通过这种方式，我们**复用了 FedLab `ClientTrainer` 的接口规范和 `ModelMaintainer` 的便利性**，同时**避免了启动完整 FedLab 网络服务的开销**。这使得每一次联邦学习的完整仿真（`FLSimulation.run()`）都可以被看作是一个单一的、可被遗传算法评估器调用的“黑盒函数”，极大地简化了 GA-FL 的集成。
