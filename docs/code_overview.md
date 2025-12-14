# 代码导读（联邦学习无线边缘调度项目）

> 本文面向阅读代码的同学，按“模块/文件”的方式，简要说明每个核心文件做什么、和论文思路中的哪一部分对应。更细节的数学推导可以回到论文，这里重点讲工程结构和调用关系。

## 1. 配置与入口脚本

- **configs/*.yaml**  
  - `default_cifar10.yaml` / `default_emnist.yaml`：完整实验配置，使用真实 CIFAR‑10 / EMNIST 数据集，轮数较多，适合长时间跑实验。  
  - `quick_cifar10.yaml` / `quick_emnist.yaml`：快速实验配置，轮数少、样本量小；当无法下载真实数据集时，会自动回退 FakeData，方便在低算力或离线环境快速验证流程。
- **src/configs/config.py**  
  - `load_config()`：从 YAML 读取配置字典。  
  - `merge_defaults()`：为缺失字段补上安全默认值（如客户端数、调度权重、评估轮数等），避免 KeyError。
- **scripts/run_baselines.py**  
  - 命令行入口，用来批量跑 FedAvg / FedProx / SCAFFOLD / FedBuff 等基线。  
  - 负责：加载配置 → 加载数据与划分 → 构建 `Server` → 按轮调用 `server.round()` → 收集指标并写入 CSV / 画图。
- **scripts/run_ga_optimization.py**  
  - 用改进型多目标遗传算法（NSGA‑III 或 MOEA/D）搜索调度参数：评分权重、Top‑K 的 k、迟滞阈值、FedBuff 陈旧度权重等。  
  - 内部通过一个简化的 “短轮评估” 模块，只跑少量联邦训练轮，就估计 accuracy / 时间 / 公平性 / 能耗 四个目标，用作 GA 的适应度。

## 2. 数据加载与 non‑IID 划分

- **src/data/dataset_loader.py**  
  - `DatasetManager`：负责统一的数据入口。  
  - 支持：
    - CIFAR‑10：彩色图像分类；
    - EMNIST：手写字符（balanced / letters 等 split）。  
  - 若下载失败或无网络，且 `use_fake_if_unavailable=True`，会自动使用 `torchvision.datasets.FakeData` 构造一个小型伪数据集，保证项目 “随时可跑”。
- **src/data/partition.py**  
  - `dirichlet_partition()`：按 Dirichlet 分布为每个客户端分配不同类别比例，用于模拟 label-level non‑IID。  
  - `label_bias_partition()`：给每个客户端随机分配少量固定类别，只在这些类上抽样，模拟 “每个设备只有少数类” 的极端 non‑IID。  
  - `quantity_bias_partition()`：用 log-normal 随机决定每个客户端样本量，模拟数据量不均。  
  - `apply_quick_limit()`：对每个客户端截断样本数，用在 quick 配置下加速实验。

## 3. 无线仿真与能耗近似

- **src/wireless/channel.py**  
  - `ChannelSimulator`：简化的无线信道仿真器，内部已经接入 `channel_models.py` 提供的多场景近似。  
  - 每轮生成：每个客户端的 Rayleigh 块衰落系数 → SNR（线性与 dB）→ 通过 `per = exp(-k·SNR_lin)` 近似得到丢包率 PER。  
  - 本轮训练中，Server 会根据每个客户端的 PER 决定其上传是否被 “随机丢弃”。
- **src/wireless/bandwidth.py**  
  - `BandwidthAllocator`：每轮有一个总带宽预算（Mb），在选中的客户端之间平均分配。  
  - `estimate_tx_time()`：用 “payload / 分到的带宽” 粗略估算通信时间，用于后续的时间和能耗近似。
- **src/wireless/energy.py**  
  - `EnergyEstimator`：把通信时间和计算样本数转换成能耗：  
    - 通信能耗 ≈ 发送功率 × 发送时间；  
    - 计算能耗 ≈ 计算功率 × (样本数 / 每秒可处理样本数)。  
  - 这些近似足以在毕设中比较不同调度/聚合策略的能耗趋势。
- **src/wireless/channel_models.py**  
  - `get_channel_model_params()`：为 `deepmimo_like_urban`、`nyusim_like_mmwave`、`quadriga_like_macro`、`tr38901_umi` 等场景返回一组近似的 `block_fading_intensity` / `base_snr_db` / `per_k`，用于在 `ChannelSimulator` 中构造不同的无线环境。

## 4. 调度评分、Top‑K 选择与公平债务

- **src/scheduling/fairness_ledger.py**  
  - `FairnessDebtLedger`：为每个客户端维护一个“公平债务”数值。  
  - 每轮结束：
    - 被选中的客户端：债务减少（视为“还债”）；
    - 未被选中的客户端：债务增加（视为“拖欠”）。  
  - 上限 `max_debt` 防止数值爆炸。调度评分里会把债务作为一维特征，提高长期未被选中设备的被选中概率。
- **src/scheduling/scorer.py**  
  - `ClientScorer`：多指标评分器，对每个客户端计算一个综合得分。  
  - 指标包括：
    - 能量充足度（越高越好）；
    - 信道质量（1−PER）；
    - 数据价值（目前用样本数近似，可进一步改成“标签稀有度”等）；
    - 公平债务（债务越大说明长期未参与，得分可适当提高）；
    - 带宽成本（通常是负权重）。  
  - 所有指标先做归一化，再加权求和，权重由配置或 GA 决定。
- **src/scheduling/selector.py**  
  - `TopKSelector`：在评分结果上做 Top‑K 选择，并带有简单的 “防抖” 机制：
    - 每轮先按分数排序取前 K 个；
    - 再参考上一轮历史，如果某些上轮客户端的分数没有明显变差（在一个 `hysteresis` 范围内），则保留它们，避免频繁切换客户端集合；
    - 通过滑动窗口 `sliding_window` 保留最近若干轮的选择用于统计 Jain 公平性。

## 5. 联邦训练与聚合算法

- **src/training/models/**  
  - `cifar_cnn.py` / `emnist_cnn.py`：分别是 CIFAR‑10 和 EMNIST 的轻量 CNN 模型，适合在单机 GPU 或 CPU 上快速训练。
- **src/training/client.py**  
  - `Client` 封装单个客户端的本地训练逻辑：
    - 从全局模型参数开始；
    - 若配置启用 FedProx（`fedprox_mu>0`），在 loss 中加入与全局模型的 L2 距离作为近端正则；
    - 返回本地训练后的 state_dict，用于聚合。
- **src/training/algorithms/fedavg.py**  
  - `aggregate_fedavg()`：经典的 FedAvg 权重平均，按每个客户端样本数做加权平均。
- **src/training/algorithms/fedprox.py**  
  - 当前主要逻辑在 `client.py` 中实现近端正则，这个文件中预留了进一步封装 FedProx 的接口。
- **src/training/algorithms/scaffold.py**  
  - `ScaffoldState`：存放全局和本地的控制变元（control variates），用于 SCAFFOLD 式的方差减小（目前为简化版，占位结构）。
- **src/training/algorithms/fedbuff.py**  
  - `FedBuffBuffer`：实现异步缓冲式聚合的核心数据结构：
    - `push()`：将 (本地模型、陈旧度) 放入缓冲；
    - `age_entries()`：在跨轮时整体增加缓冲条目的陈旧度，并可按 `max_staleness` 丢弃过旧更新；
    - `aggregate()`：根据陈旧度做加权平均，越陈旧的更新权重越小。  
  - 这部分对应问题3 中“缓冲队列 + 陈旧度加权”的核心逻辑。
- **src/training/aggregator.py**  
  - `Aggregator`：对聚合逻辑做统一封装：
    - 若未启用 FedBuff，或处于 `sync` / `semi_sync` 模式，直接调用 FedAvg；
    - 若启用 FedBuff 且处于 `async` 模式，则将更新推入 `FedBuffBuffer`，按缓冲大小 / 更新次数 / 轮次间隔触发异步聚合；
    - 若处于 `bridge` 模式，则同时计算同步 FedAvg 结果与异步 FedBuff 结果，并按权重线性混合，形成桥接态的混合聚合。  
  - 这里直接对应问题2/3/4 中的“半同步聚合、异步聚合（FedBuff 风格）以及桥接态混合聚合”。
- **src/training/server.py**  
  - `Server`：联邦学习主控：
    - 初始化全局模型、客户端集合、调度器、无线模块、聚合器与策略控制器等；
    - 每一轮：
      1. 从无线信道采样 PER；
      2. 调用 `StrategyController` 基于历史 PER / 公平性 / 能耗等信号，给出本轮聚合模式（半同步 / 异步 / 桥接态）及带宽缩放因子；
      3. 用能量/信道/数据价值/公平债务/带宽成本进行评分；
      4. 用 Top‑K 选择参与客户端并分配带宽；
      5. 在半同步/桥接态下只接收部分“按时返回”的客户端更新；
      6. 调用 `Aggregator.aggregate()`，在不同模式下完成同步 / 异步 / 混合聚合；
      7. 更新公平债务账本；
      8. 评估当前全局模型的测试准确率，并估算通信时间与能耗；
      9. 将本轮的丢包率均值、Jain 公平指数与能耗回写给 `StrategyController`，用于下一轮的门控决策。
  - `jain_index_selection()`：统计最近若干轮每个客户端被选中的频率，计算 Jain’s 公平指数。
- **src/training/strategy_controller.py**  
  - `StrategyController`：实现多指标门控、迟滞与防抖、桥接态和带宽再平衡：
    - 基于滑动窗口内的平均 PER、Jain 指数和能量消耗计算一个 gate_score；
    - 当 gate_score 较大时倾向切换到异步模式，较小时恢复半同步；
    - 使用最小间隔轮数和阈值区间避免频繁切换；
    - 在切换前进入 `bridge` 状态，输出随轮次线性变化的混合权重，用于 Aggregator 做同步/异步混合；
    - 同时给出一个简单的带宽缩放因子，用于在能量紧张时略微降低带宽预算，对应“预算再平衡”的直观近似。

## 6. GA 模块：多目标遗传优化（MOGA‑FL）

- **src/ga/pareto.py**  
  - `dominates()` / `non_dominated_set()`：基于 (acc, fairness, time, energy) 四个指标判断解之间的支配关系，并返回非支配集索引，用于 Pareto 前沿构建。
- **src/ga/constraints.py**  
  - `penalty()`：若能耗超过一个粗略预算，会加上惩罚；当前版本只对能耗做简单惩罚，后续可扩展到时间/带宽等约束。  
  - `repair()`：对 GA 产生的个体参数进行简单修复，如保证 Top‑K 至少为 1，迟滞在合理范围内。
- **src/ga/nsga3.py**  
  - 简化版 NSGA‑III：目前实现了：
    - 初始化种群（调度权重、Top‑K、迟滞、陈旧度指数等）；
    - 交叉与高斯噪声变异；
    - 利用 `non_dominated_set()` 做非劣解保留；
    - 使用 `constraints.repair()` 与 `constraints.penalty()` 做轻量约束处理。  
  - 局限：尚未实现真正意义上的参考点分配与多维度拥挤度控制，本轮将作为 “子组件” 被更高层的 MOGA‑FL 控制器调用。
- **src/ga/moead.py**  
  - 简化版 MOEA/D：通过随机生成一组权重向量，把多目标问题分解为多个标量化子问题。  
  - 对每个子问题，在邻域内做交叉+变异，若新解在该权重下的标量化目标更好，则替换原解。  
  - 目前也只是基础结构，本轮会在其之上封装更高层的 MOGA‑FL 控制。
- **src/ga/objectives.py**  
  - `evaluate_solution()`：给定一个“短轮联邦训练”结果，抽取 acc / time / fairness / energy 四个指标作为 GA 的目标。  
  - 与 `scripts/run_ga_optimization.py` 内部的 `make_sim_runner()` 协同工作。
- **scripts/run_ga_optimization.py**  
  - 构建数据集与划分，分别准备低保真与高保真两个 Server 评估器 `make_sim_runner(...)`；
  - 支持三种 `--algo` 选项：
    - `nsga3`：直接调用简化版 NSGA‑III；
    - `moead`：直接调用简化版 MOEA/D；
    - `moga_fl`：调用统一的 `MOGAFLController`，内部组合 NSGA‑III + MOEA/D + 岛屿模型 + 局部搜索 + 多保真评估。  
  - 所有候选解会写入 `outputs/results/pareto_candidates.csv`；同时按照 `eval.preference`（时间/公平/能耗优先）自动挑选一个部署用解，写回 `outputs/results/best_moga_fl_config.yaml`，用于后续完整训练复现。

## 7. 评估与绘图

- **src/eval/metrics.py**  
  - `MetricsRecorder`：简单的行缓冲器，将每一轮的 round / accuracy / time / energy / fairness 等信息收集起来，最后写到 CSV。
- **src/eval/plot.py**  
  - `plot_curves()`：从 CSV 中画出三类曲线：准确率、通信+计算能耗、Jain 公平指数。  
  - 已配置中文字体 Noto Sans CJK SC，用于在图例和坐标轴上正确显示中文。

## 8. 工具与测试

- **src/utils/**  
  - 目前包含随机种子设置、计时等小工具，用于保证结果尽量可重复、便于实验记时。
- **tests/**  
  - `test_partition.py`：检查 non‑IID 划分是否满足约束（客户端数量、总样本数等）。  
  - `test_scorer.py`：验证多指标打分在简单输入下的行为是否符合预期。  
  - `test_training_pipeline.py`：使用 FakeData 在极简配置下跑通若干轮训练，确保 pipeline 结构不崩。  
  - `test_wireless.py`：检查信道仿真输出的 PER / SNR 是否在合理区间。

---

当前版本已经完成以下关键实现，对应到你关心的 5 个问题：

1. **统一的 MOGA‑FL 控制器**：在 `src/ga/moga_fl.py` 中组合 NSGA‑III + MOEA/D + 岛屿模型 + 局部搜索 + 多保真评估，并在 `scripts/run_ga_optimization.py` 中实现了从 Pareto 候选到“部署用配置”的落地逻辑。  
2. **半同步 / 异步聚合与动态切换**：在 `src/training/aggregator.py`（聚合器）、`src/training/server.py`（轮次逻辑）与 `src/training/strategy_controller.py`（策略控制器）中实现了：
   - 中低丢包场景下按发送时间分位数的半同步聚合；
   - 高丢包场景下基于 FedBuff 缓冲队列与陈旧度加权的异步聚合；
   - 以及二者之间通过桥接态和混合聚合的平滑切换。  
3. **无线信道模型接口**：在 `src/wireless/channel_models.py` 与 `src/wireless/channel.py` 中加入了对 DeepMIMO-like / NYUSIM-like / QuaDRiGa-like / 3GPP TR 38.901-like 场景的参数化近似，并通过配置项 `wireless.channel_model` 进行选择。  
4. **GA 参数部署**：在 `scripts/run_ga_optimization.py` 中，增加了将 MOGA‑FL 选出的最优参数写回 YAML 配置（`best_moga_fl_config.yaml`）的能力，后续可以直接用该配置再次运行基线训练脚本进行完整验证。
