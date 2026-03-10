# 联邦学习调度与优化算法说明（门控切换 / 半同步 / 异步 / MOGA‑FL）

> 面向毕设论文与答辩的算法讲解，对应当前代码工程中的关键实现。
>
> 对照代码位置：
> - 混合策略（含门控）：`src/training/strategy/hybrid_wireless.py`
> - 调度与选择门：`src/scheduling/gate.py`
> - 异步聚合（FedBuff）：`src/training/algorithms/fedbuff.py`
> - 多目标遗传算法（MOGA‑FL）：
>   - 评估器：`src/ga/sim_runner_flower.py`
>   - 优化器：`src/ga/moga_fl.py` + `nsga3.py` + `moead.py`
>   - 可选 `pymoo` 优化器：`src/ga/pymoo_nsga3.py`
> - 无线建模：
>   - WSN: `data/wsn-indfeat-dataset/combined_wsn.csv`
>   - Simulated: `src/wireless/channel.py` + `src/wireless/channel_models.py`
> - FedProx / SCAFFOLD：`src/training/algorithms/fedprox.py` + `scaffold.py`

---

## 1. 门控切换策略：如何在半同步 / 异步 / 桥接态之间切换？

### 1.1 多指标门控（gate score）

在真实无线环境下，链路状况、公平性和能耗是耦合在一起的：

- 丢包率高（PER 高），意味着很多客户端的更新传不上来；
- 公平性差（Jain 指数低），意味着“强设备垄断训练、弱设备长期被冷落”；
- 能量消耗高，意味着继续保持“高强度训练”会拉高成本。

为此，`ModeController`（`src/scheduling/gate.py`）为每一轮维护一个滑动窗口，记录三类指标：

- `avg_per`：最近几轮的平均丢包率；
- `jain`：最近几轮的 Jain 公平指数（由选择历史计算）；
- `energy`：最近几轮的总能耗（通信 + 计算近似）。

在 `_compute_gate_score()` 中，这些指标被归一化后线性加权，得到一个 0~1 之间的 `gate_score`：

- 丢包率高 → gate_score 增大，倾向转向异步；
- 公平性差（1-Jain 大）→ gate_score 增大，倾向让更多尾部节点参与；
- 能量消耗过高 → gate_score 增大，倾向降低强同步强度，给系统“降档休息”。

这个 gate_score 可以理解为一个“综合压力表”：数值越大，说明当前网络越艰难、系统越“累”，需要从重同步模式退到更鲁棒的异步模式。

### 1.2 状态机：semi_sync / async / bridge

`ModeController.decide()` 内部维护一个简单的三态状态机：

- `semi_sync`：半同步模式，适用于中低丢包、网络相对稳定时；
- `async`：异步模式，适用于高丢包、强异构场景；
- `bridge`：桥接态，用于从 `semi_sync` 平滑切换到 `async`，或从 `async` 平滑切回 `semi_sync`。

关键逻辑：

1. 通过 `gate_score` 和阈值 `to_async` / `to_semi_sync` 决定“目标模式”：
   - gate_score 明显高于 `to_async + hysteresis_margin` → 目标模式设为 `async`；
   - 明显低于 `to_semi_sync - hysteresis_margin` → 目标模式设为 `semi_sync`；
   - 否则保持当前模式。
2. 为了防止频繁来回切换，使用 `min_rounds_between_switch` 限制两次切换之间的最小间隔；在间隔期内，即便 gate_score 有轻微波动，也不会立刻改模式，从而起到“防抖”作用。
3. 当目标模式与当前模式不同，状态机会先进入 `bridge`，并记录 `bridge_start_round` 与 `bridge_rounds`（桥接需要持续多少轮）：
   - 在桥接期间，通过 `decide()` 输出一个从 0 逐渐增大的 `bridge_weight`；
   - 聚合时，`HybridWirelessStrategy` 会用这个权重在“半同步结果”和“异步缓冲结果”之间做线性插值，实现“调音台式权重混合”；
   - 桥接轮数跑完后，模式正式切换到目标模式。

可以把这套逻辑理解为一个“自动变速箱”：半同步是高档、异步是低档，桥接态是“踩着离合器平滑换挡”。

### 1.3 带宽再平衡：bandwidth_factor

同一个控制器还会输出一个 `bandwidth_factor`（0.8~1.0 之间）：

- 能量消耗高 → 降低 `bandwidth_factor`，减少每轮带宽预算；
- 能量压力小 → 提高 `bandwidth_factor`，恢复到较高带宽水平。

`HybridWirelessStrategy` 会在每轮临时将 `BandwidthAllocator.budget_mb` 乘以 `bandwidth_factor`，用“更紧或更松”的带宽预算驱动客户端传输时间与能耗，从而形成一个轻量的“能量/带宽预算再平衡”回路。

---

## 2. 半同步聚合算法（对应中低丢包场景）

半同步模式由 `HybridWirelessStrategy.aggregate_fit()` 实现。

### 2.1 等待比例与按时返回客户端集合

在 `aggregate_fit()` 中：

1. 先通过 `ChannelSimulator.sample_round()` 为每个客户端生成本轮的信道统计：`snr_db`、`per`、`distance_m`。
2. 根据调度评分（能量、信道、数据价值、公平债务）选出 Top‑K 客户端 `selected_cids`；
3. 用 `BandwidthAllocator` 按选中客户端平均分配带宽，并估算每个客户端的发送时间 `tx_time[cid]`；
4. 在半同步模式下，根据配置的 `semi_sync_wait_ratio`（例如 0.7）计算发送时间分位数：
   - 取 `tx_time` 的 70% 分位数作为“等待阈值”；
   - 将发送时间不超过阈值的客户端视为“按时”，其余客户端视为“迟到”；
   - 在本轮训练中，只允许“按时”的客户端参与同步聚合，迟到的客户端更新虽然也被接收，但不会计入本轮的同步部分。

直观地说，这就像设定一个“迟到线”：只要在合理时间内能把梯度传回来的客户端都算“按时”，其余则延后。

### 2.2 FedAvg 聚合与成本估计

对于按时返回的客户端：

1. 服务器将全局模型参数下发给每个客户端；
2. 客户端执行本地训练（算法为 FedAvg 或 FedProx），返回本地模型参数；
3. `HybridWirelessStrategy.aggregate_fit()` 在 `sync`/`semi_sync` 模式下直接调用 `_weighted_avg`：
   - 按客户端样本数加权平均各自模型参数，得到新的全局参数；
4. 同时，`HybridWirelessStrategy` 会估算：
   - 本轮总通信时间 `est_upload_time`；
   - 通信能耗 `comm_energy`（功率×时间）；
   - 计算能耗 `comp_energy`（按处理样本数近似）；
   - 并写入 `metrics.csv`，供后续 GA 优化与仪表盘使用。

半同步的效果是：在中低丢包时，绝大多数客户端都能“按时”，整体行为接近 FedAvg；在偶尔有少数慢节点时，可以不被拖累，从而提高单位时间内的有效训练轮数。

---

## 3. 异步聚合算法（FedBuff 风格，对应高丢包场景）

异步聚合由 `FedBuffState`（`src/training/algorithms/fedbuff.py`）和 `HybridWirelessStrategy.aggregate_fit()` 的 `async` 分支共同实现。

### 3.1 FedBuff 缓冲结构与陈旧度

`FedBuffState` 内部维护一个队列 `entries = [(state_dict, num_examples, server_round), ...]`：

- `state_dict`：客户端本地训练后的模型参数；
- `num_examples`：对应的样本数；
- `server_round`：更新到达时的服务器轮次。

在 `aggregate_fit()` 的异步路径中：

1. 每轮开头，调用 `fedbuff.age()`，更新缓冲中所有条目的陈旧度；若 `staleness` 超过 `max_staleness`，该条目会被丢弃；
2. 本轮新到的有效更新会通过 `fedbuff.push()` 加入队列；
3. 当满足以下任一条件时触发一次聚合：
   - 缓冲长度达到 `buffer_size`；
   - 自上次聚合以来累计更新数达到 `min_updates_to_aggregate`；
   - 距离上次聚合的轮数不小于 `async_agg_interval`。

### 3.2 陈旧度加权与聚合

触发聚合时，`fedbuff.aggregate()` 会：

1. 遍历缓冲中的所有条目，计算其相对于当前轮次的 `staleness`；
2. 为每个条目分配权重 `w = num_examples / (1 + staleness)^alpha`（`alpha` 在配置中为 `staleness_alpha`）；
3. 按这些权重对所有 `state_dict` 做加权平均，得到新的全局模型；
4. 清空缓冲，为下一轮积累新更新。

这样，越“旧”的梯度（陈旧度大）对全局模型的影响越小，极旧的更新要么被淡化，要么被直接丢弃，从而保证异步训练在高丢包、高延迟环境下仍然收敛且不过度被陈旧信息拖累。

### 3.3 异步模式下的服务器行为

在 `HybridWirelessStrategy` 中，当策略控制器给出的模式为 `async` 时：

- 每轮仍然会进行调度、发模型、收更新，但收不到更新的客户端可以长期“缺席”；
- `aggregate_fit()` 会将收到的更新累积到 FedBuff 缓冲中，只有缓冲“积累到一定程度”时才推进一次全局模型；
- 即便某些客户端长期丢包，只要还有一部分客户端能稳定上传更新，训练就能持续向前推进。

---

## 4. 桥接态与混合聚合：如何平滑完成策略切换？

桥接态（`bridge`）是本工程一个重要的“工程增强点”，对应论文中提到的：

> “权重混合聚合 + 公平债务 + 能量与带宽再平衡 + 不变量检查”的双模策略衔接。

在代码中，桥接态主要体现在两个地方：

1. `ModeController` 将当前模式设为 `bridge`，并随轮数输出从 0→1 的 `bridge_weight`；
2. `HybridWirelessStrategy.aggregate_fit()` 中，对于 `mode == 'bridge'` 的情况，会同时计算：
   - `sync_result`：本轮按时返回客户端的 FedAvg 同步结果；
   - `async_result`：FedBuff 缓冲当前给出的异步结果；
   - 最终全局模型：`mixed = (1-w)*sync_result + w*async_result`。

当 w 从 0 缓慢扫到 1 时，系统的行为从“几乎完全依赖半同步更新”逐渐过渡到“几乎完全依赖异步缓冲”，在曲线层面表现为：

- 精度曲线不会出现强烈的“断崖式跳变”；
- 通信时间与能耗的统计也会有一个缓慢的过渡过程，而不是瞬间翻倍或腰斩。

同时，`ModeController` 中还通过：

- `min_rounds_between_switch`：限制切换频率，避免来回抖动；
- `bandwidth_factor`：在桥接期对带宽进行适度缩放，起到“预算再平衡”的作用；
- 公平性（Jain 指数）和能耗信号：在 gate_score 中占权重，防止切换过程长期伤害尾部客户端或击穿能量预算。

这些都是代码里“调音台式权重混合”和“公平债务 + 预算再平衡”思想的具体落地。

不变量检查方法：
可以，先把机制定成“可解释 + 可落地 + 可调参”。

**1. 检测什么（每轮 bridge 后评估）**

1. 预算不变量  
- 能量预算：`E_round <= E_budget_round`  
- 带宽/时延预算：`T_upload_round <= T_budget_round`（或 `sum(payload_i / bw_i)`）  
- 预算来源：
  - 固定阈值（YAML 配）
  - 或动态阈值（过去 `W` 轮的 P90/P95）

2. 陈旧度不变量  
- `max(staleness_i) <= max_staleness`  
- `staleness` 可以用“更新产生轮次 vs 当前聚合轮次”计算  
- 对 semi-sync/bridge 下的慢客户端，若 `tx_time` 超阈值导致持续滞后，也视为潜在 staleness 风险

3. 公平债务趋势不变量  
- 定义债务 `debt_i = max(0, target_select_rate - actual_rate_i)`（滑窗）  
- 期望 bridge 期内“总债务下降”：  
  - `mean_debt_t <= mean_debt_{t-1}`，或更稳健：`EMA_debt_t` 单调不升  
- 若债务不上升但高位震荡，也可判“未改善”

---

**2. 三种动作怎么执行**

1. 客户端降权（soft）  
- 场景：预算超标或 staleness 风险来自少数客户端  
- 做法：聚合权重乘惩罚因子  
  - `w_i' = w_i * penalty_i`
  - `penalty_i` 由风险分数组合：高 `tx_time`、高 `per`、高 staleness、低剩余能量  
- 优点：不直接剔除，训练更平滑

2. 限流（rate limit）  
- 场景：总体预算超标（尤其上传时延/带宽）  
- 做法：
  - 降低本轮总带宽预算 `bw_budget *= gamma`（`gamma<1`）
  - 或减少 top-k / 提高 semi-sync 等待分位阈值策略
- 优先对“高成本低收益”客户端限流（非全体一刀切）

3. 延长 bridge（stability-first）  
- 场景：连续 `K` 轮不变量失败，说明直接切 async/semi-sync 风险高  
- 做法：`bridge_rounds += extra_rounds`（有上限）  
- 触发条件建议“连续失败 + 指标未改善”，避免频繁延长

---

**3. 触发依据建议**

每轮 bridge 计算 violation score：
- `v_budget = max(0, E_round/E_budget - 1) + max(0, T_round/T_budget - 1)`
- `v_stale = max(0, stale_max/max_staleness - 1)`
- `v_fair = max(0, mean_debt_t - mean_debt_{t-1})`

总分：
- `V = a*v_budget + b*v_stale + c*v_fair`

动作映射：
- `V < th1`：无动作  
- `th1 <= V < th2`：客户端降权  
- `th2 <= V < th3`：降权 + 限流  
- `V >= th3` 或连续 `K` 轮失败：降权 + 限流 + 延长 bridge

---

**4. 参数放哪里**

在 YAML `controller.bridge_invariants` 下统一配置，例如：
- `energy_budget_round`
- `upload_time_budget_round`
- `fairness_debt_trend: non_increasing`
- `violation_weights: {budget, stale, fair}`
- `thresholds: {th1, th2, th3}`
- `rate_limit_factor`
- `bridge_extend_rounds`
- `max_bridge_rounds`
- `fail_streak_for_extend`

---

---

## 5. FedProx 与 SCAFFOLD：如何在 non‑IID 场景中稳住训练？

### 5.1 FedProx：在本地损失中加入近端正则

相关代码：

- 本地训练：`src/training/client.py` 中的 `CifarClient.fit()`；
- 近端正则计算：`src/training/algorithms/fedprox.py` 中的 `fedprox_regularizer()`。

当配置 `algorithm.name: fedprox` 且 `fedprox_mu>0` 时：

- 客户端本地目标变为：`f_k(w) + μ/2 ||w - w_global||^2`；
- 代码中在交叉熵损失 `loss` 上叠加 `fedprox_regularizer(self.model, global_params, mu, self.device)`；
- 其中 `global_params` 是当前轮下发的全局参数，`μ` 由配置 `fedprox_mu` 控制。

直观理解：FedProx 会惩罚“偏离当前全局模型太远的本地更新”，从而在数据分布高度不均（non‑IID）时缓解局部模型过拟合自身数据的问题，有助于提高全局收敛稳定性。

### 5.2 SCAFFOLD：控制变元修正梯度漂移

相关代码：

- 控制变元状态：`src/training/algorithms/scaffold.py` 中的 `ScaffoldState`；
- SCAFFOLD 本地训练：`src/training/client.py` 中的 `CifarClient.fit()`；
- 服务器整合：`HybridWirelessStrategy` 中处理 `algorithm == 'scaffold'` 的分支。

实现要点：

1. **控制向量初始化**：
   - 服务器创建 `ScaffoldState(global_model_state)`，生成与模型参数同形状的 `c_global`（全 0）和每个客户端的 `c_i`（懒初始化时为 0）。
2. **本地训练中的梯度校正**：
   - 在 `fit()` 中，客户端从服务器下发的 `config` 中解包 `c_global`，并从自身状态中取出 `c_i`；
   - 正常反向传播得到梯度后，对每个参数执行：`grad ← grad + (c_i - c_global)`；
   - 这样更新规则变为 `w ← w - η (g - c + c_i)`，对应论文中的控制变元修正项，可显著减缓 non‑IID 导致的客户端漂移。
3. **Δc_i 的计算与回传**：
   - 本地训练结束后，客户端将最终本地参数 `w_local` 和训练步数 `num_steps` 交给 `ScaffoldState.compute_delta_ci`；
   - 该函数按论文的近似公式：`c_i^{new} = c_i - c + (1/(η·τ))(w_global - w_local)` 计算新的 `c_i`，并返回 Δc_i；
4. **服务器端更新全局控制向量**：
   - 在 `HybridWirelessStrategy.aggregate_fit()` 聚合完模型参数之后，收集所有参与客户端的 Δc_i，按样本数加权平均；
   - 调用 `ScaffoldState.update_global(delta_list, weights)` 更新 `c_global`。

这样，FedProx 和 SCAFFOLD 分别从“损失函数层面”和“梯度层面”双重修正 non‑IID 带来的训练不稳定性，且都与门控切换和半同步/异步聚合兼容。

---

## 6. 改进型 MOGA‑FL：多目标遗传优化的工作流程（2026-03-08 实现版）

相关代码集中在 `src/ga/`：

- `objectives.py`：统一目标接口（5 目标：精度、时间、能耗、通信成本、公平性）；
- `constraints.py`：预算惩罚 + 染色体修复（Top-K、阈值关系、压缩比例、带宽因子、权重归一化）；
- `nsga3.py`：带参考方向关联与 niche 选择的 NSGA‑III 风格实现；
- `moead.py`：基于权重向量分解与邻域替换的 MOEA/D；
- `moga_fl.py`：`MOGAFLController`（NSGA‑III→MOEA/D 双阶段、岛屿迁移、可行域局部搜索、多保真与精英记忆）；
- `sim_runner_flower.py`：把染色体参数映射到 Flower 配置并聚合 `metrics.csv`；
- `scripts/run_ga_optimization.py`：GA 入口与部署配置写回；
- `pymoo_nsga3.py`：可选的 `pymoo` NSGA-III 对照实现（同一套 10 维编码、5 目标）。

### 6.1 染色体编码（联合优化）

当前实现将以下变量共同编码为个体：

- 综合评分权重：`energy_w`, `channel_w`, `data_w`, `fair_w`；
- 选择规模：`selection_top_k`；
- 带宽分配系数：`bandwidth_alloc_factor`；
- 异步陈旧度衰减：`staleness_alpha`；
- 桥接触发阈值：`bridge_to_async`, `bridge_to_semi_sync`；
- 通信压缩比例：`compression_ratio`（映射到 `wireless.payload_compression_ratio`）。

### 6.2 多目标定义（5 目标）

`evaluate_solution()` 返回并统一以下目标：

- 最大化：`acc`, `fairness`；
- 最小化：`time`, `energy`, `comm_cost`。

其中：

- `time` 来自 `metrics.csv` 的 `est_upload_time`；
- `energy` 来自 `metrics.csv` 的 `energy`；
- `comm_cost` 采用通信代理 `mean(topk * payload_mb)`；
- `fairness` 采用 `jain` 均值。

### 6.3 约束修复与惩罚

`constraints.repair()` 负责修复不可行个体：

- `selection_top_k >= 1`；
- `staleness_alpha`、`bandwidth_alloc_factor`、`compression_ratio` 在合法范围内；
- `bridge_to_semi_sync <= bridge_to_async - 0.03`（保持门控阈值顺序）；
- 调度权重非负并归一化。

`constraints.penalty()` 对预算超限进行软惩罚并加到最小化目标中，覆盖：

- 能量预算（优先 `controller.bridge_invariants.energy_budget_round`）；
- 上传时间预算（`upload_time_budget_round`）；
- 通信成本预算（按当前配置估计的 proxy budget）。

### 6.4 NSGA‑III + MOEA/D 双阶段协同

`MOGAFLController.run()` 的核心流程：

1. NSGA‑III 阶段：参考方向维持解集分布性，做广覆盖探索；
2. 岛屿迁移：将 NSGA‑III 非支配精英迁移到 MOEA/D 初始种群；
3. MOEA/D 阶段：按权重子问题和邻域替换强化收敛；
4. 合并后再取低保真非支配精英。

### 6.5 可行域局部搜索 + 多保真 + 精英记忆

- 可行域局部搜索：围绕能耗/时间/通信成本约束边界，对精英个体做定向微调（如压缩比例、Top-K、带宽因子、桥接阈值）；
- 多保真评估：早期低保真筛选，后期对“精英+邻域”做高保真复评；
- 精英记忆：以标准化染色体 key 缓存低/高保真评估结果，减少重复开销。

### 6.6 部署偏好与配置落地

`scripts/run_ga_optimization.py` 支持三类部署偏好：

- `time` 优先；
- `energy` 优先；
- `fairness` 优先。

脚本输出：

- `outputs/results/pareto_candidates.csv`；
- `outputs/results/best_moga_fl_config.yaml`。

并将最优个体回写到运行配置中，包括：

- `scheduler.weights.*`；
- `scheduler.selection_top_k`；
- `fedbuff.staleness_alpha`；
- `controller.gate_thresholds.to_async/to_semi_sync`；
- `wireless.bandwidth_budget_mb_per_round`（经 `bandwidth_alloc_factor`）；
- `wireless.payload_compression_ratio`。

---

## 7. 总结

通过上述模块，本工程已经实现了：

- 模型侧：从小 CNN 升级到 CIFAR‑10 / EMNIST 上的 ResNet‑18 变体（支持宽度缩放），在 quick 配置下使用“窄版 ResNet”，在完整配置下使用标准宽度；
- 训练侧：在 FedAvg 基础上，引入严格遵循论文定义的 FedProx 与控制变元版本的 SCAFFOLD；
- 聚合侧：基于 Flower 框架实现了半同步、FedBuff 异步和桥接态混合聚合，并用门控 + 迟滞实现动态切换；
- 无线侧：引入基于路径损耗 + 阴影 + Rayleigh 衰落的统计信道模型，近似 DeepMIMO / NYUSIM / QuaDRiGa / 3GPP TR 38.901；
- 优化侧：实现了 NSGA‑III + MOEA/D + 岛屿模型 + 局部搜索 + 多保真评估的 MOGA‑FL，用于自动搜索调度与异步相关参数。新增了对 `pymoo` 库的支持以增强优化鲁棒性。

这些实现都在代码层面给出了清晰注释，明确指出“对应论文中的哪一部分思想”，便于在毕设论文和答辩中直接引用和讲解。用户可以围绕门控切换、半同步/异步聚合、遗传算法搜索与无线建模这四条主线，系统回答“我具体实现了什么复杂方法”。

--

## TODO

### FL
Experiments

### GA
1. 构造低保真与高保真评估器 `low_sim` / `high_sim`：
   - `low_sim`：缩短轮数、减少客户端数，用于快速粗评估；
   - `high_sim`：使用原配置的轮数和客户端数，对少数精英解高保真评估；
2. 考虑把 `hysteresis`（选择防抖的迟滞系数）划入参数搜索范围

## 8. Hybrid Client-Scoring Upgrade (2026-03-05)
This section documents the planned upgrade to improve `hybrid_invTrue` under jitter.

### 8.1 Composite quality score
For each active client i, define:

`quality_i = a*(1-PER_i) + b*data_value_i + c*historical_contribution_i`

`survival_i = clip(remaining_energy_i / expected_round_energy_i, 0, 1)`

`score_i = quality_i * survival_i^gamma + fair_w*fairness_debt_i`

where:
- `data_value_i` captures client-side data utility,
- `historical_contribution_i` is a smoothed estimate of effectiveness per cost,
- `gamma` controls how strongly low-energy clients are downweighted.

### 8.2 data_value definition (lightweight online version)
`data_value_i = novelty_w*novelty_i + tail_w*tail_i + size_w*size_i`

Current implementation approximation:
- `size_i`: normalized partition size.
- `tail_i`: inverse historical selection frequency (clients rarely selected are boosted).
- `novelty_i`: one-step loss-improvement proxy from per-client contribution history.

### 8.3 historical_contribution definition
Maintain EMA per client:
`hist_contrib_i <- beta*hist_contrib_i + (1-beta)*(delta_acc_i / max(cost_i, eps))`

Practical approximation in simulation:
- `delta_acc_i` uses round-level relative contribution proxy.
- `cost_i` uses communication + compute energy estimate.

### 8.4 Energy guardrails and anti-monopoly
- Reserve threshold: if `remaining_energy_i / initial_energy_i < min_reserve_energy_ratio`, skip client.
- Future-feasibility gate: if `remaining_energy_i < k * expected_round_energy_i`, skip client.
- Anti-monopoly: enforce `max_consecutive_selected` and `cooldown_rounds`.

### 8.5 Bridge invariant tuning direction
When bridge invariants hurt performance:
- raise budgets and thresholds,
- reduce downweight and throttle intensity,
- keep smooth switching while avoiding over-throttling useful updates.
