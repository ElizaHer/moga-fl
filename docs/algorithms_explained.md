# 联邦学习调度与优化算法说明（门控切换 / 半同步 / 异步 / MOGA‑FL）

> 面向毕设论文与答辩的“算法讲解稿”，对应当前代码工程中的关键实现。
>
> 对照代码位置：
> - 门控切换策略：`src/training/strategy_controller.py`、`src/training/server.py`、`src/training/aggregator.py`
> - 半同步聚合：`Server.round()` + `Aggregator.aggregate()` 的 `semi_sync` 分支
> - 异步聚合（FedBuff 风格）：`src/training/algorithms/fedbuff.py` + `Aggregator.aggregate()` 的 `async` 分支
> - 桥接态与混合聚合：`StrategyController` 的状态机 + `Aggregator` 的 `bridge` 分支
> - 多目标遗传算法（MOGA‑FL）：`src/ga/moga_fl.py` + `nsga3.py` + `moead.py` + `objectives.py` + `constraints.py`
> - 无线建模：`src/wireless/channel.py` + `src/wireless/channel_models.py`
> - FedProx / SCAFFOLD：`src/training/client.py` + `src/training/algorithms/fedprox.py` + `src/training/algorithms/scaffold.py`

---

## 1. 门控切换策略：如何在半同步 / 异步 / 桥接态之间切换？

### 1.1 多指标门控（gate score）

在真实无线环境下，链路状况、公平性和能耗是耦合在一起的：

- 丢包率高（PER 高），意味着很多客户端的更新传不上来；
- 公平性差（Jain 指数低），意味着“强设备垄断训练、弱设备长期被冷落”；
- 能量消耗高，意味着继续保持“高强度训练”会拉高成本。

为此，`StrategyController`（`src/training/strategy_controller.py`）为每一轮维护一个滑动窗口，记录三类指标：

- `avg_per`：最近几轮的平均丢包率；
- `jain`：最近几轮的 Jain 公平指数（由选择历史计算）；
- `energy`：最近几轮的总能耗（通信 + 计算近似）。

在 `_compute_gate_score()` 中，这些指标被归一化后线性加权，得到一个 0~1 之间的 `gate_score`：

- 丢包率高 → gate_score 增大，倾向转向异步；
- 公平性差（1-Jain 大）→ gate_score 增大，倾向让更多尾部节点参与；
- 能量消耗过高 → gate_score 增大，倾向降低强同步强度，给系统“降档休息”。

这个 gate_score 可以理解为一个“综合压力表”：数值越大，说明当前网络越艰难、系统越“累”，需要从重同步模式退到更鲁棒的异步模式。

### 1.2 状态机：semi_sync / async / bridge

`StrategyController.decide_mode()` 内部维护一个简单的三态状态机：

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
   - 在桥接期间，通过 `_current_mode_and_bridge_weight()` 输出一个从 0 逐渐增大的 `bridge_weight`；
   - 聚合时，`Aggregator` 会用这个权重在“半同步结果”和“异步缓冲结果”之间做线性插值，实现“调音台式权重混合”；
   - 桥接轮数跑完后，模式正式切换到目标模式。

可以把这套逻辑理解为一个“自动变速箱”：半同步是高档、异步是低档，桥接态是“踩着离合器平滑换挡”。

### 1.3 带宽再平衡：bandwidth_factor

同一个控制器还会输出一个 `bandwidth_factor`（0.8~1.0 之间）：

- 能量消耗高 → 降低 `bandwidth_factor`，减少每轮带宽预算；
- 能量压力小 → 提高 `bandwidth_factor`，恢复到较高带宽水平。

`Server.round()` 会在每轮临时将 `BandwidthAllocator.budget_mb` 乘以 `bandwidth_factor`，用“更紧或更松”的带宽预算驱动客户端传输时间与能耗，从而形成一个轻量的“能量/带宽预算再平衡”回路。

---

## 2. 半同步聚合算法（对应中低丢包场景）

半同步模式由 `Server.round()` 和 `Aggregator.aggregate()` 联合实现。

### 2.1 等待比例与按时返回客户端集合

在 `Server.round()` 中：

1. 先通过 `ChannelSimulator.sample_round()` 为每个客户端生成本轮的信道统计：`snr_db`、`per`、`distance_m`。
2. 根据调度评分（能量、信道、数据价值、公平债务、带宽成本）选出 Top‑K 客户端 `selected`；
3. 用 `BandwidthAllocator` 按选中客户端平均分配带宽，并估算每个客户端的发送时间 `tx_time[cid]`；
4. 在半同步模式下，根据配置的 `semi_sync_wait_ratio`（例如 0.7）计算发送时间分位数：
   - 取 `tx_time` 的 70% 分位数作为“等待阈值”；
   - 将发送时间不超过阈值的客户端组成 `on_time_clients` 集合；
   - 在本轮训练中，只允许 `on_time_clients` 参与同步聚合，其余客户端视为“迟到”，本轮跳过。

直观地说，这就像设定一个“迟到线”：只要在合理时间内能把梯度传回来的客户端都算“按时”，其余则延后。

### 2.2 FedAvg 聚合与成本估计

对于按时返回的客户端：

1. 服务器将全局模型参数下发给每个客户端；
2. 客户端执行本地训练（算法为 FedAvg 或 FedProx），返回本地模型参数；
3. `Aggregator.aggregate()` 在 `sync`/`semi_sync` 模式下直接调用 `aggregate_fedavg`：
   - 按客户端样本数加权平均各自模型参数，得到新的全局参数；
4. 同时，`Server.round()` 会估算：
   - 本轮总通信时间 `comm_time`（按 1MB 载荷和分配带宽近似）；
   - 通信能耗 `comm_energy`（功率×时间）；
   - 计算能耗 `comp_energy`（按处理样本数近似）；
   - 并写入 `metrics.csv`，供后续 GA 优化与仪表盘使用。

半同步的效果是：在中低丢包时，绝大多数客户端都能“按时”，整体行为接近 FedAvg；在偶尔有少数慢节点时，可以不被拖累，从而提高单位时间内的有效训练轮数。

---

## 3. 异步聚合算法（FedBuff 风格，对应高丢包场景）

异步聚合由 `FedBuffBuffer`（`src/training/algorithms/fedbuff.py`）和 `Aggregator.aggregate()` 的 `async` 分支共同实现。

### 3.1 FedBuff 缓冲结构与陈旧度

`FedBuffBuffer` 内部维护一个队列 `entries = [(state_dict, staleness), ...]`：

- `state_dict`：客户端本地训练后的模型参数；
- `staleness`：该更新相对于当前全局轮次的“陈旧度”，即延迟的轮数。

在 `Aggregator.aggregate()` 的异步路径中：

1. 每轮开头，根据轮次差值调用 `buffer.age_entries(delta)`，将缓冲中所有条目的 `staleness` 增大 `delta`；若 `staleness` 超过 `max_staleness`，该条目会被丢弃；
2. 本轮新到的更新会通过 `buffer.push(sd, st)` 加入队列，初始 `staleness=0`；
3. 当满足以下任一条件时触发一次聚合：
   - 缓冲长度达到 `buffer_size`；
   - 自上次聚合以来累计更新数达到 `min_updates_to_aggregate`；
   - 距离上次聚合的轮数不小于 `async_agg_interval`。

### 3.2 陈旧度加权与聚合

触发聚合时，`buffer.aggregate(global_state)` 会：

1. 遍历缓冲中的所有条目 `(sd, staleness)`；
2. 为每个条目分配权重 `w = 1 / (1 + staleness)^alpha`（`alpha` 在配置中为 `staleness_alpha`）；
3. 按这些权重对所有 `state_dict` 做加权平均，得到新的全局模型；
4. 清空缓冲，为下一轮积累新更新。

这样，越“旧”的梯度（陈旧度大）对全局模型的影响越小，极旧的更新要么被淡化，要么被直接丢弃，从而保证异步训练在高丢包、高延迟环境下仍然收敛且不过度被陈旧信息拖累。

### 3.3 异步模式下的服务器行为

在 `Server.round()` 中，当策略控制器给出的模式为 `async` 时：

- 每轮仍然会进行调度、发模型、收更新，但收不到更新的客户端可以长期“缺席”；
- `Aggregator.aggregate()` 会将收到的更新累积到 FedBuff 缓冲中，只有缓冲“积累到一定程度”时才推进一次全局模型；
- 即便某些客户端长期丢包，只要还有一部分客户端能稳定上传更新，训练就能持续向前推进。

---

## 4. 桥接态与混合聚合：如何平滑完成策略切换？

桥接态（`bridge`）是本工程一个重要的“工程增强点”，对应论文中提到的：

> “权重混合聚合 + 公平债务 + 能量与带宽再平衡 + 不变量检查”的双模策略衔接。

在代码中，桥接态主要体现在两个地方：

1. `StrategyController` 将当前模式设为 `bridge`，并随轮数输出从 0→1 的 `bridge_weight`；
2. `Aggregator.aggregate()` 中，对于 `mode == 'bridge'` 的情况，会同时计算：
   - `sync_result`：本轮按时返回客户端的 FedAvg 同步结果；
   - `async_result`：FedBuff 缓冲当前给出的异步结果；
   - 最终全局模型：`mixed = (1-w)*sync_result + w*async_result`。

当 w 从 0 缓慢扫到 1 时，系统的行为从“几乎完全依赖半同步更新”逐渐过渡到“几乎完全依赖异步缓冲”，在曲线层面表现为：

- 精度曲线不会出现强烈的“断崖式跳变”；
- 通信时间与能耗的统计也会有一个缓慢的过渡过程，而不是瞬间翻倍或腰斩。

同时，`StrategyController` 中还通过：

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

- 本地训练：`src/training/client.py` 中的 `Client.local_train()`；
- 近端正则计算：`src/training/algorithms/fedprox.py` 中的 `fedprox_regularizer()`。

当配置 `training.algorithm: fedprox` 且 `fedprox_mu>0` 时：

- 客户端本地目标变为：`f_k(w) + μ/2 ||w - w_global||^2`；
- 代码中在交叉熵损失 `loss` 上叠加 `fedprox_regularizer(self.model, global_state, mu, device)`；
- 其中 `global_state` 是当前轮下发的全局参数，`μ` 由配置 `fedprox_mu` 控制。

直观理解：FedProx 会惩罚“偏离当前全局模型太远的本地更新”，从而在数据分布高度不均（non‑IID）时缓解局部模型过拟合自身数据的问题，有助于提高全局收敛稳定性。

### 5.2 SCAFFOLD：控制变元修正梯度漂移

相关代码：

- 控制变元状态：`src/training/algorithms/scaffold.py` 中的 `ScaffoldState`；
- SCAFFOLD 本地训练：`Client.local_train_scaffold()`；
- 服务器整合：`Server.__init__` 与 `Server.round()` 中处理 `algorithm == 'scaffold'` 的分支。

实现要点：

1. **控制向量初始化**：
   - 服务器创建 `ScaffoldState(self.global_state)`，生成与模型参数同形状的 `c_global`（全 0）和每个客户端的 `c_i`（懒初始化时为 0）。
2. **本地训练中的梯度校正**：
   - 在 `local_train_scaffold()` 中，客户端先从 `ScaffoldState` 取出 `c_global` 和自身的 `c_i`；
   - 正常反向传播得到梯度后，对每个参数执行：`grad ← grad + (c_i - c_global)`；
   - 这样更新规则变为 `w ← w - η (g - c + c_i)`，对应论文中的控制变元修正项，可显著减缓 non‑IID 导致的客户端漂移。
3. **Δc_i 的计算与回传**：
   - 本地训练结束后，客户端将最终本地参数 `w_local` 和训练步数 `num_steps` 交给 `ScaffoldState.compute_delta_ci`；
   - 该函数按论文的近似公式：`c_i^{new} = c_i - c + (1/(η·τ))(w_global - w_local)` 计算新的 `c_i`，并返回 Δc_i；
4. **服务器端更新全局控制向量**：
   - 在 `Server.round()` 聚合完模型参数之后，收集所有参与客户端的 Δc_i，按样本数加权平均；
   - 调用 `ScaffoldState.update_global(delta_list, weights)` 更新 `c_global`。

这样，FedProx 和 SCAFFOLD 分别从“损失函数层面”和“梯度层面”双重修正 non‑IID 带来的训练不稳定性，且都与门控切换和半同步/异步聚合兼容。

---

## 6. 改进型 MOGA‑FL：多目标遗传优化的工作流程

相关代码集中在 `src/ga/`：

- `objectives.py`：定义四个目标指标（精度、时间/轮数、公平性、能耗）；
- `constraints.py`：实现能耗预算等约束的惩罚与参数修复；
- `nsga3.py`：简化版 NSGA‑III 风格优化器（保持非支配解多样性）；
- `moead.py`：简化版 MOEA/D 风格优化器（基于权重向量的分解与邻域更新）；
- `moga_fl.py`：`MOGAFLController`，组合 NSGA‑III + MOEA/D + 岛屿模型 + 局部搜索 + 多保真评估；
- `scripts/run_ga_optimization.py`：GA 入口脚本，负责调用评估器并将最优解写回配置。

### 6.1 个体编码：要优化哪些参数？

在 `MOGAFLController._random_individual()` 中，一个候选解（染色体）包含：

- 调度评分权重：`energy_w, channel_w, data_w, fair_w, bwcost_w`，分别对应能量、信道质量、数据价值、公平债务与带宽成本；
- Top‑K 相关参数：`selection_top_k`（每轮选多少客户端）、`hysteresis`（选择防抖的迟滞系数）；
- 异步相关参数：`staleness_alpha`（陈旧度加权的指数）。

这些变量共同决定了：

- 哪些客户端更容易被选中参与训练；
- 半同步/异步切换时队列的稳定性；
- 在高丢包场景下，旧梯度对新模型的贡献大小。

### 6.2 多目标评估：四个维度的折衷

`objectives.evaluate_solution()` 封装了运行短轮联邦训练的过程：

- 给定一组参数，构造一个缩短轮数的“低保真模拟器”，运行若干轮训练；
- 输出四个目标：
  - `acc`：全局精度（希望越大越好）；
  - `time`：通信时间或轮数（希望越小越好）；
  - `fairness`：Jain 指数等公平性度量（希望越大越好）；
  - `energy`：能耗（通信 + 计算，越小越好）。

`constraints.penalty()` 会根据能耗是否超过预算等条件给出一个惩罚值，并在指标中略微拉高 `energy`，从而在优化过程中自然排斥超预算解；`repair()` 则负责修复不合法的参数（例如 Top‑K 至少为 1、迟滞在 [0, 0.2] 区间）。

### 6.3 NSGA‑III 岛屿：保持多样性

`NSGA3` 类实现了一个简化版 NSGA‑III 流程：

- 初始化种群，随机采样一批候选参数；
- 每一代：
  1. 用 `evaluate()` 计算每个个体的多目标指标和惩罚；
  2. 使用 `non_dominated_set()` 选出当前非支配解集合（即 Pareto 前沿）；
  3. 以这些非支配解为核心，随机交叉 + 变异生成下一代，保持种群多样性。

在 `MOGAFLController._run_nsga3_island()` 中，这个 NSGA‑III 岛屿被用作“广度优先探索器”：在较小代数下找到覆盖面较广的 Pareto 候选，为后续 MOEA/D 岛屿提供好的起点。

### 6.4 MOEA/D 岛屿：加速收敛

`MOEAD` 类实现了基于权重向量分解的简化 MOEA/D：

- 初始化一组权重向量 `w[i]`，每个向量代表一个“子问题”（不同的目标偏好组合）；
- 对于每个个体，使用 `scalarize(metrics, w[i])` 将多目标指标压缩为一个标量；
- 每一代内，在邻域中选择父代进行交叉和微小变异，如果子代的 scalar 值更好，就替换当前个体。

在 `MOGAFLController._run_moead_island()` 中，这个 MOEA/D 岛屿以 NSGA‑III 岛屿输出的精英解为初始种群的一部分，从而在保持多样性的前提下增强“局部收敛能力”。

### 6.5 岛屿模型 + 迁移 + 局部搜索

`MOGAFLController.run()` 将上述两个子算法组合成一个结构化的搜索流程：

1. 岛屿 1（NSGA‑III）：
   - 以随机种群为起点运行若干代，得到一批非支配解 `nsga_pop`；
2. 迁移：
   - 从 `nsga_pop` 中选出非支配前沿精英做“移民”，作为 MOEA/D 岛屿的初始种群一部分；
3. 岛屿 2（MOEA/D）：
   - 以“精英 + 随机填充”的方式初始化种群，运行若干代加强收敛；
4. 合并：
   - 将两个岛屿的候选合并为大种群 `all_pop`；
5. 局部搜索（Memetic）：
   - 在 `non_dominated_set(all_metrics_low)` 得到的低保真非支配精英附近做小范围扰动，得到邻域解 `neighbors`；
   - 扰动时对权重做微小高斯噪声并重新归一化，对 Top‑K、hysteresis、staleness_alpha 做轻微调整；
6. 多保真评估：
   - 对“精英 + 邻域”组合使用高保真评估器（完整轮数或更长训练）重新评估；
   - 再次取 Pareto 前沿，作为最终 MOGA‑FL 输出的候选解集合。

### 6.6 GA 结果如何部署到联邦学习模型？

入口脚本 `scripts/run_ga_optimization.py` 负责完成“从搜索到部署”的闭环：

1. 构造低保真与高保真评估器 `low_sim` / `high_sim`：
   - `low_sim`：缩短轮数、减少客户端数，用于快速粗评估；
   - `high_sim`：使用原配置的轮数和客户端数，对少数精英解高保真评估；
2. 调用 `MOGAFLController.run()` 获得 Pareto 候选解及其指标；
3. 根据 `eval.preference`（`time` / `fairness` / `energy` 等）对候选解做简单加权评分，选出一个“偏好最优”解；
4. 将该解中的参数写入 `outputs/results/best_moga_fl_config.yaml`：
   - `scheduling.weights.*`：覆盖能量/信道/数据价值/公平债务/带宽成本权重；
   - `clients.selection_top_k`、`clients.hysteresis`：覆盖调度 Top‑K 与防抖参数；
   - `training.fedbuff.staleness_alpha`：覆盖异步陈旧度加权参数；
5. 最终，用户可以直接使用这个 best 配置重新运行 `scripts/run_baselines.py`，即完成“经 GA 优化后的联邦训练部署”。

---

## 7. 总结：从“教学级 Demo”升级到“论文级原型”

通过上述模块，本工程已经实现了：

- 模型侧：从小 CNN 升级到 CIFAR‑10 / EMNIST 上的 ResNet‑18 变体（支持宽度缩放），在 quick 配置下使用“窄版 ResNet”，在完整配置下使用标准宽度；
- 训练侧：在 FedAvg 基础上，引入严格遵循论文定义的 FedProx 与控制变元版本的 SCAFFOLD；
- 聚合侧：实现了半同步、FedBuff 异步和桥接态混合聚合，并用门控 + 迟滞实现动态切换；
- 无线侧：引入基于路径损耗 + 阴影 + Rayleigh 衰落的统计信道模型，近似 DeepMIMO / NYUSIM / QuaDRiGa / 3GPP TR 38.901；
- 优化侧：实现了 NSGA‑III + MOEA/D + 岛屿模型 + 局部搜索 + 多保真评估的 MOGA‑FL，用于自动搜索调度与异步相关参数。

这些实现都在代码层面给出了清晰注释，明确指出“对应论文中的哪一部分思想”，便于在毕设论文和答辩中直接引用和讲解。用户可以围绕门控切换、半同步/异步聚合、遗传算法搜索与无线建模这四条主线，系统回答“我具体实现了什么复杂方法”。
