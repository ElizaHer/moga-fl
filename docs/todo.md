主要未对齐项（按严重级别）

高：缺少“调度评分 + Top‑K + 公平债务”主流程
当前 configure_fit 只是随机采样 fraction_fit 客户端，没有按文档中的能量/信道/数据价值/公平债务/带宽成本打分和 Top‑K 选择。
位置：hybrid_demo.py

高：未实现 bandwidth_factor（能量-带宽再平衡）
文档要求控制器输出 bandwidth_factor 并缩放每轮带宽预算；当前没有该输出，也没有动态改 BandwidthAllocator.budget_mb。
位置：hybrid_demo.py, hybrid_demo.py

中：FedBuff aging 逻辑与文档描述不一致（会过度老化）
现在每轮用 server_round - last_agg_round 去 age，但 last_agg_round 只在聚合后更新，导致多轮累计时老化增量被重复放大。
位置：hybrid_demo.py

中：异步触发条件缺少 async_agg_interval 配置项
当前固定 >=2 轮触发，不是文档里可配置的三条件之一（buffer_size / min_updates / async_agg_interval）。
位置：hybrid_demo.py

中：Jain 公平性统计口径不一致
当前用 selection_count 全历史累计计算 Jain，不是文档/原实现里“滑动窗口选择历史”。
位置：hybrid_demo.py, hybrid_demo.py

中：未对齐训练算法维度（FedProx/SCAFFOLD）
文档包含 FedAvg/FedProx/SCAFFOLD 兼容；当前客户端仅普通本地训练。
位置：hybrid_demo.py

低：无线配置是硬编码，不走配置系统
_base_wireless_cfg() 固定参数，不便于与现有 YAML/实验配置联动。
位置：hybrid_demo.py

低：未输出文档里提到的轮级记录（如 metrics.csv）
当前只返回最终摘要，未形成 GA/仪表盘直接复用的日志产物。
位置：hybrid_demo.py