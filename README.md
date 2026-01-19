# 联邦学习无线异构边缘调度代码工程（MOGA-FL Wireless FL Scheduler）

本项目提供一个面向无线异构边缘场景的联邦学习调度与训练的完整可复现工程，支持在单机 1×T4/3060 或 CPU 上运行。包含数据加载与非IID划分、轻量无线仿真（块衰落→丢包概率，带宽预算、能耗≈功率×时间）、客户端多指标评分+Top‑K 选择（含公平债务）、联邦训练基线（FedAvg/FedProx/SCAFFOLD、简化版FedBuff异步）、通信压缩占位（量化/Top‑K+误差反馈）、以及改进型多目标遗传算法（简化版NSGA‑III/MOEA/D框架）优化评分权重、Top‑K、带宽与陈旧度权重等。

- 项目结构：见下文“目录结构”。
- 快速运行：见“快速开始”。
- 配置说明与复现实验：见“配置与实验”。

## 目录结构
- src/
  - configs/config.py：加载 YAML 配置与默认合并
  - data/：数据集加载与非IID划分
  - wireless/：信道仿真、带宽分配、能耗近似
  - scheduling/：评分、选择、防抖、公平债务
  - training/：模型、算法、聚合器、压缩占位
  - ga/：多目标遗传优化（简化NSGA‑III/MOEA/D）
  - eval/：指标、日志与绘图
  - utils/：随机种子、计时、注册器
- configs/
  - default_cifar10.yaml
  - default_emnist.yaml
  - quick_cifar10.yaml
  - quick_emnist.yaml
- scripts/
  - run_baselines.py：批量运行与对比
  - run_ga_optimization.py：遗传搜索与候选部署
- data/：数据缓存与划分存档
- outputs/
  - logs/：运行日志
  - results/：CSV结果
  - plots/：图表
- tests/：最小单元测试（pytest）
- requirements.txt：依赖

## 环境要求
- Python ≥ 3.9
- 建议：CUDA 11.x + PyTorch 2.x（可 CPU 运行）

安装：
```
pip install -r requirements.txt
```

## 快速开始（单机可复现，≤5分钟）
- 使用“快速模式”与 FakeData 回退（若无法联网下载数据）生成样例结果：
```
python scripts/run_baselines.py --config configs/quick_cifar10.yaml
```
生成：
- outputs/results/*（accuracy.csv、fairness.csv、costs.csv 等）
- outputs/plots/*（精度/能耗/公平性图表）

批量对比（FedAvg/FedProx/SCAFFOLD/FedBuff）：
```
python scripts/run_baselines.py --config configs/quick_cifar10.yaml --algos fedavg fedprox scaffold fedbuff
```

遗传优化（快速评估，自动搜索评分权重、Top‑K、带宽与陈旧度权重）：
```
python scripts/run_ga_optimization.py --config configs/quick_cifar10.yaml --generations 6 --pop 16
```
完成后会在 outputs/results/ 下生成 pareto_candidates.csv 与偏好部署示例。

## 可视化仪表盘
- 本项目提供一个无需打包工具的静态可视化页面，位于 `dashboards/` 目录。
- 页面聚合展示：
  - 训练过程曲线：精度 vs 轮次、能耗 vs 轮次、Jain 公平指数 vs 轮次；
  - MOGA‑FL Pareto 候选点：可视化精度 / 时间 / 公平 / 能耗之间的折衷，并查看每个候选点的评分权重与 Top‑K 等参数；
  - 运行统计：每轮选中客户端数量及当前运行的概览指标。

**生成 / 刷新数据：**
1. 运行联邦训练基线，生成最新的 `outputs/results/metrics.csv`：
   ```bash
   python scripts/run_baselines.py --config configs/quick_cifar10.yaml
   python scripts/run_baselines.py --config configs/default_cifar10.yaml
   ```
2. （可选）运行遗传优化，生成 `outputs/results/pareto_candidates.csv`：
   ```bash
   python scripts/run_ga_optimization.py --config configs/quick_cifar10.yaml
   ```
3. 调用预处理脚本，将 CSV 汇总为前端使用的 JSON：
   ```bash
   python scripts/prepare_dashboard_data.py
   ```
   该脚本会在 `dashboards/` 中生成 `metrics_summary.json` 和 `pareto_summary.json`，仪表盘页面通过这两个文件加载数据。

**本地查看仪表盘：**
1. 在项目根目录执行：
   ```bash
   cd dashboards
   python -m http.server 8000
   ```
2. 在浏览器访问 `http://localhost:8000`，即可看到“无线边缘联邦学习调度实验面板”页面。若直接用 `file://` 打开 HTML，部分浏览器会禁止 `fetch` 读取本地 JSON，建议使用上述简易 HTTP 服务。

## 配置与实验
- 数据集：CIFAR‑10、EMNIST（balanced）；默认自动下载至 data/（若失败自动回退 FakeData）。
- 非IID划分：Dirichlet（alpha）、标签偏置（每客户端限标签数）、数量偏置（样本量不均）。
- 无线仿真：块衰落强度（0.3–1.5）、基础 SNR dB（5–25）、丢包映射 per=exp(−k·SNR_lin)，带宽预算（Mb/轮）、能耗≈功率×时间（通信/计算）。
- 调度评分：能量充足度、链路质量、数据价值（新颖度/尾部覆盖）、公平债务、带宽成本；Top‑K 与防抖（滑窗5–10、迟滞阈值0.03–0.08）。
- 训练：FedAvg/FedProx/SCAFFOLD（同步/半同步）、简化版 FedBuff（异步缓冲，陈旧度加权），压缩占位：8bit 量化、Top‑K+误差反馈。
- 评估：准确率、收敛轮数/时间、Jain’s 公平指数、通信与能耗成本、鲁棒性（丢包率/异构强度变化）。

完整运行（下载真实数据，训练更久）示例：
```
python scripts/run_baselines.py --config configs/default_cifar10.yaml
```
建议在 1×T4/3060 上运行（完整模式可能需 20–60 分钟）。

## 常见问题（FAQ）
- 无法下载数据：使用 quick_* 配置自动回退 FakeData。或手动提前下载 torchvision 数据集。
- 运行慢：将 local_epochs 降为 1、clients=10–20、samples_per_client 降低；使用 CPU 也可运行但更慢。
- 图表中文乱码：已在代码中设置 Noto Sans CJK SC；若依然有警告，请安装字体并检查路径。
