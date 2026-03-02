from __future__ import annotations 


import argparse 
import os 
import re 
from datetime import datetime 
from typing import Dict, List, Optional, Tuple 


import matplotlib.pyplot as plt 
import pandas as pd 



TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})(?=\.csv$)") 



def parse_args() -> argparse.Namespace: 
    parser = argparse.ArgumentParser(description="Plot hybrid vs baselines from latest metrics CSVs") 
    parser.add_argument("--metrics-root", type=str, default="outputs/fl_comp/20260228_033709/B_matrix_mu0p01") 
    parser.add_argument("--out-dir", type=str, default="outputs/fl_comp/20260228_033709/analysis/plots") 
    return parser.parse_args() 



def _csv_timestamp(name: str) -> Optional[datetime]: 
    match = TIMESTAMP_RE.search(name) 
    if not match: 
        return None 
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S") 



def latest_csv_in_dir(folder: str) -> Optional[str]: 
    if not os.path.isdir(folder): 
        return None 
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")] 
    if not files: 
        return None 


    dated: List[Tuple[datetime, str]] = [] 
    undated: List[str] = [] 
    for name in files: 
        ts = _csv_timestamp(name) 
        if ts is None: 
            undated.append(name) 
        else: 
            dated.append((ts, name)) 


    if dated: 
        dated.sort(key=lambda x: x[0], reverse=True) 
        return os.path.join(folder, dated[0][1]) 


    undated.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)), reverse=True) 
    return os.path.join(folder, undated[0]) 


def get_network_type(strategy: str) -> str: 
    """从策略名称中提取网络类型"""
    if "wsn" in strategy: 
        return "wsn"
    elif "jitter" in strategy: 
        return "jitter"
    else: 
        return "unknown"


def get_strategy_type(strategy: str) -> str: 
    """从策略名称中提取策略类型"""
    if "wsn" in strategy: 
        return strategy.replace("_wsn", "")
    elif "jitter" in strategy: 
        return strategy.replace("_jitter", "")
    else: 
        return strategy


def get_color(strategy_type: str) -> str:
    """根据策略类型返回颜色"""
    # 固定的颜色映射，确保每个方法对应唯一的颜色
    # 暖色系颜色（用于 hybrid 方法）
    hybrid_colors = {
        'hybrid_invFalse': '#E74C3C',  # 红色
        'hybrid_invTrue': '#E67E22',   # 橙色
        'hybrid': '#F39C12',           # 黄色
    }
    
    # 冷色系颜色（用于其他基线方法）
    baseline_colors = {
        'sync': '#3498DB',       # 蓝色
        'async': '#2ECC71',      # 绿色
        'bridge_free': '#9B59B6', # 紫色
        'bandwidth_first': '#1ABC9C', # 青色
        'energy_first': '#34495E', # 深灰色
        'bridgefree': '#95A5A6',  # 灰色
        'bwfirst': '#BDC3C7',     # 浅灰色
        'energyfirst': '#7F8C8D', # 中灰色
    }
    
    # 检查是否是 hybrid 方法
    if strategy_type.startswith('hybrid'):
        # 从 hybrid_colors 中查找，找不到则使用默认暖色系
        for key in hybrid_colors:
            if key in strategy_type:
                return hybrid_colors[key]
        # 默认暖色系颜色
        return '#E74C3C'
    else:
        # 从 baseline_colors 中查找，找不到则使用默认冷色系
        for key in baseline_colors:
            if key in strategy_type:
                return baseline_colors[key]
        # 默认冷色系颜色
        return '#3498DB'

def load_latest_metrics(metrics_root: str) -> Dict[str, pd.DataFrame]: 
    data: Dict[str, pd.DataFrame] = {} 
    # 遍历 metrics_root 下的所有子目录
    print(f"[INFO] Scanning directory: {metrics_root}")
    print(f"[INFO] Subdirectories: {os.listdir(metrics_root)}")
    for strategy in os.listdir(metrics_root): 
        folder = os.path.join(metrics_root, strategy) 
        if not os.path.isdir(folder): 
            continue 
        csv_path = latest_csv_in_dir(folder) 
        if csv_path is None: 
            print(f"[WARN] No CSV found for strategy: {strategy} ({folder})") 
            continue 
        df = pd.read_csv(csv_path) 
        if "round" not in df.columns: 
            print(f"[WARN] Missing 'round' column in {csv_path}, skip") 
            continue 
        data[strategy] = df.sort_values("round") 
        print(f"[INFO] {strategy}: {csv_path}") 
        print(f"[INFO] Columns: {list(df.columns)}")
    return data


def plot_metric( 
    data: Dict[str, pd.DataFrame], 
    network_type: str, 
    metric_col: str, 
    ylabel: str, 
    out_path: str, 
) -> None: 
    plt.figure(figsize=(10, 6)) 
    plotted_strategies = []
    for strategy, df in data.items(): 
        if get_network_type(strategy) != network_type: 
            continue 
        if metric_col not in df.columns: 
            print(f"[WARN] '{metric_col}' not in {strategy}, skip this line") 
            continue 
        strategy_type = get_strategy_type(strategy)
        color = get_color(strategy_type)
        plt.plot(df["round"], df[metric_col], color=color, label=strategy_type) 
        plotted_strategies.append(strategy)
    print(f"[INFO] Plotted strategies for {metric_col} ({network_type}): {plotted_strategies}")
    plt.xlabel("Round") 
    plt.ylabel(ylabel) 
    plt.title(f"{ylabel} Comparison ({network_type})") 
    plt.grid(True, linestyle="--", alpha=0.4) 
    plt.legend() 
    plt.tight_layout() 
    plt.savefig(out_path, dpi=150) 
    plt.close() 
    print(f"[OK] Saved: {out_path}")


def main() -> None: 
    args = parse_args() 
    os.makedirs(args.out_dir, exist_ok=True) 


    data = load_latest_metrics(args.metrics_root) 
    if not data: 
        raise RuntimeError("No valid metrics CSV found. Please run strategies first.") 


    # 定义要绘制的指标
    metric_specs = [ 
        ("accuracy", "Accuracy", "acc"), 
        ("loss", "Loss", "loss"), 
        ("energy", "Energy", "energy"), 
        ("est_upload_time", "Total Upload Time", "upload_time"), 
        ("jain", "Fairness (Jain)", "fairness"), 
    ] 
    
    # 为每个网络类型生成图表
    for network_type in ["wsn", "jitter"]: 
        for col, ylabel, prefix in metric_specs: 
            filename = f"{prefix}_comparison_{network_type}.png"
            plot_metric(data, network_type, col, ylabel, os.path.join(args.out_dir, filename)) 



if __name__ == "__main__": 
    main()