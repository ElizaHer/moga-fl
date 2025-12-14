import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import ast


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    v = value.strip()
    if v == "" or v.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """读取 metrics.csv 并整理为前端友好的 JSON 结构。

    输出结构示例：
    {
      "runs": [
        {
          "run_id": "default",
          "display_name": "默认运行 (outputs/results/metrics.csv)",
          "metrics": [
            {"round": 0, "accuracy": 0.1, "comm_time": 4.5, ...}
          ]
        }
      ]
    }
    """
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    runs: Dict[str, Dict[str, Any]] = {}

    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # 支持未来扩展：如存在 run_id/config 字段，则按不同 run 分组
        if "run_id" in fieldnames:
            run_key_field = "run_id"
        elif "config" in fieldnames:
            run_key_field = "config"
        else:
            run_key_field = None

        for idx, row in enumerate(reader):
            if run_key_field is None:
                run_id = "default"
            else:
                run_id = row.get(run_key_field) or "run_0"

            bucket = runs.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "display_name": run_id if run_key_field else "默认运行 (metrics.csv)",
                    "metrics": [],
                },
            )

            m: Dict[str, Any] = {}

            # 轮次
            r_raw = row.get("round")
            try:
                m["round"] = int(r_raw) if r_raw is not None and r_raw != "" else idx
            except ValueError:
                m["round"] = idx

            # 关键指标
            acc = _safe_float(row.get("accuracy"))
            m["accuracy"] = acc

            comm_time = _safe_float(row.get("comm_time")) or 0.0
            comm_energy = _safe_float(row.get("comm_energy")) or 0.0
            comp_energy = _safe_float(row.get("comp_energy")) or 0.0
            jain = _safe_float(row.get("jain_index"))

            m["comm_time"] = comm_time
            m["comm_energy"] = comm_energy
            m["comp_energy"] = comp_energy
            m["total_energy"] = comm_energy + comp_energy
            m["jain_index"] = jain

            # 选中客户端列表（若有）
            selected_raw = row.get("selected")
            selected: Optional[List[int]] = None
            if selected_raw:
                try:
                    parsed = ast.literal_eval(selected_raw)
                    if isinstance(parsed, (list, tuple)):
                        selected = [int(x) for x in parsed]
                except (ValueError, SyntaxError):
                    selected = None
            m["selected"] = selected

            # 记录其它额外字段，方便后续扩展，例如 sync_mode / avg_per / bandwidth_factor 等
            extra: Dict[str, Any] = {}
            for key, value in row.items():
                if key in {"round", "selected", "accuracy", "comm_time", "comm_energy", "comp_energy", "jain_index"}:
                    continue
                if key == run_key_field:
                    continue
                if value is None or value == "":
                    continue
                # 尽量转为数值
                num = _safe_float(value)
                extra[key] = num if num is not None else value
            if extra:
                m["extra"] = extra

            bucket["metrics"].append(m)

    # 按轮次排序，避免 CSV 打乱
    for run in runs.values():
        run["metrics"] = sorted(run["metrics"], key=lambda x: x.get("round", 0))

    return {"runs": list(runs.values())}


def load_pareto(pareto_path: Path) -> Dict[str, Any]:
    """读取 pareto_candidates.csv，整理成简单列表结构。"""
    if not pareto_path.exists():
        # 允许 GA 尚未运行的情况
        return {"solutions": []}

    solutions: List[Dict[str, Any]] = []
    with pareto_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for idx, row in enumerate(reader):
            s: Dict[str, Any] = {"id": idx}
            for key in fieldnames:
                val = row.get(key)
                if val is None or val == "":
                    continue
                num = _safe_float(val)
                s[key] = num if num is not None else val
            solutions.append(s)

    return {"solutions": solutions, "fields": fieldnames}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare JSON data for FL dashboard.")
    parser.add_argument("--metrics", type=str, default="outputs/results/metrics.csv", help="Path to metrics.csv")
    parser.add_argument("--pareto", type=str, default="outputs/results/pareto_candidates.csv", help="Path to pareto_candidates.csv")
    parser.add_argument("--out-dir", type=str, default="dashboards", help="Directory to write JSON files")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    pareto_path = Path(args.pareto)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    metrics_summary = load_metrics(metrics_path)
    pareto_summary = load_pareto(pareto_path)

    metrics_out = out_dir / "metrics_summary.json"
    pareto_out = out_dir / "pareto_summary.json"

    # 使用 json.dumps + 手动写入，避免 NaN 等非标准值
    metrics_json = json.dumps(metrics_summary, ensure_ascii=False, indent=2)
    pareto_json = json.dumps(pareto_summary, ensure_ascii=False, indent=2)

    metrics_out.write_text(metrics_json, encoding="utf-8")
    pareto_out.write_text(pareto_json, encoding="utf-8")

    print(f"Saved metrics summary to {metrics_out}")
    print(f"Saved pareto summary to {pareto_out}")


if __name__ == "__main__":
    main()
