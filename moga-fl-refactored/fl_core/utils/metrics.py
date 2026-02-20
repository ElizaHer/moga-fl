from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class MetricsRecorder:
    """轻量级指标收集器。

    与原工程的 `src/eval/metrics.py` 兼容，增加 dataclass 包装便于扩展。
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def to_csv(self, path: str) -> pd.DataFrame:
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return df
