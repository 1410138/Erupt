import os
import json
import math
from typing import Dict, Optional, List

import torch
import torch.nn as nn
from sqlglot import expressions as exp
from udf_placement import _expr_sql
from config import SAMPLE_ROWS, SAMPLE_MODE, SAMPLE_PERCENT

TYPE_BUCKETS = [
    "BOOLEAN", "INTEGER", "HUGEINT", "FLOAT", "DOUBLE",
    "VARCHAR", "DATE", "TIME", "TIMESTAMP", "OTHER"
]
TYPE2IDX = {t: i for i, t in enumerate(TYPE_BUCKETS)}

def _one_hot_type(t: str) -> List[float]:
    idx = TYPE2IDX.get((t).upper())
    v = [0.0] * len(TYPE_BUCKETS)
    v[idx] = 1.0
    return v

def _type_histogram(arg_types: List[str]) -> List[float]:
    v = [0.0] * len(TYPE_BUCKETS)
    for t in arg_types:
        v[TYPE2IDX.get((t).upper())] += 1.0
    return v

def _neglog(x: float, eps: float = 1e-12) -> float:
    x = max(eps, min(1.0, float(x)))
    return float(-math.log(x))

def _neglog_pos(x: float, eps: float = 1e-12) -> float:
    x = max(eps, float(x))
    return float(-math.log(x))

class UdfFeatureProvider(nn.Module):
    def __init__(self, feature_file: Optional[str] = None, d_model: int = 128):
        super().__init__()
        # Output feature dimension.
        self.d = d_model
        # Hidden-layer dimension.
        self.hidden = 256
        self.feature_file = feature_file
        self.data: Dict[str, Dict] = {}
        if feature_file and os.path.exists(feature_file):
            with open(feature_file, "r", encoding="utf-8") as f:
                self.data = json.load(f)

        # 2 (input/output counts) + input-type histogram + output-type one-hot + 3 (selectivity, per-row cost, cyclomatic complexity).
        in_dim = 2 + len(TYPE_BUCKETS) + len(TYPE_BUCKETS) + 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.d),
        )

    def embed(self, t: Optional[str], expr: Optional[exp.Expression], device: str = "cpu") -> torch.Tensor:
        if SAMPLE_MODE == "rows":
            sample_config = str(SAMPLE_ROWS)
        else:
            sample_config = f"{SAMPLE_PERCENT}%"
        key = f"{t}||{_expr_sql(expr)}||{sample_config}"
        rec = self.data.get(key)

        num_in = float(rec.get("num_in"))
        num_out = float(rec.get("num_out"))
        # Input types are encoded as a histogram.
        in_hist = _type_histogram(rec.get("arg_types"))
        # Output type is one-hot encoded.
        out_oh = _one_hot_type(rec.get("ret_type"))
        p = float(rec.get("selectivity_p"))
        cpr = float(rec.get("cost_per_row"))
        cc = float(rec.get("cyclomatic_complexity"))

        x = []
        x += [num_in, num_out]
        x += in_hist
        x += out_oh
        x += [_neglog(p), _neglog_pos(cpr), cc]

        feat = torch.tensor(x, dtype=torch.float32, device=device)

        return self.mlp(feat)
