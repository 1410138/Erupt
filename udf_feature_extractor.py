import os
import re
import json
import time
import argparse
import datetime
from typing import Dict, List, Tuple, Optional, Any

import duckdb
import math
import numpy
from sqlglot import expressions as exp

from config import DATASET_DIR, SAMPLE_ROWS, SAMPLE_MODE, SAMPLE_PERCENT, SAMPLE_MIN_ROWS
from udf_placement import parse_query, _rewrite_columns, _expr_sql

from radon.complexity import cc_visit

FUNC_PREFIX = re.compile(r'^func_\w+$', re.IGNORECASE)

TYPE_BUCKETS = [
    "BOOLEAN", "INTEGER", "HUGEINT", "FLOAT", "DOUBLE",
    "VARCHAR", "DATE", "TIME", "TIMESTAMP", "OTHER"
]

def _normalize_type(t: str) -> str:
    s = t.strip().upper()
    if s in TYPE_BUCKETS:
        return s
    else:
        raise ValueError(f"Unknown type: {s}")

class UdfRegistry:
    """
    - Source code for def func_x(...)
    - db_conn.create_function('func_x', func_x, [ARG_TYPES], RET_TYPE)
    """
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.udf_file = os.path.join(DATASET_DIR, db_name, f"{db_name}_udf.sql")
        self.raw = ""
        self.arg_types: Dict[str, List[str]] = {}
        self.ret_types: Dict[str, str] = {}
        self.func_code: Dict[str, str] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.udf_file):
            return
        with open(self.udf_file, "r", encoding="utf-8") as f:
            self.raw = f.read()

        regx = re.compile(
            r"create_function\(\s*['\"](?P<fname>func_[A-Za-z0-9_]+)['\"]\s*,\s*(?P<pyname>[A-Za-z0-9_]+)\s*,\s*\[(?P<args>[^\]]*)\]\s*,\s*(?P<ret>[A-Za-z0-9_\'\"]+)\s*\)",
            re.IGNORECASE,
        )
        for m in regx.finditer(self.raw):
            fname = m.group("fname").lower()
            args_raw = m.group("args").strip()
            ret_raw = m.group("ret").strip()
            ats = []
            if args_raw:
                for tok in args_raw.split(","):
                    tok = tok.strip().strip("'").strip('"')
                    if tok:
                        ats.append(_normalize_type(tok))
            self.arg_types[fname] = ats
            self.ret_types[fname] = _normalize_type(ret_raw.strip().strip("'").strip('"'))

        func_def_re = re.compile(r"^\s*def\s+(func_[A-Za-z0-9_]+)\s*\(", re.MULTILINE)
        starts: List[Tuple[str, int]] = []
        for m in func_def_re.finditer(self.raw):
            starts.append((m.group(1).lower(), m.start()))
        for i, (name, pos) in enumerate(starts):
            end = starts[i + 1][1] if i + 1 < len(starts) else len(self.raw)
            code = self.raw[pos:end]
            self.func_code[name] = code

    def arg_ret(self, func_name: str) -> Tuple[List[str], str]:
        name = func_name.lower()
        return list(self.arg_types.get(name, [])), self.ret_types.get(name, "OTHER")

    def cc_of(self, func_name: str) -> float:
        name = func_name.lower()
        code = self.func_code.get(name)
        if not code:
            return 0.0
        items = cc_visit(code)
        cand = [x for x in items if getattr(x, "name", "").lower() == name]
        if cand:
            return float(getattr(cand[0], "complexity", 0.0))
        return float(max((getattr(x, "complexity", 0.0) for x in items), default=0.0))

def load_udfs_into(con: duckdb.DuckDBPyConnection, code_chunks: List[str]):
    env = {
        "db_conn": con,
        "date": datetime.date,
        "time": datetime.time,
        "VARCHAR": duckdb.typing.VARCHAR,
        "INTEGER": duckdb.typing.INTEGER,
        "DOUBLE": duckdb.typing.DOUBLE,
        "TIMESTAMP": duckdb.typing.TIMESTAMP,
        "DATE": duckdb.typing.DATE,
        "TIME": duckdb.typing.TIME,
        "BOOLEAN": duckdb.typing.BOOLEAN,
        "HUGEINT": duckdb.typing.HUGEINT,
        "FLOAT": duckdb.typing.FLOAT,
        "math": math,
        "numpy": numpy,
        "re": re,
        "json": json,
    }
    for code in code_chunks:
        exec(code, env)

def extract_static_features(
    reg: UdfRegistry,
    expr: exp.Expression,
    profile_time: bool = False,
) -> Tuple[List[str], str, int, float, List[str], Dict[str, float]]:
    """Extract static UDF interface and cyclomatic-complexity features."""
    fns = [fn.name.lower() for fn in expr.find_all(exp.Func)]
    arg_types: List[str] = []
    ret_type = "OTHER"
    num_in = 0
    cc = 0.0
    timings: Dict[str, float] = {}

    if fns:
        t0 = time.perf_counter()
        arg_types, ret_type = reg.arg_ret(fns[0])
        num_in = len(arg_types)
        t1 = time.perf_counter()
        ccs = [reg.cc_of(fn) for fn in fns]
        cc = float(sum(ccs) / max(1, len(ccs)))
        t2 = time.perf_counter()
        if profile_time:
            timings = {
                "interface_extraction_time": t1 - t0,
                "complexity_extraction_time": t2 - t1,
                "static_extraction_time": t2 - t0,
            }

    return arg_types, ret_type, num_in, cc, fns, timings


def estimate_selectivity_and_cost(con: duckdb.DuckDBPyConnection, table: str, expr: exp.Expression, sample_config: str, idx: int) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    pred_sql = _expr_sql(_rewrite_columns(expr, "SampleTable"))
    
    if SAMPLE_MODE == "rows":
        sample_rows = SAMPLE_ROWS
        cte = f'WITH SampleTable AS (SELECT * FROM "{table}" USING SAMPLE reservoir({sample_rows} ROWS) REPEATABLE(42))'
    else:
        total_rows = int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        expected_rows = int(total_rows * SAMPLE_PERCENT / 100.0)
        if expected_rows < SAMPLE_MIN_ROWS:
            sample_rows = SAMPLE_MIN_ROWS
            cte = f'WITH SampleTable AS (SELECT * FROM "{table}" USING SAMPLE reservoir({sample_rows} ROWS) REPEATABLE(42))'
        else:
            cte = f'WITH SampleTable AS (SELECT * FROM "{table}" TABLESAMPLE BERNOULLI ({SAMPLE_PERCENT} PERCENT) REPEATABLE(42))'
    
    qtot = f"{cte} SELECT COUNT(*) FROM SampleTable;"
    qpred = f"{cte} SELECT COUNT(*) FROM SampleTable WHERE {pred_sql};"

    t0 = time.perf_counter()
    tot = int(con.execute(qtot).fetchone()[0])
    t1 = time.perf_counter()
    sel = int(con.execute(qpred).fetchone()[0])
    t2 = time.perf_counter()
    selectivity = max(0.0, min(1.0, float(sel) / float(tot)))
    extra = max(0.0, (t2 - t1) - (t1 - t0))
    cost_per_row = (extra / float(tot)) if tot > 0 else extra
    return selectivity, cost_per_row, t2 - t1, tot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", type=str, default="imdb")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--profile_time", action="store_true")
    args = parser.parse_args()

    db = args.db_name

    if SAMPLE_MODE == "rows":
        suffix = str(SAMPLE_ROWS)
    else:
        suffix = str(SAMPLE_PERCENT)
    out_file = args.out or os.path.join(DATASET_DIR, db, f"udf_features_{suffix}.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    sqls: List[str] = []
    for split in ("train", "val", "test"):
        sql_file = os.path.join(DATASET_DIR, db, f"{split}.sql")
        with open(sql_file, "r", encoding="utf-8") as f:
            sqls.extend(s for s in f.read().splitlines() if s.strip())
    
    db_file = os.path.join(DATASET_DIR, db, f"{db}_10_1.db")
    con = duckdb.connect(db_file)
    con.execute("PRAGMA disable_optimizer;")

    reg = UdfRegistry(db)
    load_udfs_into(con, [reg.func_code.get(fn) for fn in reg.func_code])
 
    if SAMPLE_MODE == "rows":
        sample_config = str(SAMPLE_ROWS)
    else:
        sample_config = f"{SAMPLE_PERCENT}%"
    
    feats: Dict[str, Dict[str, Any]] = {}
    for i, sql in enumerate(sqls):
        parsed = parse_query(sql)
        print(f"Processing {i+1} of {len(sqls)} sqls")
        for t, preds in parsed["udf_preds"].items():
            for e in preds:
                key = f"{t}||{e}||{sample_config}"
                if key in feats:
                    continue

                arg_types, ret_type, num_in, cc, fns, static_timings = extract_static_features(
                    reg, e, profile_time=args.profile_time
                )

                # Scalar UDFs have exactly one output.
                num_out = 1

                # Selectivity and cost.
                selectivity, cost_per_row, sampling_time, actual_sample_rows = estimate_selectivity_and_cost(con, t, e, sample_config, i)
                if not (cost_per_row >= 0.0):
                    cost_per_row = 0.0

                feat: Dict[str, Any] = {
                    "db": db,
                    "table": t,
                    "expr": _expr_sql(e),
                    "func_names": fns,
                    "num_in": num_in,           # Number of input arguments.
                    "num_out": num_out,         # Number of output values.
                    "arg_types": arg_types,     # Input argument types.
                    "ret_type": ret_type,       # Output type.
                    "selectivity_p": float(selectivity),  # Selectivity.
                    "cost_per_row": float(cost_per_row),  # Per-row cost.
                    "cyclomatic_complexity": float(cc),   # Cyclomatic complexity.
                    "sample_config": sample_config, # Sampling configuration.
                    "sample_rows": int(actual_sample_rows), # Actual sample size.
                    "sampling_time": float(sampling_time), # Sampling overhead.
                }
                if args.profile_time:
                    feat.update(static_timings)
                    feat["total_extraction_time"] = float(
                        static_timings.get("static_extraction_time", 0.0) + sampling_time
                    )
                feats[key] = feat
                
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
