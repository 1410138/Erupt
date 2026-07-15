import os
from typing import Dict, List, Tuple, Any, Optional

import duckdb
from sqlglot import expressions as exp

from selectivity import SelectivityProvider
from udf_placement import _expr_sql
from config import SAMPLE_ROWS, SAMPLE_MODE, SAMPLE_PERCENT
from utils import read_json, write_json

DUCKDB_DB_PATH = "datasets/imdb/imdb_10_1.db"
TABLE_STATS_PATH = "datasets/imdb/table_stats.json"
UDF_FEATURE_PATH = "datasets/imdb/udf_features.json"
POSTGRES_DB_NAME = "imdb"

class CostModel:
    def __init__(
        self,
        table_stats_path: str = TABLE_STATS_PATH,
        udf_feature_path: str = UDF_FEATURE_PATH,
        duckdb_path: str = DUCKDB_DB_PATH,
        pg_db_name: str = POSTGRES_DB_NAME,
        udf_beta: float = 20.0, # Hyperparameter controlling UDF cost.
        debug: bool = False,
        table_node_cls: Optional[type] = None,
        filter_node_cls: Optional[type] = None,
        join_node_cls: Optional[type] = None,
    ):
        self.TableNode = table_node_cls
        self.FilterNode = filter_node_cls
        self.JoinNode = join_node_cls

        self.duckdb_path = duckdb_path
        self.table_stats_path = table_stats_path
        self.udf_feature_path = udf_feature_path
        self.udf_beta = float(udf_beta)
        self.debug = bool(debug)

        if os.path.exists(self.table_stats_path):
            self.table_stats: Dict[str, Any] = read_json(self.table_stats_path)
        else:
            self.table_stats = self._compute_table_stats()
            os.makedirs(os.path.dirname(self.table_stats_path), exist_ok=True)
            write_json(self.table_stats_path, self.table_stats)

        if os.path.exists(self.udf_feature_path):
            self.udf_feats: Dict[str, Dict[str, Any]] = read_json(self.udf_feature_path)
        else:
            self.udf_feats = {}

        # Initialize the PostgreSQL selectivity estimator.
        self.sel_provider = SelectivityProvider(db_name=pg_db_name)

        self.max_cost_per_row = 0.0
        for rec in self.udf_feats.values():
            cpr = float(rec.get("cost_per_row", 0.0) or 0.0)
            if cpr > self.max_cost_per_row:
                self.max_cost_per_row = cpr
        if self.max_cost_per_row <= 0.0:
            self.max_cost_per_row = 1.0

    # ---------- Logging ----------

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(f"[CostModel] {msg}")

    # ---------- Table statistics ----------

    def _compute_table_stats(self) -> Dict[str, Any]:
        self._dbg("Computing table stats from DuckDB...")
        con = duckdb.connect(self.duckdb_path)
        try:
            rows = con.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                ORDER BY 1, 2
                """
            ).fetchall()

            stats: Dict[str, Any] = {"tables": {}}
            for schema, name in rows:
                t = str(name).split(".")[-1].lower()
                full_name = f'"{name}"'
                row_cnt = int(con.execute(f"SELECT COUNT(*) FROM {full_name}").fetchone()[0])
                cols = con.execute(f"PRAGMA table_info({full_name})").fetchall()
                ndv: Dict[str, int] = {}
                for col in cols:
                    col_name = str(col[1])
                    q_ndv = f'SELECT COUNT(DISTINCT "{col_name}") FROM {full_name}'
                    ndv_val = int(con.execute(q_ndv).fetchone()[0])
                    ndv[col_name.lower()] = ndv_val

                self._dbg(f"  table {t}: rows={row_cnt}, ndv_cols={len(ndv)}")

                stats["tables"][t] = {
                    "row_count": row_cnt,
                    "ndv": ndv,
                }

            return stats
        finally:
            con.close()

    def _base_rows(self, table: str) -> float:
        t = table.lower()
        info = self.table_stats.get("tables", {}).get(t)
        if not info:
            return 1.0
        return float(info.get("row_count", 1.0))

    def _ndv(self, table: str, column: str) -> float:
        t = table.lower()
        c = column.lower()
        info = self.table_stats.get("tables", {}).get(t)
        if not info:
            return 1.0
        ndv_map = info.get("ndv", {})
        v = ndv_map.get(c)
        if v is None or v <= 0:
            return 1.0
        return float(v)

    # ---------- UDF-related helpers ----------

    def _udf_key(self, table: str, expr: exp.Expression) -> str:
        if SAMPLE_MODE == "rows":
            sample_config = str(SAMPLE_ROWS)
        else:
            sample_config = f"{SAMPLE_PERCENT}%"
        return f"{table}||{_expr_sql(expr)}||{sample_config}"

    def _udf_record(self, table: str, expr: exp.Expression) -> Optional[Dict[str, Any]]:
        key = self._udf_key(table, expr)
        return self.udf_feats.get(key)

    def _udf_weights(self, rec: Optional[Dict[str, Any]]) -> Tuple[float, float]:
        if not rec:
            return 1.0, 0.0

        p = float(rec.get("selectivity_p", 1.0) or 1.0)
        p = max(1e-12, min(1.0, p))

        raw_cpr = float(rec.get("cost_per_row", 0.0) or 0.0)
        if raw_cpr <= 0.0:
            return p, 0.0

        x = raw_cpr / self.max_cost_per_row
        x = max(0.0, min(1.0, x))
        w = self.udf_beta * x
        self._dbg(
            f"UDF rec: p_udf={p:.4g}, raw_cpr={raw_cpr:.4g}, max_cpr={self.max_cost_per_row:.4g}, w_udf={w:.4g}"
        )
        return p, w

    def _estimate_base_table(self, table: str, local_preds: List[exp.Expression]) -> Tuple[float, float]:
        """
        Estimate the combined selectivity of non-UDF predicates with PostgreSQL's SelectivityProvider:
        """
        base_rows = self._base_rows(table)
        self._dbg(f"Estimate base table {table}: base_rows={base_rows:.2f}")

        if not local_preds:
            return base_rows, base_rows

        p_total = 1.0
        for e in local_preds:
            try:
                p_e = float(self.sel_provider.estimate_selectivity(table, "", e))
            except Exception:
                p_e = 1.0
            p_e = max(1e-12, min(1.0, p_e))
            p_total *= p_e
            self._dbg(f"  predicate {e}: p_e={p_e:.4g}")

        p_total = max(1e-12, min(1.0, p_total))
        r_base = base_rows * p_total
        c_base = r_base
        self._dbg(f"  p_total={p_total:.4g}, R_base={r_base:.2f}, C_base={c_base:.2f}")
        return r_base, c_base

    def _estimate_join_rows(self, left_rows: float, right_rows: float, on_exprs: List[exp.Expression]) -> float:
        self._dbg(
            f"Estimate join rows: left_rows={left_rows:.2f}, right_rows={right_rows:.2f}, on_exprs={len(on_exprs)}"
        )

        if left_rows <= 0 or right_rows <= 0:
            return 0.0

        if not on_exprs:
            return left_rows * right_rows

        e = on_exprs[0]
        cols = list(e.find_all(exp.Column))
        if len(cols) < 2:
            return left_rows * right_rows

        c1, c2 = cols[0], cols[1]
        t1 = (c1.table or "").lower()
        t2 = (c2.table or "").lower()
        n1 = (c1.name or "").lower()
        n2 = (c2.name or "").lower()

        ndv1 = self._ndv(t1, n1)
        ndv2 = self._ndv(t2, n2)
        denom = max(ndv1, ndv2, 1.0)

        rj = (left_rows * right_rows) / denom
        rj = max(0.0, min(left_rows * right_rows, rj))
        self._dbg(
            f"  join on {t1}.{n1} vs {t2}.{n2}: ndv1={ndv1:.2f}, ndv2={ndv2:.2f}, R_join={rj:.2f}"
        )
        return rj

    def estimate_plan(self, plan, parsed: Dict[str, Any]) -> Tuple[float, float]:
        local_preds: Dict[str, List[exp.Expression]] = parsed.get("local_preds", {})

        def rec(node, depth: int = 0) -> Tuple[float, float]:
            indent = "  " * depth
            # Leaf: TableNode.
            if self.TableNode is not None and isinstance(node, self.TableNode):
                t = node.t
                preds = local_preds.get(t, [])
                rows, cost = self._estimate_base_table(t, preds)
                self._dbg(f"{indent}Node Table({t}): rows={rows:.2f}, cost={cost:.2f}")
                return rows, cost

            # Unary node: FilterNode.
            if self.FilterNode is not None and isinstance(node, self.FilterNode):
                child_rows, child_cost = rec(node.child, depth + 1)
                rec_udf = self._udf_record(node.t, node.expr)
                p_udf, w_udf = self._udf_weights(rec_udf)

                r_in = child_rows
                c_udf = w_udf * r_in
                r_out = r_in * p_udf
                c_out = child_cost + c_udf
                self._dbg(
                    f"{indent}Node Filter(UDF on {node.t}): R_in={r_in:.2f}, p_udf={p_udf:.4g}, "
                    f"w_udf={w_udf:.4g}, R_out={r_out:.2f}, C_out={c_out:.2f}"
                )
                return r_out, c_out

            # Binary node: JoinNode.
            if self.JoinNode is not None and isinstance(node, self.JoinNode):
                left_rows, left_cost = rec(node.left, depth + 1)
                right_rows, right_cost = rec(node.right, depth + 1)

                rj = self._estimate_join_rows(left_rows, right_rows, node.on_exprs)
                c_join = rj

                r_cur = rj
                c_cur = left_cost + right_cost + c_join
                self._dbg(
                    f"{indent}Node Join: R_L={left_rows:.2f}, C_L={left_cost:.2f}, "
                    f"R_R={right_rows:.2f}, C_R={right_cost:.2f}, "
                    f"R_join={rj:.2f}, C_join={c_join:.2f}"
                )

                for (t, _eid, e) in getattr(node, "post_filters", []) or []:
                    rec_udf = self._udf_record(t, e)
                    p_udf, w_udf = self._udf_weights(rec_udf)
                    r_in = r_cur
                    c_udf = w_udf * r_in
                    r_cur = r_in * p_udf
                    c_cur += c_udf
                    self._dbg(
                        f"{indent}  Post-UDF on {t}: R_in={r_in:.2f}, p_udf={p_udf:.4g}, "
                        f"w_udf={w_udf:.4g}, R_out={r_cur:.2f}, C_cur={c_cur:.2f}"
                    )

                self._dbg(f"{indent}Join node final: R_out={r_cur:.2f}, C_out={c_cur:.2f}")
                return r_cur, c_cur

            self._dbg(f"{indent}Unknown node type {type(node)}, treat as zero.")
            return 0.0, 0.0

        return rec(plan, 0)


__all__ = ["CostModel"]
