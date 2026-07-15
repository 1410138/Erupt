from sqlglot import parse_one, expressions as exp
import re
from typing import List, Dict, Tuple, Set, FrozenSet, Optional

FUNC_PREFIX = re.compile(r'^func_\w+$', re.IGNORECASE)

def _split_conjuncts(e: exp.Expression) -> List[exp.Expression]:
    # unwrap WHERE and parentheses first
    if isinstance(e, exp.Where):
        return _split_conjuncts(e.this)
    if isinstance(e, exp.Paren):
        return _split_conjuncts(e.this)
    # split by AND
    if isinstance(e, exp.And):
        return _split_conjuncts(e.this) + _split_conjuncts(e.expression)
    return [e]

def _ref_tables(e: exp.Expression) -> Set[str]:
    tabs = set()
    for c in e.find_all(exp.Column):
        if c.table:
            tabs.add(str(c.table).lower())
    return tabs

def _has_udf(e: exp.Expression) -> bool:
    for f in e.find_all(exp.Func):
        name = (f.name or "").lower()
        if FUNC_PREFIX.match(name):
            return True
    return False

def _expr_sql(e: exp.Expression) -> str:
    return e.sql(dialect="duckdb")

def _rewrite_columns(e: exp.Expression, to_alias: Optional[str]) -> exp.Expression:
    # Rewrite column qualifiers to a fixed alias: 'LHS' / 'RHS' / 'CUR', or remove if None
    e = e.copy()
    for c in e.find_all(exp.Column):
        if to_alias is None:
            c.set("table", None)
        else:
            c.set("table", exp.to_identifier(to_alias))
    return e

def sanitize_ident(s: str) -> str:
    s = re.sub(r'[^A-Za-z0-9_]', '_', s)
    if not re.match(r'[A-Za-z_]', s):
        s = f"t_{s}"
    return s

# --------------------------------
# Parse: tables, join edges, predicates
# --------------------------------

def parse_query(sql: str):
    """
    Extract:
    - tables: tables referenced in the query
    - join_edges: edges only from explicit JOIN ... ON
    - local_preds: single-table simple predicates (from WHERE)
    - udf_preds: single-table udf predicates (from WHERE)
    """
    q = parse_one(sql, read="duckdb")

    # tables
    tables: Set[str] = set()
    for t in q.find_all(exp.Table):
        tables.add(t.name.lower())
    tables = set(tables)

    # join edges only from JOIN ... ON
    join_edges: Dict[FrozenSet[str], List[exp.Expression]] = {}

    def add_edge(a: str, b: str, predicate: exp.Expression):
        key = frozenset([a, b])
        join_edges.setdefault(key, []).append(predicate)

    for j in q.find_all(exp.Join):
        on_expr = j.args.get("on")
        if on_expr:
            for c in _split_conjuncts(on_expr):
                tabs = _ref_tables(c)
                if len(tabs) == 2:
                    ta, tb = list(tabs) 
                    add_edge(ta, tb, c)

    # WHERE: only single-table predicates
    local_preds: Dict[str, List[exp.Expression]] = {t: [] for t in tables}
    udf_preds: Dict[str, List[exp.Expression]] = {t: [] for t in tables}

    where_node = q.args.get("where")
    if where_node:
        cond = where_node.this  # exp.Where -> its inner boolean expression
        for c in _split_conjuncts(cond):
            tabs = _ref_tables(c)
            if len(tabs) == 1:
                t = list(tabs)[0]
                if _has_udf(c):
                    udf_preds[t].append(c)
                else:
                    local_preds[t].append(c)
            else:
                # ignore multi-table predicates in WHERE
                pass

    return {
        "tables": sorted(tables),
        "join_edges": join_edges,
        "local_preds": local_preds,
        "udf_preds": udf_preds,
    }

# Node id: table -> 'T:table'; udf -> 'U:table:eid'
def make_node_id_for_table(t: str) -> str:
    return f"T:{t}"

def make_node_id_for_udf(t: str, eid: int) -> str:
    return f"U:{t}:{eid}"

class PlanNode:
    kind: str
    def __init__(self, kind: str):
        self.kind = kind
    def tables(self) -> Set[str]: ...
    def nodes(self) -> Set[str]: ...
    def __str__(self) -> str: ...

class TableNode(PlanNode):
    def __init__(self, t: str):
        super().__init__("TABLE")
        self.t = t
    def tables(self): return {self.t}
    def nodes(self): return {make_node_id_for_table(self.t)}
    def __str__(self): return self.t

class JoinNode(PlanNode):
    def __init__(self, left: PlanNode, right: PlanNode,
                 on_exprs: List[exp.Expression],
                 post_filters: List[Tuple[str, int, exp.Expression]]):
        super().__init__("JOIN")
        self.left = left
        self.right = right
        self.on_exprs = on_exprs
        self.post_filters = post_filters  # [(table, eid, expr), ...]
    def tables(self): return self.left.tables().union(self.right.tables())
    def nodes(self): return self.left.nodes().union(self.right.nodes())
    def __str__(self):
        return f"({self.left} ⋈ {self.right})"

class FilterNode(PlanNode):
    def __init__(self, child: PlanNode, t: str, eid: int, expr: exp.Expression):
        super().__init__("FILTER")
        self.child = child
        self.t = t
        self.eid = eid
        self.expr = expr
    def tables(self): return self.child.tables()
    def nodes(self): return self.child.nodes().union({make_node_id_for_udf(self.t, self.eid)})
    def __str__(self): return f"({self.child} ▷ e({self.t}#{self.eid}))"

def canonical_shape(node) -> str:
    # Table
    if isinstance(node, TableNode):
        return node.t
    # UDF
    if isinstance(node, FilterNode):
        child = canonical_shape(node.child)
        return f"({child} ▷ e({node.t}#{node.eid}))"
    # Join
    if isinstance(node, JoinNode):
        l = canonical_shape(node.left)
        r = canonical_shape(node.right)
        if l <= r:
            return f"({l} ⋈ {r})"
        else:
            return f"({r} ⋈ {l})"
    raise ValueError("unknown node type for shape")

# --------------------------------
# CTE generation
# --------------------------------

def build_ctes_for_tables(tables: List[str], local_preds: Dict[str, List[exp.Expression]]) -> Dict[str, str]:
    base_cte_name = {}
    for t in tables:
        base_cte_name[t] = sanitize_ident(f"{t}_1")
    return base_cte_name

def build_sql_from_plan(plan: PlanNode,
                        base_cte_name: Dict[str, str],
                        join_edges: Dict[FrozenSet[str], List[exp.Expression]]) -> str:
    ctes: List[Tuple[str, str]] = []
    used_tables = sorted(base_cte_name.keys())

    # 1) base CTEs with local predicates
    def emit_base_ctes(local_preds: Dict[str, List[exp.Expression]]):
        for t in used_tables:
            name = base_cte_name[t]
            where_parts = [_expr_sql(_rewrite_columns(p, None)) for p in local_preds.get(t, [])]
            if where_parts:
                body = f"SELECT * FROM {t} WHERE " + " AND ".join(where_parts)
            else:
                body = f"SELECT * FROM {t}"
            ctes.append((name, body))

    # 2) recursive emit
    seq = {"n": 0}
    def new_name(prefix: str) -> str:
        seq["n"] += 1
        return sanitize_ident(f"{prefix}_{seq['n']}")

    def gen(node: PlanNode, local_preds: Dict[str, List[exp.Expression]]) -> str:
        if isinstance(node, TableNode):
            return base_cte_name[node.t]

        if isinstance(node, FilterNode):
            child_name = gen(node.child, local_preds)
            name = new_name(f"{child_name}_e")
            pred = _expr_sql(_rewrite_columns(node.expr, "CUR"))
            ctes.append((name, f"SELECT * FROM {child_name} AS CUR WHERE {pred}"))
            return name

        if isinstance(node, JoinNode):
            left_name = gen(node.left, local_preds)
            right_name = gen(node.right, local_preds)
            name = new_name("j")

            # relabel ON to LHS/RHS
            if node.on_exprs:
                def relabel(e: exp.Expression) -> str:
                    e2 = e.copy()
                    for c in e2.find_all(exp.Column):
                        tab = (c.table or "").lower()
                        if tab in node.left.tables():
                            c.set("table", exp.to_identifier("LHS"))
                        elif tab in node.right.tables():
                            c.set("table", exp.to_identifier("RHS"))
                        else:
                            c.set("table", None)
                    return _expr_sql(e2)
                on_sql = " AND ".join(relabel(e) for e in node.on_exprs)
            else:
                on_sql = "1=1"

            ctes.append((name, f"SELECT LHS.*, RHS.* FROM {left_name} AS LHS JOIN {right_name} AS RHS ON {on_sql}"))

            # post udf filters on the merged result
            current = name
            for (t, eid, e) in node.post_filters:
                fname = new_name(f"{current}_e")
                pred = _expr_sql(_rewrite_columns(e, "CUR"))
                ctes.append((fname, f"SELECT * FROM {current} AS CUR WHERE {pred}"))
                current = fname
            return current

        raise ValueError("Unknown node type")

    return ctes, emit_base_ctes, gen

def build_pullup_sql(sql: str) -> str:
    q = parse_one(sql, read="duckdb")
    where_node = q.args.get("where")
    if not where_node:
        return sql

    all_conj = _split_conjuncts(where_node)
    inner_conj = [c for c in all_conj if not _has_udf(c)]
    outer_conj = [c for c in all_conj if _has_udf(c)]

    if not outer_conj:
        return sql

    q_inner = q.copy()
    if not inner_conj:
        q_inner.set("where", None)
    else:
        cur = None
        for c in inner_conj:
            cur = c if cur is None else exp.And(this=cur, expression=c)
        q_inner.set("where", exp.Where(this=cur))
    inner_sql = q_inner.sql(dialect="duckdb")

    outer_parts = [_expr_sql(_rewrite_columns(c, "SUB")) for c in outer_conj]
    outer_where = " AND ".join(outer_parts)

    return f"SELECT * FROM (\n  {inner_sql}\n) AS SUB WHERE {outer_where};"
