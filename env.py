from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from sqlglot import expressions as exp

from udf_placement import (
    parse_query,
    build_ctes_for_tables,
    build_sql_from_plan,
    make_node_id_for_table,
    make_node_id_for_udf,
    canonical_shape,
    TableNode,
    FilterNode,
    JoinNode,
)

@dataclass
class Component:
    plan: Optional[object]               # PlanNode or None (for udf-only)
    node_ids: Set[str]                   # e.g., {'T:part', 'U:partsupp:0'}
    table_set: Set[str]                  # tables present in this component
    udf_only: Optional[Tuple[str, int, exp.Expression]] = None  # (table, eid, expr)

    def is_udf_only(self) -> bool:
        return self.udf_only is not None

class UdfJoinEnv:
    def __init__(self, sql: str):
        self.sql = sql
        self.parsed = parse_query(sql)
        self.tables: List[str] = self.parsed["tables"]
        self.join_edges: Dict[frozenset, List[exp.Expression]] = self.parsed["join_edges"]
        self.local_preds: Dict[str, List[exp.Expression]] = self.parsed["local_preds"]
        self.udf_preds: Dict[str, List[exp.Expression]] = self.parsed["udf_preds"]
        self.base_cte_name = build_ctes_for_tables(self.tables, self.local_preds)

        # Initial components: one per table; one per udf predicate
        comps: List[Component] = []
        for t in self.tables:
            comps.append(Component(
                plan=TableNode(t),
                node_ids={make_node_id_for_table(t)},
                table_set={t},
                udf_only=None
            ))
        for t, preds in self.udf_preds.items():
            for eid, e in enumerate(preds):
                comps.append(Component(
                    plan=None,
                    node_ids={make_node_id_for_udf(t, eid)},
                    table_set=set(),
                    udf_only=(t, eid, e)
                ))
        self.components: List[Component] = comps

        self._join_key_set = set(self.join_edges.keys())

    def all_actions(self) -> List[Tuple[int, int, str]]:
        """
        Return list of legal actions as tuples: (i, j, kind) with kind in {'JOIN','FILTER'}.
        """
        acts: List[Tuple[int, int, str]] = []
        n = len(self.components)
        for i in range(n):
            for j in range(i + 1, n):
                A = self.components[i]
                B = self.components[j]

                # FILTER case
                if A.is_udf_only() and not B.is_udf_only():
                    t, eid, _ = A.udf_only
                    if t in B.table_set:
                        acts.append((i, j, "FILTER"))  # A filters B
                    continue
                if B.is_udf_only() and not A.is_udf_only():
                    t, eid, _ = B.udf_only
                    if t in A.table_set:
                        acts.append((i, j, "FILTER"))  # B filters A
                    continue

                # JOIN case
                if not A.is_udf_only() and not B.is_udf_only():
                    if self._has_join_edge(A.table_set, B.table_set):
                        acts.append((i, j, "JOIN"))
        return acts

    def _has_join_edge(self, A_tabs: Set[str], B_tabs: Set[str]) -> bool:
        for ta in A_tabs:
            for tb in B_tabs:
                if frozenset([ta, tb]) in self._join_key_set:
                    return True
        return False

    def take_action(self, i: int, j: int, kind: str) -> None:
        A = self.components[i]
        B = self.components[j]

        if kind == "FILTER":
            if A.is_udf_only() and not B.is_udf_only():
                udf_comp, main_comp = A, B
            elif B.is_udf_only() and not A.is_udf_only():
                udf_comp, main_comp = B, A
            else:
                raise ValueError("Invalid FILTER action")

            t, eid, e = udf_comp.udf_only
            child_plan = main_comp.plan
            new_plan = FilterNode(child=child_plan, t=t, eid=eid, expr=e)
            new_nodes = set(main_comp.node_ids)
            new_nodes.add(make_node_id_for_udf(t, eid))
            new_tables = set(main_comp.table_set)
            merged = Component(plan=new_plan, node_ids=new_nodes, table_set=new_tables)

        elif kind == "JOIN":
            if A.is_udf_only() or B.is_udf_only():
                raise ValueError("Invalid JOIN action")
            on_exprs = []
            for ta in A.table_set:
                for tb in B.table_set:
                    key = frozenset([ta, tb])
                    if key in self.join_edges:
                        on_exprs.extend(self.join_edges[key])
            if not on_exprs:
                return
            new_plan = JoinNode(left=A.plan, right=B.plan, on_exprs=on_exprs, post_filters=[])
            new_nodes = A.node_ids.union(B.node_ids)
            new_tables = A.table_set.union(B.table_set)
            merged = Component(plan=new_plan, node_ids=new_nodes, table_set=new_tables)
        else:
            raise ValueError("Unknown action kind")

        for idx in sorted([i, j], reverse=True):
            self.components.pop(idx)
        self.components.append(merged)

    def is_terminal(self) -> bool:
        # terminal if only one component
        if len(self.components) != 1:
            return False
        return not self.components[0].is_udf_only()

    def current_shape(self) -> str:
        if len(self.components) == 1 and self.components[0].plan is not None:
            return canonical_shape(self.components[0].plan)
        parts = []
        for c in self.components:
            if c.is_udf_only():
                t, eid, _ = c.udf_only
                parts.append(f"e({t}#{eid})")
            else:
                parts.append(str(c.plan))
        return " | ".join(parts)

    def build_final_sql(self) -> str:
        if not self.is_terminal():
            raise RuntimeError("Not terminal; cannot build final SQL")
        root = self.components[0].plan
        ctes, emit_base_ctes, gen = build_sql_from_plan(root, self.base_cte_name, self.join_edges)
        emit_base_ctes(self.local_preds)
        root_name = gen(root, self.local_preds)
        with_sql = "WITH\n" + ",\n".join(f"  {name} AS (\n    {body}\n  )" for name, body in ctes)
        return f"{with_sql}\nSELECT * FROM {root_name};"