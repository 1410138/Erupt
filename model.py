import hashlib
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from sqlglot import expressions as exp
import math
from udf_placement import TableNode, FilterNode, JoinNode, PlanNode
from schema import SchemaProvider
from selectivity import SelectivityProvider
from udf_features import UdfFeatureProvider

KIND2ID = {"JOIN": 0, "FILTER": 1}


class TableFeatureBuilder:
    """
   Build a column-level feature vector for each table:
    - 2*idx: whether the column appears in a JOIN condition
    - 2*idx+1: column-filter selectivity
    """
    def __init__(self, schema: SchemaProvider, selectivity: SelectivityProvider, parsed: dict, max_cols: int = 32):
        self.schema = schema
        self.selectivity = selectivity
        self.parsed = parsed
        self.max_cols = max_cols

        self.col_idx: Dict[str, Dict[str, int]] = {}
        for t in parsed.get("tables", []):
            cols = self.schema.columns_of(t)
            mapping = {c.lower(): i for i, c in enumerate(cols[: self.max_cols])}
            self.col_idx[t] = mapping

    def _set_join_flag(self, feats: Dict[str, List[float]], table: str, col: str):
        t = table.lower()
        c = col.lower()
        idx = self.col_idx[t].get(c)
        feats[t][2 * idx] = 1.0

    def _set_filter_sel(self, feats: Dict[str, List[float]], table: str, col: str, expr: exp.Expression):
        t = table.lower()
        c = col.lower()
        idx = self.col_idx[t].get(c)
        p = self.selectivity.estimate_selectivity(t, c, expr)
        sel = float(-math.log(p))
        feats[t][2 * idx + 1] = max(feats[t][2 * idx + 1], sel)

    def build(self) -> Dict[str, torch.Tensor]:
        tables: List[str] = self.parsed.get("tables", [])
        join_edges: Dict[frozenset, List[exp.Expression]] = self.parsed.get("join_edges", {})
        local_preds: Dict[str, List[exp.Expression]] = self.parsed.get("local_preds", {})

        feats: Dict[str, List[float]] = {t: [0.0] * (2 * self.max_cols) for t in tables}

        # JOIN-column indicators.
        for exprs in join_edges.values():
            for e in exprs:
                for c in e.find_all(exp.Column):
                    tab = (c.table or "").lower()
                    name = (c.name or "").lower()
                    if tab and name and tab in feats:
                        self._set_join_flag(feats, tab, name)


        # Column selectivity.
        for t, preds in local_preds.items():
            for e in preds:
                for c in e.find_all(exp.Column):
                    name = (c.name or "").lower()
                    if name:
                        self._set_filter_sel(feats, t, name, e)

        return {t: torch.tensor(v, dtype=torch.float32).view(1, -1) for t, v in feats.items()}


class NodeTreeLSTM(nn.Module):
    """
    Binary TreeLSTM: (lh,lc) + (rh,rc) + inputX -> (h,c)
    """
    def __init__(self, d: int):
        super().__init__()
        self.fc_l = nn.Linear(d, 5 * d)
        self.fc_r = nn.Linear(d, 5 * d)
        self.fc_x = nn.Linear(d, 5 * d)
        self.ln_h = nn.LayerNorm(d, elementwise_affine=False)

    def forward(self, lh: torch.Tensor, lc: torch.Tensor,
                rh: torch.Tensor, rc: torch.Tensor,
                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = self.fc_l(lh) + self.fc_r(rh) + self.fc_x(x)
        a, i, f1, f2, o = gates.chunk(5, dim=-1)
        c = torch.tanh(a) * torch.sigmoid(i) + torch.sigmoid(f1) * lc + torch.sigmoid(f2) * rc
        h = torch.sigmoid(o) * torch.tanh(c)
        return self.ln_h(h), c

class TreeEncoder(nn.Module):
    """
    - Table leaf: gate column embeddings [(2*max_cols)*d] with column features [(2*max_cols)], then pool to [d]
    - JoinNode: binary TreeLSTM
    - FilterNode: unary node (right child is empty); inject UDF features through inputX = f([hl, udf_feat])
    """
    def __init__(self, d_model: int = 128, max_cols: int = 32,
                 schema: Optional[SchemaProvider] = None,
                 selectivity: Optional[SelectivityProvider] = None,
                 udf_provider: Optional[UdfFeatureProvider] = None):
        super().__init__()
        self.d = d_model
        self.max_cols = max_cols

        self.schema = schema or SchemaProvider({})
        self.selectivity = selectivity or SelectivityProvider()
        self.udf_provider = udf_provider or UdfFeatureProvider(d_model)
        
        schema_tables = self.schema.table_names()
        self.table2id: Dict[str, int] = {t: i for i, t in enumerate(schema_tables)}
        
        self.vocab = len(self.table2id)

        # One column embedding per table, with length (2*max_cols)*d.
        self.column_bank = nn.Embedding(self.vocab, (2 * max_cols) * d_model)
        self.leaf_fc = nn.Linear(d_model, d_model)
        self.leaf_ln = nn.LayerNorm(d_model, elementwise_affine=False)

        self.input_fc = nn.Linear(2 * d_model, d_model)   # for JoinNode: [hl, hr]
        self.input_fc_unary = nn.Linear(2 * d_model, d_model)  # for FilterNode: [hl, udf_feat]
        self.treelstm = NodeTreeLSTM(d_model)

        self.ctx_feats: Dict[str, torch.Tensor] = {}
        self._ctx_parsed_id: Optional[int] = None

    def set_providers(self, schema: SchemaProvider, selectivity: SelectivityProvider, udf_provider: Optional[UdfFeatureProvider] = None):
        self.schema = schema
        self.selectivity = selectivity
        if udf_provider is not None:
            self.udf_provider = udf_provider
        self._ctx_parsed_id = None
        self.ctx_feats = {}

    def _ensure_table_ids(self, tables: List[str]):
        missing = [t for t in tables if t not in self.table2id]
        if missing:
            raise KeyError(f"Unknown tables {missing}; check schema.json")

    def set_context_from_parsed(self, parsed: dict):
        parsed_id = id(parsed)
        if self._ctx_parsed_id == parsed_id and self.ctx_feats:
            return
        self._ensure_table_ids(parsed.get("tables", []))
        builder = TableFeatureBuilder(self.schema, self.selectivity, parsed, max_cols=self.max_cols)
        self.ctx_feats = builder.build()
        self._ctx_parsed_id = parsed_id

    def set_context_from_env(self, env):
        self.set_context_from_parsed(env.parsed)

    def set_context_from_state(self, state):
        self.set_context_from_parsed(state.parsed)

    def _leaf_table(self, table: str, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        tid = self.table2id.get(table)
        bank = self.column_bank(torch.tensor([tid], dtype=torch.long, device=device))  # [1, L*d]
        bank = bank.view(2 * self.max_cols, self.d)  # [L, d]

        fea = self.ctx_feats.get(table)
        fea = fea.to(device)
        gated = fea.view(-1, 1) * bank  # [L, d]
        h = torch.relu(self.leaf_fc(gated))  # [L, d]
        h, _ = torch.max(h, dim=0)  # max pooling over L -> [d]
        h = self.leaf_ln(h)
        c = torch.zeros_like(h)
        return h, c

    def _inputX_binary(self, hl: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.input_fc(torch.cat([hl, hr], dim=-1)))

    def _inputX_unary_with_udf(self, hl: torch.Tensor, udf_feat: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.input_fc_unary(torch.cat([hl, udf_feat], dim=-1)))

    def _encode_plan(self, node: PlanNode, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        # Leaf: TableNode.
        if isinstance(node, TableNode):
            return self._leaf_table(node.t, device=device)

        # Unary node: FilterNode.
        if isinstance(node, FilterNode):
            # Recursively encode the child node (TableNode or JoinNode).
            hl, cl = self._encode_plan(node.child, device=device)
            # Encode UDF features.
            udf_feat = self.udf_provider.embed(node.t, node.expr, device=device)  # [d]
            # Set the right child to zero.
            hr = torch.zeros_like(hl)
            cr = torch.zeros_like(cl)
            # Use UDF features as part of inputX.
            x = self._inputX_unary_with_udf(hl, udf_feat)
            return self.treelstm(hl, cl, hr, cr, x)

        # Binary node: Join(left, right).
        if isinstance(node, JoinNode):
            hl, cl = self._encode_plan(node.left, device=device)
            hr, cr = self._encode_plan(node.right, device=device)
            x = self._inputX_binary(hl, hr)
            return self.treelstm(hl, cl, hr, cr, x)

        raise ValueError("Unknown plan node type")

    def encode_component(self, comp, device: str = "cpu") -> torch.Tensor:
        # Extract UDF features.
        if getattr(comp, "udf_only", None) is not None and comp.plan is None:
            t, eid, e = comp.udf_only
            return self.udf_provider.embed(t, e, device=device)
        # Extract features for other components.
        h, _ = self._encode_plan(comp.plan, device=device)
        return h


class PairwiseQHead(nn.Module):
    def __init__(self, d_model: int = 128, hidden: int = 128, kind_vocab: int = 2):
        super().__init__()
        self.kind_embed = nn.Embedding(kind_vocab, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor, kind_id: torch.Tensor) -> torch.Tensor:
        k = self.kind_embed(kind_id)
        x = torch.cat([emb_i, emb_j, k], dim=-1)
        return self.mlp(x).squeeze(-1)

class ForestQNet(nn.Module):
    def __init__(self, d_model: int = 128, hidden: int = 128, max_cols: int = 32,
                 schema: Optional[SchemaProvider] = None,
                 selectivity: Optional[SelectivityProvider] = None,
                 udf_provider: Optional[UdfFeatureProvider] = None):
        super().__init__()
        self.encoder = TreeEncoder(
            d_model=d_model,
            max_cols=max_cols,
            schema=schema,
            selectivity=selectivity,
            udf_provider=udf_provider,
        )
        self.q_head = PairwiseQHead(d_model=d_model, hidden=hidden, kind_vocab=len(KIND2ID))
        self.value_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.state_ln = nn.LayerNorm(d_model, elementwise_affine=False)

    def set_providers(self, schema: SchemaProvider, selectivity: SelectivityProvider, udf_provider: Optional[UdfFeatureProvider] = None):
        self.encoder.set_providers(schema, selectivity, udf_provider)

    def encode_components(self, components, device: str = "cpu"):
        embs = []
        for comp in components:
            h = self.encoder.encode_component(comp, device=device)
            embs.append(h)
        return torch.stack(embs, dim=0).to(device)

    def _state_embed(self, embs: torch.Tensor) -> torch.Tensor:
        # Mean-pool component embeddings into a state vector.
        s = torch.mean(embs, dim=0)
        return self.state_ln(s)

    def score_actions(self, env, device: str = "cpu"):
        self.encoder.set_context_from_env(env)
        actions = env.all_actions()
        if not actions:
            return actions, torch.zeros(0, device=device)
        embs = self.encode_components(env.components, device=device)
        s_emb = self._state_embed(embs)
        V = self.value_head(s_emb).squeeze(-1)  # Scalar value.
        idx_i = torch.tensor([i for (i, _, _) in actions], dtype=torch.long, device=device)
        idx_j = torch.tensor([j for (_, j, _) in actions], dtype=torch.long, device=device)
        kinds = torch.tensor([KIND2ID[k] for (_, _, k) in actions], dtype=torch.long, device=device)
        A = self.q_head(embs[idx_i], embs[idx_j], kinds)  # [N]
        Q = V + (A - A.mean())
        return actions, Q

    def encode_components_from_state(self, state, device: str = "cpu"):
        self.encoder.set_context_from_state(state)
        return self.encode_components(state.components, device=device)

    def score_actions_from_state(self, state, device: str = "cpu"):
        actions = state.all_actions()
        if not actions:
            return actions, torch.zeros(0, device=device)
        embs = self.encode_components_from_state(state, device=device)
        s_emb = self._state_embed(embs)
        V = self.value_head(s_emb).squeeze(-1)
        idx_i = torch.tensor([i for (i, _, _) in actions], dtype=torch.long, device=device)
        idx_j = torch.tensor([j for (_, j, _) in actions], dtype=torch.long, device=device)
        kinds = torch.tensor([KIND2ID[k] for (_, _, k) in actions], dtype=torch.long, device=device)
        A = self.q_head(embs[idx_i], embs[idx_j], kinds)
        Q = V + (A - A.mean())
        return actions, Q
