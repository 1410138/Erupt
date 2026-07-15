import random
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from env import Component
from model import ForestQNet

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

@dataclass
class UdfJoinState:
    components: List[Component]
    join_key_set: Set[frozenset]
    parsed: Dict[str, Any]

    @classmethod
    def from_env(cls, env) -> "UdfJoinState":
        comps: List[Component] = []
        for c in env.components:
            comps.append(Component(
                plan=c.plan,
                node_ids=set(c.node_ids),
                table_set=set(c.table_set),
                udf_only=c.udf_only
            ))
        return cls(components=comps, join_key_set=set(env._join_key_set), parsed=env.parsed)

    def _has_join_edge(self, A_tabs: Set[str], B_tabs: Set[str]) -> bool:
        for ta in A_tabs:
            for tb in B_tabs:
                if frozenset([ta, tb]) in self.join_key_set:
                    return True
        return False

    def all_actions(self) -> List[Tuple[int, int, str]]:
        acts: List[Tuple[int, int, str]] = []
        n = len(self.components)
        for i in range(n):
            for j in range(i + 1, n):
                A = self.components[i]
                B = self.components[j]
                if A.is_udf_only() and not B.is_udf_only():
                    t, eid, _ = A.udf_only
                    if t in B.table_set:
                        acts.append((i, j, "FILTER"))
                    continue
                if B.is_udf_only() and not A.is_udf_only():
                    t, eid, _ = B.udf_only
                    if t in A.table_set:
                        acts.append((i, j, "FILTER"))
                    continue
                if not A.is_udf_only() and not B.is_udf_only():
                    if self._has_join_edge(A.table_set, B.table_set):
                        acts.append((i, j, "JOIN"))
        return acts

class SumTree:
    def __init__(self, capacity: int):
        assert capacity > 0
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s: float):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = idx
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = leaf - (self.capacity - 1)
        return leaf, self.tree[leaf], self.data[data_idx]

    def size(self) -> int:
        return self.n_entries

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 200000, eps: float = 1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.max_priority = 1.0
        self.min_priority = 1.0

    def size(self) -> int:
        return self.tree.size()

    def beta_by_step(self, step: int) -> float:
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * (step / float(self.beta_frames)))

    def add(self, transition, priority: float):
        p = (priority + self.eps) ** self.alpha
        self.tree.add(p, transition)
        self.max_priority = max(self.max_priority, priority)
        self.min_priority = min(self.min_priority, priority)

    def sample(self, batch_size: int, beta: float):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        total = self.tree.total()
        probs = [p / total for p in priorities]
        N = self.size()
        p_min = (self.min_priority + self.eps) ** self.alpha / (total + self.eps)
        max_weight = (p_min * N) ** (-beta)
        weights = [((prob * N) ** (-beta)) / (max_weight + 1e-12) for prob in probs]
        return batch, idxs, weights

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            priority = float(err) + self.eps
            p = (priority) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, priority)
            self.min_priority = min(self.min_priority, priority)

# Hyperparameters
# UDFBench: eps_decay=4500, memory_size=10000, beta_frames = 10000
# TPC-H: eps_decay=45000, memory_size=30000, beta_frames = 100000
# IMDB: eps_decay=30000, memory_size=20000, beta_frames = 80000

class DQNAgent:
    def __init__(self, lr=1e-3, gamma=1.0, eps_start=0.9, eps_end=0.2, eps_decay=30000,
                 memory_size=20000, batch_size=128, device="cpu", d_model=128, hidden=128,
                 schema=None, selectivity=None, udf_provider=None):
        self.device = device
        self.policy = ForestQNet(d_model=d_model, hidden=hidden, schema=schema, selectivity=selectivity, udf_provider=udf_provider).to(device)
        self.target = ForestQNet(d_model=d_model, hidden=hidden, schema=schema, selectivity=selectivity, udf_provider=udf_provider).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.gamma = gamma
        # Apply weight decay.
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.SmoothL1Loss()
        self.tau = 0.005

        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

        # Use prioritized experience replay (PER).
        self.memory = PrioritizedReplayBuffer(capacity=memory_size, alpha=0.6, beta_start=0.4, beta_frames=80000)

        self.batch_size = batch_size

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

        self.learn_start = 500           # Minimum transitions required before learning.
        self.optimize_every = 4          # Optimize every K learning steps (1 = every step).
        self.optimize_steps = 0          # Global optimization step count.

    def epsilon(self):
        frac = min(1.0, self.steps_done / float(self.eps_decay))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def choose_action(self, env, greedy=False):
        with torch.no_grad():
            actions, q_values = self.policy.score_actions(env, device=self.device)
        if not actions:
            return None
        eps = 0.0 if greedy else self.epsilon()

        if not greedy:
            self.steps_done += 1

        if random.random() < eps:
            idx = random.randrange(len(actions))
        else:
            with torch.no_grad():
                idx = int(torch.argmax(q_values).item())
        return actions[idx]  # (i,j,kind)

    def remember(self, state: UdfJoinState, action: Tuple[int,int,str], reward: float,
                 next_state: UdfJoinState, done: float):
        transition = Transition(state, action, reward, next_state, done)
        p_init = self.memory.max_priority if self.memory.size() > 0 else 1.0
        self.memory.add(transition, p_init)

    def optimize(self):
        # Ensure enough transitions have been collected before learning.
        if self.memory.size() < max(self.batch_size, self.learn_start):
            return None

        if (self.optimize_steps % self.optimize_every) != 0:
            self.optimize_steps += 1
            return None

        # Sample from PER.
        beta = self.memory.beta_by_step(self.optimize_steps)
        batch, idxs, is_weights = self.memory.sample(self.batch_size, beta=beta)
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)

        q_list = []
        target_list = []
        for trans in batch:
            s: UdfJoinState = trans.state
            a_i, a_j, a_kind = trans.action
            r = trans.reward
            s2: UdfJoinState = trans.next_state
            done = trans.done

            # Q(s,a) with policy net (dueling)
            acts_s = s.all_actions()
            if not acts_s:
                q_sa = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            else:
                actions_all, q_values_all = self.policy.score_actions_from_state(s, device=self.device)
                try:
                    idx_sa = actions_all.index((a_i, a_j, a_kind))
                    q_sa = q_values_all[idx_sa]
                except ValueError:
                    q_sa = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            acts_s2 = s2.all_actions()
            if not acts_s2:
                max_q_next = 0.0
            else:
                with torch.no_grad():
                    _, q_values_next_policy = self.policy.score_actions_from_state(s2, device=self.device)
                    best_action_idx = torch.argmax(q_values_next_policy)
                    _, q_values_next_target = self.target.score_actions_from_state(s2, device=self.device)
                    max_q_next = q_values_next_target[best_action_idx].item()

            target = r + (1.0 - done) * self.gamma * max_q_next

            q_list.append(q_sa.unsqueeze(0))
            target_list.append(torch.tensor([target], dtype=torch.float32, device=self.device))

        q = torch.cat(q_list, dim=0)                     # [B]
        target = torch.cat(target_list, dim=0)           # [B]

        per_sample_loss = F.smooth_l1_loss(q, target, reduction='none')
        loss = (per_sample_loss * is_weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Apply gradient clipping.
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.update_target()

        # Update PER priorities.
        with torch.no_grad():
            td_errors = (q - target).abs().detach().cpu().tolist()
        self.memory.update_priorities(idxs, td_errors)

        self.optimize_steps += 1

        return loss.item()

    def update_target(self):
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target.load_state_dict(target_net_state_dict)
