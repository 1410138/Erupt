import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from model import TreeEncoder, PairwiseQHead, KIND2ID
from dqn import UdfJoinState, Component


class PPOPolicy(nn.Module):
    def __init__(self, d_model: int = 128, hidden: int = 128, max_cols: int = 32,
                 schema=None, selectivity=None, udf_provider=None):
        super().__init__()
        self.encoder = TreeEncoder(
            d_model=d_model,
            max_cols=max_cols,
            schema=schema,
            selectivity=selectivity,
            udf_provider=udf_provider,
        )
        self.policy_head = PairwiseQHead(d_model=d_model, hidden=hidden, kind_vocab=len(KIND2ID))
        self.value_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.state_ln = nn.LayerNorm(d_model, elementwise_affine=False)

    def set_providers(self, schema, selectivity, udf_provider=None):
        self.encoder.set_providers(schema, selectivity, udf_provider)

    def _encode_components(self, components: List[Component], device: str):
        embs = []
        for comp in components:
            h = self.encoder.encode_component(comp, device=device)
            embs.append(h)
        return torch.stack(embs, dim=0).to(device)

    def _state_embed(self, embs: torch.Tensor) -> torch.Tensor:
        s = torch.mean(embs, dim=0)
        return self.state_ln(s)

    def evaluate_state(self, state: UdfJoinState, device: str = "cpu"):
        self.encoder.set_context_from_state(state)
        actions = state.all_actions()
        embs = self._encode_components(state.components, device=device)
        s_emb = self._state_embed(embs)
        value = self.value_head(s_emb).squeeze(-1)
        if not actions:
            return actions, torch.zeros(0, device=device), value
        idx_i = torch.tensor([i for (i, _, _) in actions], dtype=torch.long, device=device)
        idx_j = torch.tensor([j for (_, j, _) in actions], dtype=torch.long, device=device)
        kinds = torch.tensor([KIND2ID[k] for (_, _, k) in actions], dtype=torch.long, device=device)
        logits = self.policy_head(embs[idx_i], embs[idx_j], kinds)
        return actions, logits, value


class RolloutBuffer:
    def __init__(self):
        self.states: List[UdfJoinState] = []
        self.actions: List[Tuple[int, int, str]] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []

    def add(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns_advantages(self, gamma: float, gae_lambda: float, device: str):
        returns = []
        advantages = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(self.rewards))):
            done = self.dones[t]
            delta = self.rewards[t] + gamma * (1.0 - done) * next_value - self.values[t].item()
            gae = delta + gamma * gae_lambda * (1.0 - done) * gae
            adv = gae
            ret = adv + self.values[t].item()
            returns.insert(0, ret)
            advantages.insert(0, adv)
            next_value = self.values[t].item()

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        old_log_probs = torch.stack(self.log_probs).detach().to(device)
        return returns, advantages, old_log_probs

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()


class PPOAgent:
    def __init__(self, lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.1, entropy_coef: float = 0.01, value_coef: float = 0.5,
                 train_epochs: int = 4, batch_size: int = 32, max_grad_norm: float = 1.0,
                 device: str = "cpu", d_model: int = 128, hidden: int = 128,
                 schema=None, selectivity=None, udf_provider=None):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.policy = PPOPolicy(
            d_model=d_model,
            hidden=hidden,
            schema=schema,
            selectivity=selectivity,
            udf_provider=udf_provider,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.buffer = RolloutBuffer()

    def choose_action(self, env, greedy: bool = False):
        state = UdfJoinState.from_env(env)
        actions, logits, value = self.policy.evaluate_state(state, device=self.device)
        if not actions:
            return None, None, None, state

        dist = Categorical(logits=logits)
        if greedy:
            idx = int(torch.argmax(dist.probs).item())
        else:
            idx = int(dist.sample().item())

        log_prob = dist.log_prob(torch.tensor(idx, device=self.device))
        action = actions[idx]
        return action, log_prob.detach(), value.detach(), state

    def store_transition(self, state, action, log_prob, value, reward, done):
        self.buffer.add(state, action, log_prob, value, reward, done)

    def _evaluate_action(self, state: UdfJoinState, action: Tuple[int, int, str]):
        actions_all, logits, value = self.policy.evaluate_state(state, device=self.device)
        if not actions_all:
            return None, None, None
        try:
            idx = actions_all.index(action)
        except ValueError:
            return None, None, None
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(torch.tensor(idx, device=self.device))
        entropy = dist.entropy().mean()
        return log_prob, value, entropy

    def update(self):
        if not self.buffer.rewards:
            return None

        returns, advantages, old_log_probs = self.buffer.compute_returns_advantages(
            self.gamma, self.gae_lambda, device=self.device
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_steps = len(self.buffer.states)
        indices = list(range(total_steps))

        info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        updates = 0

        for _ in range(self.train_epochs):
            random.shuffle(indices)
            for start in range(0, total_steps, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                if not batch_idx:
                    continue

                batch_policy_loss = []
                batch_value_loss = []
                batch_entropy = []

                for i in batch_idx:
                    state = self.buffer.states[i]
                    action = self.buffer.actions[i]
                    old_log_prob = old_log_probs[i]
                    ret = returns[i]
                    adv = advantages[i]

                    eval_log_prob, value_pred, entropy = self._evaluate_action(state, action)
                    if eval_log_prob is None:
                        continue

                    ratio = torch.exp(eval_log_prob - old_log_prob)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2)
                    value_loss = F.mse_loss(value_pred, ret)

                    batch_policy_loss.append(policy_loss)
                    batch_value_loss.append(value_loss)
                    batch_entropy.append(entropy)

                if not batch_policy_loss:
                    continue

                policy_loss = torch.stack(batch_policy_loss).mean()
                value_loss = torch.stack(batch_value_loss).mean()
                entropy = torch.stack(batch_entropy).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                info["policy_loss"] += float(policy_loss.item())
                info["value_loss"] += float(value_loss.item())
                info["entropy"] += float(entropy.item())
                info["total_loss"] += float(loss.item())
                updates += 1

        self.buffer.clear()
        if updates > 0:
            for k in info:
                info[k] /= updates
        return info
