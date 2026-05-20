"""REINFORCE + learned baseline.

설계 원칙 (강의 06 범위 + variance reduction 표준 기법):
  - reward-to-go: G_t = r_t + γ G_{t+1}, 전체 return 사용 X
  - learned baseline V_φ(s): advantage A_t = G_t - V_φ(s_t)
  - advantage z-score 정규화: variance 더 줄이기
  - gradient clipping (max_norm=1.0)

Action masking을 지원해 RAGEnv에도 그대로 재사용 가능 (Phase 4).
mask는 bool tensor (shape: [n_actions])이며 True가 유효한 action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------- 네트워크 ----------


class PolicyNet(nn.Module):
    """2-layer MLP policy. state_dim → 256 → 128 → n_actions."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """logits 반환. mask가 주어지면 False 위치를 -inf로 채움."""
        logits = self.net(x)
        if mask is not None:
            # mask: True가 유효 → False 자리는 -inf
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class ValueNet(nn.Module):
    """동일 backbone + linear head로 scalar V(s) 추정."""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,) shape으로 반환
        return self.net(x).squeeze(-1)


# ---------- 한 step에 저장할 정보 ----------


@dataclass
class Transition:
    """한 step 동안 학습에 필요한 텐서 / 값."""

    state: torch.Tensor                  # (state_dim,)
    action: int
    log_prob: torch.Tensor               # scalar (grad 유지)
    reward: float
    mask: Optional[torch.Tensor] = None  # (n_actions,) bool


# ---------- 에이전트 ----------


class REINFORCEAgent:
    """REINFORCE + learned baseline.

    Args:
        state_dim, n_actions: 환경별 크기
        lr_policy, lr_value: 별도 optimizer (서로 다른 학습률 가능)
        gamma: 할인율
        hidden_dims: policy/value 공통 hidden 구조
        grad_clip: gradient norm clipping (학습 안정성용)
        normalize_advantage: trajectory 내 advantage z-score 정규화 on/off
        device: "cpu" 또는 "cuda"
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr_policy: float = 1e-3,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        hidden_dims: Sequence[int] = (256, 128),
        grad_clip: float = 1.0,
        normalize_advantage: bool = True,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.normalize_advantage = normalize_advantage
        self.n_actions = n_actions

        self.policy = PolicyNet(state_dim, n_actions, hidden_dims).to(self.device)
        self.value = ValueNet(state_dim, hidden_dims).to(self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr_value)

    # ----- action 선택 -----

    def select_action(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor]:
        """state로부터 action 샘플. (action_idx, log_prob) 반환.

        log_prob은 gradient를 유지한 채 반환됨 → trajectory에 모아두고
        update에서 직접 backward에 사용.
        """
        state_b = state.unsqueeze(0).to(self.device)
        mask_b = mask.unsqueeze(0).to(self.device) if mask is not None else None
        logits = self.policy(state_b, mask_b)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze(0)

    @torch.no_grad()
    def greedy_action(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> int:
        """평가용: argmax 선택. 학습에는 안 씀."""
        state_b = state.unsqueeze(0).to(self.device)
        mask_b = mask.unsqueeze(0).to(self.device) if mask is not None else None
        logits = self.policy(state_b, mask_b)
        return int(logits.argmax(dim=-1).item())

    # ----- update -----

    def update(self, transitions: List[Transition]) -> dict:
        """한 trajectory로 policy + value 업데이트.

        Returns:
            로깅용 dict: policy_loss, value_loss, mean_return, mean_advantage, entropy
        """
        T = len(transitions)
        if T == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        # reward-to-go 계산 (역순 누적)
        returns = torch.zeros(T, device=self.device)
        running = 0.0
        for t in reversed(range(T)):
            running = transitions[t].reward + self.gamma * running
            returns[t] = running

        states = torch.stack([tr.state.to(self.device) for tr in transitions])
        log_probs = torch.stack([tr.log_prob for tr in transitions]).to(self.device)

        # baseline 추정
        values = self.value(states)
        advantages = returns - values.detach()

        if self.normalize_advantage and T > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        # policy gradient
        policy_loss = -(log_probs * advantages).mean()

        # value MSE (baseline의 학습 타깃은 정규화 안 한 raw return)
        value_loss = F.mse_loss(values, returns)

        # entropy (로깅용)
        with torch.no_grad():
            # 다시 forward해서 entropy 추정 (mask 무시 — 로깅 용도)
            logits = self.policy(states)
            entropy = Categorical(logits=logits).entropy().mean().item()

        # backward + clip
        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_clip)
        self.value_opt.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "mean_return": float(returns.mean().item()),
            "mean_advantage": float(advantages.mean().item()),
            "entropy": entropy,
        }
