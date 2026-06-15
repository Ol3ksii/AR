# src/agents/reinforce.py
"""
REINFORCE para recomendação top-K.
Inspirado em Chen et al. (2019) "Top-K Off-Policy Correction for a REINFORCE Recommender System".

A política é uma rede que produz logits sobre o espaço de subreddits.
Em cada passo, amostra K itens sem reposição (ou usa argmax em avaliação).
A recompensa do episódio é usada para actualizar os pesos via policy gradient.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.baselines.baselines import BaseAgent
from src.config import (
    REINFORCE_LR, REINFORCE_GAMMA, REINFORCE_K,
    DQN_EMBED_DIM, DQN_HIDDEN_DIM, HISTORY_LENGTH
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Rede de política ──────────────────────────────────────────────────────────
class PolicyNetwork(nn.Module):
    """Produz logits (n_subs,) a partir da observação."""
    def __init__(self, n_users: int, n_subs: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, DQN_EMBED_DIM)
        self.sub_emb  = nn.Embedding(n_subs + 1, DQN_EMBED_DIM)

        in_dim = DQN_EMBED_DIM + HISTORY_LENGTH * DQN_EMBED_DIM + HISTORY_LENGTH
        self.net = nn.Sequential(
            nn.Linear(in_dim, DQN_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(DQN_HIDDEN_DIM, n_subs)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        user_ids = obs[:, 0].long()
        sub_ids  = obs[:, 1: 1 + HISTORY_LENGTH].long().clamp(0)
        rews     = obs[:, 1 + HISTORY_LENGTH:]

        u_emb = self.user_emb(user_ids)
        s_emb = self.sub_emb(sub_ids).view(obs.size(0), -1)
        x     = torch.cat([u_emb, s_emb, rews], dim=1)
        return self.net(x)   # logits (B, n_subs)


# ── Agente REINFORCE ──────────────────────────────────────────────────────────
class REINFORCEAgent(BaseAgent):
    """
    Acumula (log_prob, reward) durante o episódio e actualiza no fim.
    select_action devolve apenas 1 acção de cada vez (a primeira do top-K);
    em avaliação usa greedy (argmax).
    """
    def __init__(
        self,
        n_users: int,
        n_subs: int,
        k: int             = REINFORCE_K,
        lr: float          = REINFORCE_LR,
        gamma: float       = REINFORCE_GAMMA,
        seed: int          = 42
    ):
        self.n_subs  = n_subs
        self.k       = k
        self.gamma   = gamma
        self.rng     = np.random.default_rng(seed)
        self.training = True

        self.policy    = PolicyNetwork(n_users, n_subs).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Memória do episódio
        self.log_probs: list[torch.Tensor] = []
        self.rewards:   list[float]        = []

    # ── Selecção de acção ─────────────────────────────────────────────────────
    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        t      = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        logits = self.policy(t).squeeze(0)

        if self.training:
            probs  = torch.softmax(logits, dim=-1)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()
            self.log_probs.append(dist.log_prob(action))
            return int(action.item())
        else:
            return int(logits.argmax().item())

    # ── Acumulação de recompensa ───────────────────────────────────────────────
    def update(self, obs, action, reward, next_obs, done):
        """Acumula recompensa; actualiza pesos no fim do episódio."""
        self.rewards.append(float(reward))

        if done:
            self._episode_update()

    def _episode_update(self):
        """Policy gradient update com retornos descontados."""
        if not self.log_probs:
            return

        # Calcular retornos descontados
        G, returns = 0.0, []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(DEVICE)
        # Normalizar para reduzir variância
        if returns.std() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(self.log_probs)
        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # Limpar memória
        self.log_probs = []
        self.rewards   = []

    def eval_mode(self):
        self.training = False
        self.policy.eval()

    def train_mode(self):
        self.training = True
        self.policy.train()

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path, map_location=DEVICE))

    def name(self) -> str:
        return f"REINFORCE(K={self.k})"
