# src/agents/dqn.py
"""
Deep Q-Network (DQN) com:
  - Replay buffer
  - Rede-alvo (target network) com actualização periódica
  - Embeddings de utilizador e subreddit aprendidos
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from src.baselines.baselines import BaseAgent
from src.config import (
    DQN_LR, DQN_GAMMA, DQN_BATCH_SIZE, DQN_BUFFER_SIZE,
    DQN_TARGET_UPDATE, DQN_HIDDEN_DIM, DQN_EMBED_DIM,
    Q_EPSILON_START, Q_EPSILON_END, Q_EPSILON_DECAY, HISTORY_LENGTH
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Rede neuronal ─────────────────────────────────────────────────────────────
class DQNNetwork(nn.Module):
    """
    Entrada: observação  [user_id, h_sub_0..H-1, h_rew_0..H-1]
    Saída  : Q-valores para cada subreddit (n_subs)
    """
    def __init__(self, n_users: int, n_subs: int, obs_dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, DQN_EMBED_DIM)
        self.sub_emb  = nn.Embedding(n_subs + 1, DQN_EMBED_DIM)   # +1 para padding 0

        # Dimensão de entrada: embed_user + H*embed_sub + H*1 (recompensas)
        in_dim = DQN_EMBED_DIM + HISTORY_LENGTH * DQN_EMBED_DIM + HISTORY_LENGTH

        self.net = nn.Sequential(
            nn.Linear(in_dim, DQN_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(DQN_HIDDEN_DIM, DQN_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(DQN_HIDDEN_DIM, n_subs)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape: (batch, 1 + H + H)
        user_ids  = obs[:, 0].long()
        sub_ids   = obs[:, 1: 1 + HISTORY_LENGTH].long().clamp(0)
        rews      = obs[:, 1 + HISTORY_LENGTH:]

        u_emb     = self.user_emb(user_ids)                    # (B, E)
        s_emb     = self.sub_emb(sub_ids).view(obs.size(0), -1)# (B, H*E)
        x         = torch.cat([u_emb, s_emb, rews], dim=1)
        return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buf)


# ── Agente DQN ────────────────────────────────────────────────────────────────
class DQNAgent(BaseAgent):
    def __init__(
        self,
        n_users: int,
        n_subs: int,
        obs_dim: int,
        lr: float          = DQN_LR,
        gamma: float       = DQN_GAMMA,
        epsilon: float     = Q_EPSILON_START,
        epsilon_min: float = Q_EPSILON_END,
        epsilon_decay: float = Q_EPSILON_DECAY,
        seed: int          = 42
    ):
        self.n_subs        = n_subs
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)
        self.steps         = 0

        self.policy_net = DQNNetwork(n_users, n_subs, obs_dim).to(DEVICE)
        self.target_net = DQNNetwork(n_users, n_subs, obs_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(DQN_BUFFER_SIZE)
        self.loss_fn   = nn.MSELoss()

    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_subs))
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            q = self.policy_net(t)
        return int(q.argmax().item())

    def update(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)
        self.steps += 1

        if len(self.buffer) < DQN_BATCH_SIZE:
            return

        obs_b, act_b, rew_b, nobs_b, done_b = self.buffer.sample(DQN_BATCH_SIZE)
        obs_t  = torch.FloatTensor(obs_b).to(DEVICE)
        nobs_t = torch.FloatTensor(nobs_b).to(DEVICE)
        act_t  = torch.LongTensor(act_b).to(DEVICE)
        rew_t  = torch.FloatTensor(rew_b).to(DEVICE)
        done_t = torch.FloatTensor(done_b).to(DEVICE)

        q_curr = self.policy_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(nobs_t).max(1)[0]
            target = rew_t + self.gamma * q_next * (1 - done_t)

        loss = self.loss_fn(q_curr, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Actualizar rede-alvo
        if self.steps % DQN_TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decaimento de ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def name(self) -> str:
        return "DQN"
