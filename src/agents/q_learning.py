# src/agents/q_learning.py
"""
Q-Learning tabular com espaço de estados discretizado.
Estado = (user_id, último subreddit recomendado)  →  par (u, s_prev)
"""

import numpy as np
from src.baselines.baselines import BaseAgent
from src.config import Q_ALPHA, Q_GAMMA, Q_EPSILON_START, Q_EPSILON_END, Q_EPSILON_DECAY


class QLearningAgent(BaseAgent):
    """
    Q-tabela de dimensão (n_users × n_subs, n_subs).
    Estado codificado como: state_id = user_id * n_subs + last_sub_id
    """
    def __init__(
        self,
        n_users: int,
        n_subs: int,
        alpha: float     = Q_ALPHA,
        gamma: float     = Q_GAMMA,
        epsilon: float   = Q_EPSILON_START,
        epsilon_min: float = Q_EPSILON_END,
        epsilon_decay: float = Q_EPSILON_DECAY,
        seed: int        = 42
    ):
        self.n_users       = n_users
        self.n_subs        = n_subs
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)

        n_states = n_users * n_subs
        self.Q = np.zeros((n_states, n_subs), dtype=np.float32)

    def _obs_to_state(self, obs: np.ndarray) -> int:
        """Converte observação em índice de estado escalar."""
        user_id  = int(obs[0])
        last_sub = int(obs[1])  # primeiro elemento do histórico
        return user_id * self.n_subs + last_sub

    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_subs))
        state = self._obs_to_state(obs)
        return int(np.argmax(self.Q[state]))

    def update(self, obs, action, reward, next_obs, done):
        s  = self._obs_to_state(obs)
        s2 = self._obs_to_state(next_obs)

        target = reward + (0.0 if done else self.gamma * np.max(self.Q[s2]))
        self.Q[s, action] += self.alpha * (target - self.Q[s, action])

        # Decaimento de ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def name(self) -> str:
        return "Q-Learning"
