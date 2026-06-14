# src/environment/reddit_env.py
"""
Ambiente simulado de recomendação compatível com Gymnasium.

Estado  : [user_id, sub_ids do histórico recente, recompensas recentes]
Acção   : índice do subreddit a recomendar (0 … n_subs-1)
Recompensa: log(1 + count(u, s)) com penalização por repetição e bónus de diversidade
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (
    EPISODE_LENGTH, HISTORY_LENGTH,
    REPEAT_PENALTY, DIVERSITY_BONUS, RANDOM_SEED
)


class RedditRecommendEnv(gym.Env):
    """
    Ambiente de recomendação baseado em interacções Reddit.

    Parâmetros
    ----------
    interactions : pd.DataFrame
        Colunas obrigatórias: user_id, sub_id, reward
    n_users : int
    n_subs  : int
    episode_length : int  (default = EPISODE_LENGTH do config)
    mode : str  "train" ou "test" — em test, o utilizador é fixo se fornecido
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        interactions: pd.DataFrame,
        n_users: int,
        n_subs: int,
        episode_length: int = EPISODE_LENGTH,
        mode: str = "train",
        seed: int = RANDOM_SEED,
    ):
        super().__init__()
        self.interactions   = interactions
        self.n_users        = n_users
        self.n_subs         = n_subs
        self.episode_length = episode_length
        self.mode           = mode
        self.rng            = np.random.default_rng(seed)

        # Índice de recompensas: user_id → {sub_id: reward}
        self._build_reward_index()

        # Espaços Gymnasium
        # Observação: [user_id, hist_sub_0, …, hist_sub_{H-1}, hist_rew_0, …]
        obs_size = 1 + HISTORY_LENGTH * 2   # user + H subs + H recompensas
        self.observation_space = spaces.Box(
            low=0.0, high=float(max(n_users, n_subs)),
            shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_subs)

        self._reset_state()

    # ── Construção do índice ───────────────────────────────────────────────────
    def _build_reward_index(self):
        """Dicionário user_id → (array de sub_ids, array de rewards)."""
        self.user_subs   = {}
        self.user_reward = {}
        for uid, grp in self.interactions.groupby("user_id"):
            self.user_subs[uid]   = grp["sub_id"].values
            self.user_reward[uid] = dict(zip(grp["sub_id"], grp["reward"]))
        self.all_users = list(self.user_subs.keys())

    # ── Utilitários de estado ─────────────────────────────────────────────────
    def _reset_state(self):
        self.current_user  = None
        self.step_count    = 0
        self.history_subs  = deque([0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_rews  = deque([0.0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.episode_subs  = set()

    def _get_obs(self) -> np.ndarray:
        obs = np.array(
            [float(self.current_user)]
            + list(self.history_subs)
            + list(self.history_rews),
            dtype=np.float32
        )
        return obs

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        # Escolher utilizador aleatório (ou fixo se fornecido em options)
        if options and "user_id" in options:
            self.current_user = options["user_id"]
        else:
            self.current_user = int(self.rng.choice(self.all_users))

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: int):
            assert self.current_user is not None, "Chama reset() antes de step()"
            self.step_count += 1

            # Base Reward
            reward_dict = self.user_reward.get(self.current_user, {})
            base_reward = reward_dict.get(int(action), 0.0)

            shaped_reward = base_reward ** 2 

            if action in self.episode_subs:
                shaped_reward -= REPEAT_PENALTY
            else:
                shaped_reward += DIVERSITY_BONUS

            total_reward = float(shaped_reward)

            self.history_subs.append(float(action))
            self.history_rews.append(total_reward)
            self.episode_subs.add(action)

            terminated = self.step_count >= self.episode_length
            truncated  = False
            return self._get_obs(), total_reward, terminated, truncated, {}

    def render(self):
        pass  # sem renderização visual

    # ── Utilitário extra ──────────────────────────────────────────────────────
    def get_user_relevant_subs(self, user_id: int, top_k: int = 10):
        """Devolve os top-K subreddits reais do utilizador (para avaliação)."""
        rewards = self.user_reward.get(user_id, {})
        sorted_subs = sorted(rewards, key=rewards.get, reverse=True)
        return sorted_subs[:top_k]
