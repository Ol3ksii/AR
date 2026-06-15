# src/baselines/baselines.py
"""
Métodos de referência (não-RL):
  - RandomAgent          : recomendação aleatória uniforme
  - GlobalPopularityAgent: sempre recomenda os subreddits mais populares globalmente
  - PersonalizedPopAgent : recomenda com base no perfil do utilizador
  - CollabFilterAgent    : filtragem colaborativa simples (user-based cosine similarity)
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# ── Base ──────────────────────────────────────────────────────────────────────
class BaseAgent:
    """Interface mínima comum a todos os agentes."""
    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        raise NotImplementedError

    def update(self, obs, action, reward, next_obs, done):
        """Baselines não aprendem — método vazio por defeito."""
        pass

    def name(self) -> str:
        return self.__class__.__name__


# ── Random ────────────────────────────────────────────────────────────────────
class RandomAgent(BaseAgent):
    def __init__(self, n_subs: int, seed: int = 42):
        self.n_subs = n_subs
        self.rng    = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        return int(self.rng.integers(self.n_subs))


# ── Popularidade global ───────────────────────────────────────────────────────
class GlobalPopularityAgent(BaseAgent):
    """
    Recomenda iterando pelos top-K subreddits mais populares globalmente.
    Usa round-robin para não recomendar sempre o mesmo.
    """
    def __init__(self, interactions: pd.DataFrame, n_subs: int):
        pop = interactions.groupby("sub_id")["reward"].sum().sort_values(ascending=False)
        self.ranked = pop.index.tolist()
        self.n_subs = n_subs
        self._ptr   = 0  # ponteiro round-robin

    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        action = self.ranked[self._ptr % len(self.ranked)]
        self._ptr += 1
        return int(action)


# ── Popularidade personalizada ────────────────────────────────────────────────
class PersonalizedPopAgent(BaseAgent):
    """
    Para o utilizador actual, recomenda os seus subreddits mais visitados
    em round-robin. Para utilizadores desconhecidos, cai de volta para global.
    """
    def __init__(self, interactions: pd.DataFrame):
        self.user_ranked = {}
        for uid, grp in interactions.groupby("user_id"):
            ranked = grp.sort_values("reward", ascending=False)["sub_id"].tolist()
            self.user_ranked[uid] = ranked
        self._ptrs = {}

        # Fallback global
        pop = interactions.groupby("sub_id")["reward"].sum().sort_values(ascending=False)
        self.global_ranked = pop.index.tolist()

    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        ranked = self.user_ranked.get(user_id, self.global_ranked)
        ptr    = self._ptrs.get(user_id, 0)
        action = ranked[ptr % len(ranked)]
        self._ptrs[user_id] = ptr + 1
        return int(action)

    def reset_user(self, user_id: int):
        self._ptrs[user_id] = 0


# ── Filtragem Colaborativa (user-based) ──────────────────────────────────────
class CollabFilterAgent(BaseAgent):
    """
    Constrói uma matriz utilizador×subreddit e usa similaridade de cosseno
    para recomendar o que utilizadores semelhantes consumiram.
    Pré-computa vizinhos no fit() e regista sugestões em select_action().
    """
    def __init__(self, n_users: int, n_subs: int, k_neighbors: int = 20):
        self.n_users     = n_users
        self.n_subs      = n_subs
        self.k_neighbors = k_neighbors
        self.fitted      = False
        self._cache      = {}  # user_id → lista de sub_ids recomendados

    def fit(self, interactions: pd.DataFrame):
        """Treinar: construir matriz e similaridades."""
        rows   = interactions["user_id"].values
        cols   = interactions["sub_id"].values
        data   = interactions["reward"].values
        matrix = csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_subs))

        print("  A calcular similaridades de utilizadores...")
        sim = cosine_similarity(matrix, dense_output=False)
        self.sim_matrix = sim
        self.ui_matrix  = matrix
        self.fitted     = True
        print("  ✓ CollabFilter treinado")

    def _get_recommendations(self, user_id: int) -> list:
        """Top subreddits dos k vizinhos mais próximos (não vistos pelo user)."""
        if not self.fitted:
            raise RuntimeError("Chama fit() antes de select_action()")

        neighbors = np.argsort(
            np.array(self.sim_matrix[user_id].todense()).flatten()
        )[::-1][1: self.k_neighbors + 1]

        scores = np.zeros(self.n_subs)
        for nb in neighbors:
            scores += np.array(self.ui_matrix[nb].todense()).flatten()

        # Mascarar já vistos
        seen = self.ui_matrix[user_id].indices
        scores[seen] = -1.0

        return np.argsort(scores)[::-1].tolist()

    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        if user_id not in self._cache or len(self._cache[user_id]) == 0:
            self._cache[user_id] = self._get_recommendations(user_id)
        return int(self._cache[user_id].pop(0))
