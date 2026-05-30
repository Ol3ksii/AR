# src/agents/bandits.py
"""
Algoritmos Multi-Armed Bandit:
  - EpsilonGreedyAgent   : ε-greedy com estimativas POR UTILIZADOR
  - UCBAgent             : UCB1 com estimativas POR UTILIZADOR
  - ThompsonSamplingAgent: Thompson Sampling com priors Beta (global — funciona bem assim)

FIX: ε-greedy e UCB agora mantêm tabelas separadas por user_id, evitando
     reward negativo em utilizadores de teste nunca vistos durante o treino.
"""

import numpy as np
from src.baselines.baselines import BaseAgent


# ── ε-Greedy (per-user) ───────────────────────────────────────────────────────
class EpsilonGreedyAgent(BaseAgent):
    """
    Mantém uma estimativa Q(u, a) por utilizador.
    Utilizadores novos herdam a média global como prior — evita exploração cega.
    """
    def __init__(
        self,
        n_subs: int,
        epsilon: float       = 0.1,
        epsilon_min: float   = 0.01,
        epsilon_decay: float = 1.0,
        seed: int            = 42
    ):
        self.n_subs        = n_subs
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)

        # Tabelas por utilizador: user_id → array (n_subs,)
        self._q: dict[int, np.ndarray] = {}
        self._n: dict[int, np.ndarray] = {}

        # Prior global (média de todos os utilizadores vistos)
        self._global_q = np.zeros(n_subs)
        self._global_n = np.zeros(n_subs)

    def _get_q(self, user_id: int) -> np.ndarray:
        if user_id not in self._q:
            # Utilizador novo herda prior global
            self._q[user_id] = self._global_q.copy()
            self._n[user_id] = np.zeros(self.n_subs)
        return self._q[user_id]

    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        uid = int(obs[0]) if user_id is None else user_id
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_subs))
        return int(np.argmax(self._get_q(uid)))

    def update(self, obs, action, reward, next_obs, done):
        uid = int(obs[0])
        q   = self._get_q(uid)
        self._n[uid][action] += 1
        n = self._n[uid][action]
        q[action] += (reward - q[action]) / n

        # Actualizar prior global (média incremental)
        self._global_n[action] += 1
        self._global_q[action] += (reward - self._global_q[action]) / self._global_n[action]

        if self.epsilon_decay < 1.0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def name(self) -> str:
        return f"ε-greedy(ε={self.epsilon:.3f})"


# ── UCB1 (per-user) ───────────────────────────────────────────────────────────
class UCBAgent(BaseAgent):
    """
    UCB1 com estimativas por utilizador.
    Utilizadores novos usam prior global; t é contado por utilizador.
    """
    def __init__(self, n_subs: int, c: float = 0.1, init_k: int = 50):
        self.n_subs = n_subs
        self.c      = c
        self.init_k = init_k  # exploração inicial limitada ao top-K do prior global

        self._q: dict[int, np.ndarray] = {}
        self._n: dict[int, np.ndarray] = {}
        self._t: dict[int, int]        = {}

        self._global_q = np.zeros(n_subs)
        self._global_n = np.zeros(n_subs)

    def _init_user(self, uid: int):
        if uid not in self._q:
            self._q[uid] = self._global_q.copy()
            self._n[uid] = np.zeros(self.n_subs)
            self._t[uid] = 0

    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        uid = int(obs[0]) if user_id is None else user_id
        self._init_user(uid)
        self._t[uid] += 1
        t = self._t[uid]

        # Exploração inicial limitada ao top-K do prior global
        # Evita visitar todos os 1000 subreddits antes de explorar
        top_k = np.argsort(self._global_q)[::-1][:self.init_k]
        unvisited_topk = [a for a in top_k if self._n[uid][a] == 0]
        if unvisited_topk:
            return int(unvisited_topk[0])

        # UCB apenas sobre subreddits já visitados
        visited = np.where(self._n[uid] > 0)[0]
        bonus = self.c * np.sqrt(np.log(t) / self._n[uid][visited])
        return int(visited[np.argmax(self._q[uid][visited] + bonus)])

    def update(self, obs, action, reward, next_obs, done):
        uid = int(obs[0])
        self._init_user(uid)
        self._n[uid][action] += 1
        n = self._n[uid][action]
        self._q[uid][action] += (reward - self._q[uid][action]) / n

        self._global_n[action] += 1
        self._global_q[action] += (reward - self._global_q[action]) / self._global_n[action]

    def name(self) -> str:
        return f"UCB(c={self.c})"


# ── Thompson Sampling ─────────────────────────────────────────────────────────
class ThompsonSamplingAgent(BaseAgent):
    """
    Prior Beta(α, β) — mantido global (funciona bem como prior Bayesiano).
    Threshold adaptativo: mediana das recompensas observadas.
    """
    def __init__(self, n_subs: int, seed: int = 42):
        self.n_subs    = n_subs
        self.rng       = np.random.default_rng(seed)
        self.alpha     = np.ones(n_subs)
        self.beta      = np.ones(n_subs)
        self._rewards  = []   # para calcular threshold adaptativo
        self.threshold = 0.5

    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, obs, action, reward, next_obs, done):
        self._rewards.append(reward)
        # Threshold = mediana das últimas 500 recompensas
        if len(self._rewards) % 100 == 0:
            self.threshold = float(np.median(self._rewards[-500:]))

        success = float(reward > self.threshold)
        self.alpha[action] += success
        self.beta[action]  += (1.0 - success)

    def name(self) -> str:
        return "ThompsonSampling"
