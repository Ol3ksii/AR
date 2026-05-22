# src/agents/bandits.py
"""
Algoritmos Multi-Armed Bandit:
  - EpsilonGreedyAgent   : ε-greedy com decaimento opcional
  - UCBAgent             : Upper Confidence Bound (UCB1)
  - ThompsonSamplingAgent: Thompson Sampling com priors Beta
"""

import numpy as np
from src.baselines.baselines import BaseAgent


# ── ε-Greedy ──────────────────────────────────────────────────────────────────
class EpsilonGreedyAgent(BaseAgent):
    """
    Com probabilidade ε escolhe acção aleatória, caso contrário escolhe
    a acção com maior recompensa média estimada.
    ε pode decrescer ao longo do tempo (epsilon_decay < 1).
    """
    def __init__(
        self,
        n_subs: int,
        epsilon: float = 0.1,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 1.0,   # 1.0 = sem decaimento
        seed: int = 42
    ):
        self.n_subs        = n_subs
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)

        self.q_values = np.zeros(n_subs)   # recompensa média estimada
        self.counts   = np.zeros(n_subs)   # número de vezes que cada braço foi puxado

    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_subs))
        return int(np.argmax(self.q_values))

    def update(self, obs, action, reward, next_obs, done):
        self.counts[action]   += 1
        n = self.counts[action]
        # Média incremental
        self.q_values[action] += (reward - self.q_values[action]) / n
        # Decaimento de ε
        if self.epsilon_decay < 1.0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def name(self) -> str:
        return f"ε-greedy(ε={self.epsilon:.3f})"


# ── UCB1 ──────────────────────────────────────────────────────────────────────
class UCBAgent(BaseAgent):
    """
    Selecciona a acção que maximiza Q(a) + c * sqrt(ln(t) / N(a)).
    Acções não experimentadas têm prioridade máxima.
    """
    def __init__(self, n_subs: int, c: float = 1.0):
        self.n_subs   = n_subs
        self.c        = c
        self.q_values = np.zeros(n_subs)
        self.counts   = np.zeros(n_subs)
        self.t        = 0   # passo global

    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        self.t += 1
        # Acções não visitadas têm prioridade
        unvisited = np.where(self.counts == 0)[0]
        if len(unvisited) > 0:
            return int(unvisited[0])
        bonus  = self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(self.q_values + bonus))

    def update(self, obs, action, reward, next_obs, done):
        self.counts[action]   += 1
        n = self.counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n

    def name(self) -> str:
        return f"UCB(c={self.c})"


# ── Thompson Sampling ─────────────────────────────────────────────────────────
class ThompsonSamplingAgent(BaseAgent):
    """
    Prior Beta(α, β) para recompensas binárias / normalizadas [0,1].
    Actualiza α e β com base no feedback observado.
    Para recompensas contínuas, discretiza: reward > threshold → sucesso.
    """
    def __init__(self, n_subs: int, threshold: float = 0.5, seed: int = 42):
        self.n_subs    = n_subs
        self.threshold = threshold
        self.rng       = np.random.default_rng(seed)
        self.alpha     = np.ones(n_subs)   # sucessos + 1
        self.beta      = np.ones(n_subs)   # falhas + 1

    def select_action(self, obs: np.ndarray, user_id: int = None) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, obs, action, reward, next_obs, done):
        # Normalizar recompensa para [0,1] e binarizar
        success = float(reward > self.threshold)
        self.alpha[action] += success
        self.beta[action]  += (1.0 - success)

    def name(self) -> str:
        return "ThompsonSampling"
