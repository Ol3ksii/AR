# src/evaluation/metrics.py
"""
Métricas de avaliação:
  - cumulative_reward    : recompensa acumulada média por episódio
  - hit_rate_at_k        : Hit Rate@K
  - ndcg_at_k            : NDCG@K
  - intra_list_diversity : diversidade intra-lista (fracção de subreddits únicos)
  - catalog_coverage     : fracção do catálogo recomendado
  - novelty              : novidade em [0,1] — 1 = muito novo, 0 = muito popular
  - exploration_rate     : fracção de acções únicas no episódio
"""

import numpy as np
from typing import List


def cumulative_reward(episode_rewards: List[List[float]]) -> float:
    return float(np.mean([sum(ep) for ep in episode_rewards]))


def hit_rate_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    top_k = set(recommended[:k])
    return float(len(top_k & set(relevant)) > 0)


def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    relevant_set = set(relevant)
    dcg  = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant_set
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def intra_list_diversity(actions: List[int]) -> float:
    if not actions:
        return 0.0
    return len(set(actions)) / len(actions)


def catalog_coverage(all_actions: List[List[int]], n_subs: int) -> float:
    all_unique = set(a for ep in all_actions for a in ep)
    return len(all_unique) / n_subs


def novelty(actions: List[int], item_popularity: dict, max_pop: float = None) -> float:
    """
    Novidade normalizada em [0, 1].
    Usa popularidade relativa: nov(a) = 1 - pop(a) / max_pop.
    Itens raros → perto de 1; itens muito populares → perto de 0.
    """
    if not item_popularity:
        return 0.0

    if max_pop is None:
        max_pop = max(item_popularity.values()) if item_popularity else 1.0

    scores = []
    for a in actions:
        pop = item_popularity.get(a, 0.0)
        scores.append(1.0 - pop / (max_pop + 1e-8))

    return float(np.mean(scores)) if scores else 0.0


def exploration_rate(actions: List[int]) -> float:
    return intra_list_diversity(actions)


def compute_all_metrics(
    episode_rewards:  List[List[float]],
    episode_actions:  List[List[int]],
    relevant_per_ep:  List[List[int]],
    n_subs:           int,
    item_popularity:  dict,
    k:                int = 10
) -> dict:
    max_pop = max(item_popularity.values()) if item_popularity else 1.0
    hr_list, ndcg_list, div_list, nov_list = [], [], [], []

    for recs, rels in zip(episode_actions, relevant_per_ep):
        hr_list.append(hit_rate_at_k(recs, rels, k))
        ndcg_list.append(ndcg_at_k(recs, rels, k))
        div_list.append(intra_list_diversity(recs))
        nov_list.append(novelty(recs, item_popularity, max_pop))

    return {
        "cumulative_reward":    cumulative_reward(episode_rewards),
        f"hit_rate@{k}":        float(np.mean(hr_list)),
        f"ndcg@{k}":            float(np.mean(ndcg_list)),
        "intra_list_diversity": float(np.mean(div_list)),
        "catalog_coverage":     catalog_coverage(episode_actions, n_subs),
        "novelty":              float(np.mean(nov_list)),
    }
