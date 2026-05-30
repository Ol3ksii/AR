# src/evaluation/evaluate.py
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import *
from src.environment.reddit_env import RedditRecommendEnv
from src.evaluation.metrics import compute_all_metrics
from src.baselines.baselines import CollabFilterAgent
from src.training.train_all import (
    build_all_agents, run_episode,
    PICKLE_AGENTS, TORCH_AGENTS, _load_bandit
)

sns.set_theme(style="whitegrid", palette="tab10")


def load_trained_agents(agents: dict, models_dir: str, train_df, n_users, n_subs):
    obs_dim = 1 + HISTORY_LENGTH * 2

    for name in TORCH_AGENTS:
        path = os.path.join(models_dir, f"{name}.pt")
        if os.path.exists(path):
            agents[name].load(path)
            print(f"  ✓ {name} carregado (torch)")
        else:
            print(f"  ⚠ {name}.pt não encontrado")

    for name in PICKLE_AGENTS:
        path = os.path.join(models_dir, f"{name}.npy")
        if os.path.exists(path):
            agents[name] = _load_bandit(agents[name], path)
            print(f"  ✓ {name} carregado (npy)")
        else:
            print(f"  ⚠ {name}.npy não encontrado")

    print(f"  A reconstruir CollabFilter...")
    cf = CollabFilterAgent(n_users, n_subs)
    cf.fit(train_df)
    agents["CollabFilter"] = cf
    print(f"  ✓ CollabFilter pronto")

    return agents


def evaluate_agent(agent, env, n_episodes: int = N_EVAL_EPISODES):
    ep_rewards, ep_actions, ep_relevant = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        user_id = int(obs[0])
        relevant = env.get_user_relevant_subs(user_id, top_k=EVAL_K)
        total_r, actions = run_episode(env, agent, train=False)
        ep_rewards.append([total_r])
        ep_actions.append(actions)
        ep_relevant.append(relevant)
    return ep_rewards, ep_actions, ep_relevant


def plot_training_curves(curves: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    window = 50
    for name, rewards in curves.items():
        s = pd.Series(rewards).rolling(window, min_periods=1).mean()
        ax.plot(s, label=name, alpha=0.85)
    ax.set_xlabel("Episódio")
    ax.set_ylabel(f"Recompensa acumulada (média móvel {window})")
    ax.set_title("Curvas de Aprendizagem — Todos os Agentes")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Guardado: {save_path}")


def plot_metrics_bar(metrics_df: pd.DataFrame, save_path: str):
    metric_cols = [c for c in metrics_df.columns if c != "agent"]
    n_metrics   = len(metric_cols)
    fig, axes   = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    for ax, col in zip(axes, metric_cols):
        order = metrics_df.sort_values(col, ascending=False)
        sns.barplot(data=order, x=col, y="agent", ax=ax, orient="h")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    plt.suptitle("Comparação de Agentes — Métricas de Avaliação", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {save_path}")


def main():
    os.makedirs(PLOTS_DIR,   exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    models_dir = os.path.join(RESULTS, "models")

    # Carregar dimensões exactas usadas no treino
    dims_path = os.path.join(models_dir, "train_dims.json")
    if not os.path.exists(dims_path):
        raise FileNotFoundError(
            "train_dims.json não encontrado. Corre primeiro train_all.py."
        )
    with open(dims_path) as f:
        dims = json.load(f)
    n_users = dims["n_users"]
    n_subs  = dims["n_subs"]
    obs_dim = dims["obs_dim"]
    print(f"Dimensões do treino: {n_users} utilizadores, {n_subs} subreddits")

    print("A carregar dados...")
    test_df  = pd.read_parquet(FILE_TEST)
    train_df = pd.read_parquet(FILE_TRAIN)

    pop_counts = train_df.groupby("sub_id")["reward"].sum().to_dict()
    env        = RedditRecommendEnv(test_df, n_users, n_subs, mode="test")

    agents = build_all_agents(n_users, n_subs, obs_dim, train_df)
    agents["CollabFilter"] = None  # será criado em load_trained_agents

    print("\nA carregar modelos treinados...")
    agents = load_trained_agents(agents, models_dir, train_df, n_users, n_subs)

    print()
    results = []
    for agent_name, agent in agents.items():
        print(f"A avaliar: {agent_name}")
        ep_rewards, ep_actions, ep_relevant = evaluate_agent(agent, env)
        metrics = compute_all_metrics(
            ep_rewards, ep_actions, ep_relevant,
            n_subs=n_subs, item_popularity=pop_counts, k=EVAL_K
        )
        metrics["agent"] = agent_name
        results.append(metrics)
        print(f"  HR@{EVAL_K}={metrics[f'hit_rate@{EVAL_K}']:.3f}  "
              f"NDCG@{EVAL_K}={metrics[f'ndcg@{EVAL_K}']:.3f}  "
              f"Reward={metrics['cumulative_reward']:.3f}")

    metrics_df = pd.DataFrame(results)
    csv_path   = os.path.join(METRICS_DIR, "evaluation_results.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n✓ Métricas guardadas em {csv_path}")
    print(metrics_df.to_string(index=False))

    plot_metrics_bar(metrics_df, os.path.join(PLOTS_DIR, "metrics_comparison.png"))

    curves_path = os.path.join(METRICS_DIR, "training_curves.json")
    if os.path.exists(curves_path):
        with open(curves_path) as f:
            curves = json.load(f)
        plot_training_curves(curves, os.path.join(PLOTS_DIR, "training_curves.png"))


if __name__ == "__main__":
    main()
