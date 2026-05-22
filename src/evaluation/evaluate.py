# src/evaluation/evaluate.py
"""
Avaliação completa de todos os agentes treinados.
Gera tabela de métricas e gráficos comparativos.
"""

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
from src.training.train_all import build_all_agents, run_episode


sns.set_theme(style="whitegrid", palette="tab10")


def evaluate_agent(agent, env, n_episodes: int = N_EVAL_EPISODES):
    """Corre n_episodes em modo avaliação (sem treino)."""
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
    """Gráfico de recompensa acumulada ao longo do treino (média móvel)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    window = 50

    for name, rewards in curves.items():
        if len(rewards) < window:
            ax.plot(rewards, label=name, alpha=0.8)
        else:
            smoothed = pd.Series(rewards).rolling(window).mean()
            ax.plot(smoothed, label=name, alpha=0.8)

    ax.set_xlabel("Episódio")
    ax.set_ylabel(f"Recompensa acumulada (média móvel {window})")
    ax.set_title("Curvas de Aprendizagem — Todos os Agentes")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Guardado: {save_path}")


def plot_metrics_bar(metrics_df: pd.DataFrame, save_path: str):
    """Gráfico de barras comparativo para cada métrica."""
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

    # Carregar dados
    print("A carregar dados de teste...")
    test_df  = pd.read_parquet(FILE_TEST)
    train_df = pd.read_parquet(FILE_TRAIN)
    
    # Calcular o tamanho exatamente como no treino:
    n_users  = int(train_df["user_id"].max()) + 1
    n_subs   = int(train_df["sub_id"].max()) + 1
    
    # IMPORTANTE: Filtrar o test_df para remover utilizadores/subs que não 
    # existiam no treino (evita o erro IndexError: index out of bounds)
    test_df = test_df[test_df["user_id"] < n_users]
    test_df = test_df[test_df["sub_id"] < n_subs]
    obs_dim  = 1 + HISTORY_LENGTH * 2

    # Popularidade global (para novidade)
    pop_counts = train_df.groupby("sub_id")["reward"].sum().to_dict()

    # Ambiente de teste
    env = RedditRecommendEnv(test_df, n_users, n_subs, mode="test")

    # Construir agentes
    agents = build_all_agents(n_users, n_subs, obs_dim, train_df)

    # Carregar modelos RL treinados
    models_dir = os.path.join(RESULTS, "models")
    for name in ("DQN", "REINFORCE"):
        path = os.path.join(models_dir, f"{name}.pt")
        if os.path.exists(path):
            agents[name].load(path)

    # Avaliar
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

    # Tabela final
    metrics_df = pd.DataFrame(results)
    csv_path   = os.path.join(METRICS_DIR, "evaluation_results.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n✓ Métricas guardadas em {csv_path}")
    print(metrics_df.to_string(index=False))

    # Gráficos
    plot_metrics_bar(
        metrics_df,
        os.path.join(PLOTS_DIR, "metrics_comparison.png")
    )

    # Curvas de treino (se existirem)
    curves_path = os.path.join(METRICS_DIR, "training_curves.json")
    if os.path.exists(curves_path):
        with open(curves_path) as f:
            curves = json.load(f)
        plot_training_curves(curves, os.path.join(PLOTS_DIR, "training_curves.png"))


if __name__ == "__main__":
    main()
