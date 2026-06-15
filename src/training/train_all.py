# src/training/train_all.py
import os, json, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import *
from src.environment.reddit_env import RedditRecommendEnv
from src.baselines.baselines import (
    RandomAgent, GlobalPopularityAgent,
    PersonalizedPopAgent, CollabFilterAgent
)
from src.agents.bandits    import EpsilonGreedyAgent, UCBAgent, ThompsonSamplingAgent
from src.agents.q_learning import QLearningAgent
from src.agents.dqn        import DQNAgent
from src.agents.reinforce  import REINFORCEAgent


def run_episode(env, agent, train: bool = True):
    obs, _ = env.reset()
    user_id = int(obs[0])
    total_reward = 0.0
    actions = []
    while True:
        action = agent.select_action(obs, user_id=user_id)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if train:
            agent.update(obs, action, reward, next_obs, done)
        total_reward += reward
        actions.append(action)
        obs = next_obs
        if done:
            break
    return total_reward, actions


def train_agent(agent, env, n_episodes: int, desc: str = ""):
    rewards = []
    for _ in tqdm(range(n_episodes), desc=desc, leave=False):
        r, _ = run_episode(env, agent, train=True)
        rewards.append(r)
    return rewards


def build_all_agents(n_users, n_subs, obs_dim, train_df):
    return {
        "Random":           RandomAgent(n_subs),
        "GlobalPop":        GlobalPopularityAgent(train_df, n_subs),
        "PersonalPop":      PersonalizedPopAgent(train_df),
        "ε-greedy":         EpsilonGreedyAgent(n_subs, epsilon=EPSILON, epsilon_decay=0.999),
        "UCB":              UCBAgent(n_subs, c=UCB_C),
        "ThompsonSampling": ThompsonSamplingAgent(n_subs),
        "Q-Learning":       QLearningAgent(n_users, n_subs),
        "DQN":              DQNAgent(n_users, n_subs, obs_dim),
        "REINFORCE":        REINFORCEAgent(n_users, n_subs),
    }


PICKLE_AGENTS = ("ε-greedy", "UCB", "ThompsonSampling")
TORCH_AGENTS  = ("DQN", "REINFORCE")


def _save_bandit(agent, path: str):
    """Guarda apenas estado aprendido — hiperparâmetros vêm sempre do config."""
    state = {"class": type(agent).__name__}
    if hasattr(agent, "_q"):
        state["global_q"] = agent._global_q
        state["global_n"] = agent._global_n
    elif hasattr(agent, "alpha"):
        state["alpha"]     = agent.alpha
        state["beta"]      = agent.beta
        state["threshold"] = agent.threshold
    np.save(path, state, allow_pickle=True)


def _load_bandit(agent, path: str):
    """Restaura estado aprendido; hiperparâmetros mantêm-se os do config."""
    state = np.load(path, allow_pickle=True).item()
    if "global_q" in state:
        agent._global_q = state["global_q"]
        agent._global_n = state["global_n"]
    elif "alpha" in state:
        agent.alpha     = state["alpha"]
        agent.beta      = state["beta"]
        agent.threshold = state["threshold"]
    return agent


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    models_dir = os.path.join(RESULTS, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("A carregar dados processados...")
    train_df = pd.read_parquet(FILE_TRAIN)
    n_users  = int(train_df["user_id"].max()) + 1
    n_subs   = int(train_df["sub_id"].max()) + 1
    obs_dim  = 1 + HISTORY_LENGTH * 2
    print(f"  {n_users} utilizadores, {n_subs} subreddits")

    # Guardar dimensões usadas no treino — necessário para reconstruir redes em avaliação
    dims_path = os.path.join(models_dir, "train_dims.json")
    with open(dims_path, "w") as f:
        json.dump({"n_users": n_users, "n_subs": n_subs, "obs_dim": obs_dim}, f)

    env = RedditRecommendEnv(train_df, n_users, n_subs)

    cf = CollabFilterAgent(n_users, n_subs)
    cf.fit(train_df)

    agents = build_all_agents(n_users, n_subs, obs_dim, train_df)
    agents["CollabFilter"] = cf

    all_curves = {}

    for name, agent in agents.items():
        print(f"\nA treinar: {name}")
        if name in ("Random", "GlobalPop", "PersonalPop", "CollabFilter"):
            rewards = [run_episode(env, agent, train=False)[0] for _ in range(100)]
        else:
            rewards = train_agent(agent, env, N_TRAIN_EPISODES, desc=name)
        all_curves[name] = rewards
        print(f"  Recompensa média (últimos 100 ep.): {np.mean(rewards[-100:]):.4f}")

    for name in TORCH_AGENTS:
        agents[name].save(os.path.join(models_dir, f"{name}.pt"))
        print(f"  Guardado: {name}.pt")

    for name in PICKLE_AGENTS:
        path = os.path.join(models_dir, f"{name}.npy")
        _save_bandit(agents[name], path)
        print(f"  Guardado: {name}.npy")

    with open(os.path.join(METRICS_DIR, "training_curves.json"), "w") as f:
        json.dump(all_curves, f)

    print(f"\n✓ Modelos guardados em {models_dir}")
    print(f"✓ Curvas guardadas em {METRICS_DIR}")


if __name__ == "__main__":
    main()
