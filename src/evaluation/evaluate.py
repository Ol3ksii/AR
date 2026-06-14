# src/evaluation/evaluate.py
import os
import json
import argparse
import yaml
import numpy as np
import pandas as pd

from src.environment.reddit_env import RedditRecommendEnv
from src.evaluation.metrics import compute_all_metrics
from src.agents.dqn import DQNAgent

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_eval_episode(env, agent):
    obs, _ = env.reset()
    user_id = int(obs[0])
    total_reward = 0.0
    actions = []
    
    while True:
        action = agent.select_action(obs, user_id=user_id, invalid_actions=actions)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        actions.append(action)
        obs = next_obs
        
        if done:
            break
            
    return total_reward, actions

def evaluate_agent(agent, env, n_episodes: int, eval_k: int):
    ep_rewards, ep_actions, ep_relevant = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        user_id = int(obs[0])
        relevant = env.get_user_relevant_subs(user_id, top_k=eval_k)
        
        total_r, actions = run_eval_episode(env, agent)
        
        ep_rewards.append([total_r])
        ep_actions.append(actions)
        ep_relevant.append(relevant)
    return ep_rewards, ep_actions, ep_relevant

def main():
    parser = argparse.ArgumentParser(description="Evaluate a specific RL agent.")
    parser.add_argument("--agent", type=str, required=True, choices=["DQN"], help="The agent to evaluate.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics_dir = os.path.join(config["paths"]["results"], "metrics")
    models_dir = os.path.join(config["paths"]["results"], "models")
    os.makedirs(metrics_dir, exist_ok=True)

    dims_path = os.path.join(models_dir, "train_dims.json")
    if not os.path.exists(dims_path):
        raise FileNotFoundError("train_dims.json não encontrado. Corre primeiro train.py.")
    
    with open(dims_path) as f:
        dims = json.load(f)
    n_users = dims["n_users"]
    n_subs  = dims["n_subs"]
    obs_dim = dims["obs_dim"]

    print("A carregar dados de teste...")
    test_df  = pd.read_parquet(os.path.join(config["paths"]["data_proc"], "test.parquet"))
    train_df = pd.read_parquet(os.path.join(config["paths"]["data_proc"], "train.parquet"))
    
    pop_counts = train_df.groupby("sub_id")["reward"].sum().to_dict()
    
    env = RedditRecommendEnv(
        interactions=test_df, 
        n_users=n_users, 
        n_subs=n_subs, 
        episode_length=config["environment"]["episode_length"],
        mode="test"
    )

    if args.agent == "DQN":
        agent_config = config["agents"]["dqn"]
        agent = DQNAgent(
            n_users=n_users, n_subs=n_subs, obs_dim=obs_dim,
            lr=agent_config["lr"], gamma=agent_config["gamma"],
            epsilon=0.0,  
            epsilon_min=0.0, epsilon_decay=1.0
        )
        model_path = os.path.join(models_dir, f"{args.agent}.pt")
        agent.load(model_path)
        print(f"✓ {args.agent} carregado")

    print(f"\nA avaliar: {args.agent}")
    ep_rewards, ep_actions, ep_relevant = evaluate_agent(agent, env, n_episodes=200, eval_k=10)
    
    metrics = compute_all_metrics(
        episode_rewards=ep_rewards,
        episode_actions=ep_actions,
        relevant_per_ep=ep_relevant,
        n_subs=n_subs,
        item_popularity=pop_counts,
        k=10
    )
    metrics["agent"] = args.agent

    print(f"  HR@10={metrics['hit_rate@10']:.3f}  "
          f"NDCG@10={metrics['ndcg@10']:.3f}  "
          f"Reward={metrics['cumulative_reward']:.3f}")

    metrics_df = pd.DataFrame([metrics])
    csv_path = os.path.join(metrics_dir, f"{args.agent}_evaluation.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n✓ Métricas guardadas em {csv_path}")

if __name__ == "__main__":
    main()