# src/training/train.py
import os
import json
import argparse
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.environment.reddit_env import RedditRecommendEnv
from src.agents.dqn import DQNAgent

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_episode(env, agent, train: bool = True):
    obs, _ = env.reset()
    user_id = int(obs[0])
    total_reward = 0.0
    actions = [] 
    
    while True:
        action = agent.select_action(obs, user_id=user_id, invalid_actions=actions)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if train:
            agent.update(obs, action, reward, next_obs, done)
            
        total_reward += reward
        actions.append(action)
        obs = next_obs
        
        if done:
            break
            
    return total_reward

def train_agent(agent, env, n_episodes: int, desc: str = ""):
    rewards = []
    for _ in tqdm(range(n_episodes), desc=desc, leave=False):
        r = run_episode(env, agent, train=True)
        rewards.append(r)
    return rewards

def main():
    parser = argparse.ArgumentParser(description="Train a specific RL agent.")
    parser.add_argument("--agent", type=str, required=True, choices=["DQN"], help="The agent to train.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    metrics_dir = os.path.join(config["paths"]["results"], "metrics")
    models_dir = os.path.join(config["paths"]["results"], "models")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"A carregar dados processados para o agente {args.agent}...")
    train_df = pd.read_parquet(os.path.join(config["paths"]["data_proc"], "train.parquet"))
    
    n_users = int(train_df["user_id"].max()) + 1
    n_subs  = int(train_df["sub_id"].max()) + 1
    obs_dim = 1 + config["environment"]["history_length"] * 2

    with open(os.path.join(models_dir, "train_dims.json"), "w") as f:
        json.dump({"n_users": n_users, "n_subs": n_subs, "obs_dim": obs_dim}, f)

    env = RedditRecommendEnv(
        interactions=train_df, 
        n_users=n_users, 
        n_subs=n_subs, 
        episode_length=config["environment"]["episode_length"]
    )

    if args.agent == "DQN":
        agent_config = config["agents"]["dqn"]
        agent = DQNAgent(
            n_users=n_users,
            n_subs=n_subs,
            obs_dim=obs_dim,
            lr=agent_config["lr"],
            gamma=agent_config["gamma"],
            epsilon=1.0,           
            epsilon_min=0.05,      
            epsilon_decay=0.995    
        )
        
        print(f"\nA treinar: {args.agent}")
        rewards = train_agent(agent, env, agent_config["n_train_episodes"], desc=args.agent)
        print(f"  Recompensa média (últimos 100 ep.): {np.mean(rewards[-100:]):.4f}")

        agent.save(os.path.join(models_dir, f"{args.agent}.pt"))
        print(f"  Guardado: {args.agent}.pt")
        
        with open(os.path.join(metrics_dir, f"{args.agent}_curve.json"), "w") as f:
            json.dump({args.agent: rewards}, f)

if __name__ == "__main__":
    main()