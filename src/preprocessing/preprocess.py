# src/preprocessing/preprocess.py
"""
Limpeza e preparação dos dados para o ambiente de simulação.
Produz ficheiros .parquet em data/processed/.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import *


def load_raw_data():
    """Carrega os três ficheiros CSV originais."""
    print("A carregar dados brutos...")
    interactions = pd.read_csv(FILE_INTERACTIONS)   # user, subreddit, count
    subreddits   = pd.read_csv(FILE_SUBREDDITS)     # subreddit, num_subscribers, over18, public_description
    # reddit_data pode conter posts/metadados adicionais
    reddit_data  = pd.read_csv(FILE_REDDIT_DATA) if os.path.exists(FILE_REDDIT_DATA) else None

    print(f"  Interacções: {len(interactions):,} linhas")
    print(f"  Subreddits:  {len(subreddits):,} linhas")
    if reddit_data is not None:
        print(f"  Reddit data: {len(reddit_data):,} linhas")
    return interactions, subreddits, reddit_data


def clean_interactions(interactions: pd.DataFrame, subreddits: pd.DataFrame) -> pd.DataFrame:
    """
    - Remove subreddits sem metadados
    - Exclui subreddits over18 (se configurado)
    - Remove utilizadores e subreddits com poucas interacções
    - Normaliza count com log(1 + count)
    """
    print("\nA limpar interacções...")

    # Juntar metadados
    df = interactions.merge(subreddits[["subreddit", "over18"]], on="subreddit", how="inner")
    print(f"  Após merge com metadados: {len(df):,} linhas")

    # Excluir over18
    if EXCLUDE_OVER18:
        df = df[~df["over18"].fillna(False)]
        print(f"  Após exclusão over18:     {len(df):,} linhas")

    # Filtrar utilizadores com poucas interacções
    user_counts = df.groupby("user")["count"].sum()
    valid_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    df = df[df["user"].isin(valid_users)]
    print(f"  Após filtro utilizadores: {len(df):,} linhas ({len(valid_users):,} utilizadores)")

    # Filtrar subreddits com poucas interacções
    sub_counts = df.groupby("subreddit")["count"].sum()
    valid_subs = sub_counts[sub_counts >= MIN_SUB_INTERACTIONS].index
    df = df[df["subreddit"].isin(valid_subs)]
    print(f"  Após filtro subreddits:   {len(df):,} linhas ({len(valid_subs):,} subreddits)")

    # Reduzir ao top-K subreddits
    top_subs = sub_counts.nlargest(TOP_K_SUBREDDITS).index
    df = df[df["subreddit"].isin(top_subs)]
    print(f"  Após top-{TOP_K_SUBREDDITS} subreddits:  {len(df):,} linhas")

    # Recompensa log-normalizada
    df["reward"] = np.log1p(df["count"])

    # Remover coluna auxiliar
    df = df.drop(columns=["over18"])

    return df.reset_index(drop=True)


def encode_ids(df: pd.DataFrame):
    """
    Cria mapeamentos user→id e subreddit→id (inteiros contíguos).
    Adiciona colunas user_id e sub_id ao dataframe.
    Devolve (df, user2id, sub2id).
    """
    users     = sorted(df["user"].unique())
    subs      = sorted(df["subreddit"].unique())
    user2id   = {u: i for i, u in enumerate(users)}
    sub2id    = {s: i for i, s in enumerate(subs)}

    df = df.copy()
    df["user_id"] = df["user"].map(user2id)
    df["sub_id"]  = df["subreddit"].map(sub2id)
    return df, user2id, sub2id


def split_train_test(df: pd.DataFrame):
    """
    Divisão por utilizador: TEST_RATIO dos utilizadores vão para teste.
    Dentro de cada utilizador de teste, metade das suas interacções fica oculta.
    """
    users = df["user_id"].unique()
    train_users, test_users = train_test_split(
        users, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )
    train_df = df[df["user_id"].isin(train_users)].copy()
    test_df  = df[df["user_id"].isin(test_users)].copy()
    print(f"\nDivisão treino/teste:")
    print(f"  Treino: {len(train_users):,} utilizadores, {len(train_df):,} interacções")
    print(f"  Teste:  {len(test_users):,} utilizadores, {len(test_df):,} interacções")
    return train_df, test_df


def build_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada utilizador, cria um vector de perfil:
    recompensa média por subreddit, normalizada.
    Devolve DataFrame (user_id, sub_id, norm_reward).
    """
    profile = (
        df.groupby(["user_id", "sub_id"])["reward"]
        .sum()
        .reset_index()
        .rename(columns={"reward": "total_reward"})
    )
    # Normalizar por utilizador
    user_max = profile.groupby("user_id")["total_reward"].transform("max")
    profile["norm_reward"] = profile["total_reward"] / (user_max + 1e-8)
    return profile


def main():
    os.makedirs(DATA_PROC, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    interactions, subreddits, _ = load_raw_data()
    df = clean_interactions(interactions, subreddits)
    df, user2id, sub2id = encode_ids(df)

    train_df, test_df = split_train_test(df)
    profiles = build_user_profiles(train_df)

    # Guardar
    df.to_parquet(FILE_INTER_PROC, index=False)
    train_df.to_parquet(FILE_TRAIN, index=False)
    test_df.to_parquet(FILE_TEST, index=False)
    profiles.to_parquet(os.path.join(DATA_PROC, "user_profiles.parquet"), index=False)

    # Guardar mapeamentos
    pd.Series(user2id).to_json(os.path.join(DATA_PROC, "user2id.json"))
    pd.Series(sub2id).to_json(os.path.join(DATA_PROC, "sub2id.json"))

    print(f"\n✓ Dados guardados em {DATA_PROC}")
    print(f"  Utilizadores: {len(user2id):,}")
    print(f"  Subreddits:   {len(sub2id):,}")


if __name__ == "__main__":
    main()
