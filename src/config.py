# src/config.py
# Configuração central do projeto — alterar aqui e propaga para todo o código

import os

# ── Caminhos ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW   = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC  = os.path.join(BASE_DIR, "data", "processed")
RESULTS    = os.path.join(BASE_DIR, "results")
PLOTS_DIR  = os.path.join(RESULTS, "plots")
METRICS_DIR= os.path.join(RESULTS, "metrics")

# Ficheiros originais (colocar em data/raw/)
FILE_INTERACTIONS = os.path.join(DATA_RAW, "reddit_user_data_count.csv")
FILE_SUBREDDITS   = os.path.join(DATA_RAW, "subreddit_info.csv")
FILE_REDDIT_DATA  = os.path.join(DATA_RAW, "reddit_data.csv")

# Ficheiros processados
FILE_INTER_PROC   = os.path.join(DATA_PROC, "interactions.parquet")
FILE_SUB_PROC     = os.path.join(DATA_PROC, "subreddits.parquet")
FILE_TRAIN        = os.path.join(DATA_PROC, "train.parquet")
FILE_TEST         = os.path.join(DATA_PROC, "test.parquet")

# ── Pré-processamento ─────────────────────────────────────────────────────────
MIN_USER_INTERACTIONS = 5      # utilizadores com menos interacções são removidos
MIN_SUB_INTERACTIONS  = 10     # subreddits com menos interacções são removidos
EXCLUDE_OVER18        = True   # excluir subreddits marcados como adultos
TOP_K_SUBREDDITS      = 1000   # reduzir espaço de acções aos top-K subreddits
TEST_RATIO            = 0.2    # fracção de utilizadores para teste
RANDOM_SEED           = 42

# ── Ambiente ──────────────────────────────────────────────────────────────────
EPISODE_LENGTH        = 20     # número de recomendações por episódio
HISTORY_LENGTH        = 5      # tamanho da janela de histórico recente
REPEAT_PENALTY        = 0.5    # penalização por repetir subreddit recente
DIVERSITY_BONUS       = 0.1    # bónus por recomendar subreddit diferente

# ── Bandits ───────────────────────────────────────────────────────────────────
EPSILON               = 0.1    # ε-greedy: probabilidade de exploração
UCB_C                 = 1.0    # UCB: constante de exploração

# ── Q-Learning ────────────────────────────────────────────────────────────────
Q_ALPHA               = 0.1    # taxa de aprendizagem
Q_GAMMA               = 0.95   # factor de desconto
Q_EPSILON_START       = 1.0
Q_EPSILON_END         = 0.05
Q_EPSILON_DECAY       = 0.995

# ── DQN ───────────────────────────────────────────────────────────────────────
DQN_LR                = 1e-3
DQN_GAMMA             = 0.95
DQN_BATCH_SIZE        = 64
DQN_BUFFER_SIZE       = 10_000
DQN_TARGET_UPDATE     = 100    # passos entre actualizações da rede-alvo
DQN_HIDDEN_DIM        = 128
DQN_EMBED_DIM         = 32     # dimensão dos embeddings de utilizador/subreddit

# ── REINFORCE ─────────────────────────────────────────────────────────────────
REINFORCE_LR          = 1e-3
REINFORCE_GAMMA       = 0.95
REINFORCE_K           = 5      # tamanho da lista top-K recomendada

# ── Avaliação ─────────────────────────────────────────────────────────────────
EVAL_K                = 10     # K para Hit Rate@K e NDCG@K
N_EVAL_EPISODES       = 200
N_TRAIN_EPISODES      = 2_000
