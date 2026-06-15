# Sistema de Recomendação de Vídeo via Aprendizagem por Reforço

**UC Aprendizagem por Reforço — Grupo 4**  
Repositório: https://github.com/Ol3ksii/AR

## Descrição

Sistema de recomendação para feed de scroll infinito baseado em dados do Reddit.  
Modela o problema como um MDP e compara baselines, multi-armed bandits e agentes RL.

## Estrutura do Repositório

```
rl_recommender/
├── data/
│   ├── raw/                  # Dados originais (não modificar)
│   └── processed/            # Dados após pré-processamento
├── src/
│   ├── preprocessing/        # Limpeza e preparação dos dados
│   ├── environment/          # Ambiente Gymnasium simulado
│   ├── baselines/            # Métodos de referência (popularidade, CF)
│   ├── agents/               # Bandits, Q-Learning, DQN, REINFORCE
│   ├── training/             # Loops de treino
│   └── evaluation/           # Métricas e avaliação
├── notebooks/                # Exploração e análise interativa
├── results/
│   ├── plots/                # Gráficos gerados
│   └── metrics/              # CSVs com resultados
└── reports/                  # Relatório de planeamento e relatório final
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso rápido

```bash
# 1. Pré-processar dados
python src/preprocessing/preprocess.py

# 2. Treinar todos os agentes
python src/training/train_all.py

# 3. Avaliar e gerar gráficos
python src/evaluation/evaluate.py
```

## Algoritmos Implementados

| Família      | Método               | Ficheiro                          |
|--------------|----------------------|-----------------------------------|
| Baseline     | Popularidade global  | `src/baselines/popularity.py`     |
| Baseline     | Filtragem colaborativa | `src/baselines/collab_filter.py` |
| Bandit       | ε-greedy             | `src/agents/epsilon_greedy.py`    |
| Bandit       | UCB                  | `src/agents/ucb.py`               |
| Bandit       | Thompson Sampling    | `src/agents/thompson.py`          |
| RL           | Q-Learning tabular   | `src/agents/q_learning.py`        |
| RL           | DQN                  | `src/agents/dqn.py`               |
| RL           | REINFORCE (top-K)    | `src/agents/reinforce.py`         |

## Métricas

- Recompensa acumulada média por episódio
- Hit Rate@K, NDCG@K
- Diversidade intra-lista
- Cobertura do catálogo
- Novidade e taxa de exploração
