# Sistema de Recomendação de Vídeo via Aprendizagem por Reforço

UC Aprendizagem por Reforço — Grupo 4  
Repositório: https://github.com/Ol3ksii/AR

## Descrição

Este projeto implementa um sistema de recomendação sequencial inspirado em feeds de scroll infinito, utilizando dados de interações do Reddit. O problema é modelado como um Processo de Decisão de Markov, onde o agente recomenda subreddits a um utilizador e recebe uma recompensa associada ao envolvimento implícito observado.

O sistema compara diferentes famílias de métodos:

- Baselines clássicos de recomendação;
- Algoritmos de multi-armed bandits;
- Agentes de Aprendizagem por Reforço;
- API em FastAPI para servir recomendações;
- Interface frontend em React/Vite para demonstração interativa.

## Estrutura do Repositório

```text
AR/
├── configs/                  # Ficheiros de configuração
├── data/
│   └── processed/            # Dados processados
├── docs/
│   └── screenshots/          # Prints da API e da aplicação
├── frontend/                 # Interface React/Vite
├── notebooks/                # Exploração e análise interativa
├── reports/                  # Relatório final e anexos
├── results/
│   ├── metrics/              # Resultados em CSV
│   ├── models/               # Modelos treinados
│   └── plots/                # Gráficos gerados
├── src/
│   ├── agents/               # Bandits, Q-Learning, DQN e REINFORCE
│   ├── api/                  # API FastAPI
│   ├── baselines/            # Métodos de referência
│   ├── environment/          # Ambiente Gymnasium
│   ├── evaluation/           # Avaliação e métricas
│   ├── preprocessing/        # Pré-processamento dos dados
│   └── training/             # Scripts de treino
├── requirements.txt
├── apresentacao_ar.pptx
└── README.md
```

## Requisitos

Recomenda-se a utilização de **Python 3.12**.

> Nota: não se recomenda Python 3.13, uma vez que algumas dependências, nomeadamente o PyTorch, podem não ter distribuição compatível em determinados sistemas.

Também é necessário ter Node.js instalado para executar o frontend.

## Instalação

Na raiz do projeto:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Execução do Pipeline Experimental

### 1. Pré-processamento dos dados

```bash
PYTHONPATH=. python src/preprocessing/preprocess.py
```

### 2. Treino dos agentes

```bash
PYTHONPATH=. python src/training/train_all.py
```

### 3. Avaliação dos modelos

Na versão atual, o script de avaliação exige a indicação explícita do agente. Para avaliar o DQN:

```bash
PYTHONPATH=. python src/evaluation/evaluate.py --agent DQN
```

Para consultar as opções disponíveis:

```bash
PYTHONPATH=. python src/evaluation/evaluate.py --help
```

Se for usado um ficheiro de configuração:

```bash
PYTHONPATH=. python src/evaluation/evaluate.py --agent DQN --config configs/default.yaml
```

Os resultados são guardados em:

```text
results/metrics/
results/plots/
results/models/
```

## Execução da API

A API foi implementada com FastAPI.

Para iniciar o servidor, executar a partir da raiz do projeto:

```bash
PYTHONPATH=. python -m uvicorn src.api.server:app --reload
```

Depois abrir no navegador:

```text
http://127.0.0.1:8000/docs
```

A documentação Swagger/OpenAPI permite testar diretamente os endpoints.

## Endpoints da API

### GET `/api/recommend/{user_id}`

Gera uma recomendação para um utilizador.

Exemplo:

```text
GET http://127.0.0.1:8000/api/recommend/12345
```

### POST `/api/interact`

Regista uma interação do utilizador com uma recomendação.

Exemplo de corpo JSON:

```json
{
  "user_id": 12345,
  "sub_id": 10,
  "reward": 1.0
}
```

Nota: o `user_id` usado no `POST /api/interact` deve corresponder a um utilizador que já tenha recebido uma recomendação através de `GET /api/recommend/{user_id}`. Caso contrário, a API pode devolver erro interno se a sessão ainda não existir.

### GET `/api/reddit/{subreddit}`

Obtém conteúdos de um subreddit através do proxy da API.

Exemplo:

```text
GET http://127.0.0.1:8000/api/reddit/python
```

## Execução do Frontend

Abrir uma segunda janela do terminal e executar:

```bash
cd frontend
npm install
npm run dev
```

O frontend fica normalmente disponível em:

```text
http://localhost:5173
```

A API deve continuar ativa em:

```text
http://127.0.0.1:8000
```

## Testes Realizados

Foram validados os seguintes componentes:

- Arranque da API com Uvicorn;
- Acesso à documentação Swagger em `/docs`;
- Endpoint `GET /api/recommend/{user_id}`;
- Endpoint `POST /api/interact`;
- Endpoint `GET /api/reddit/{subreddit}`;
- Execução do frontend React/Vite;
- Comunicação entre frontend e backend.

## Algoritmos Implementados

| Família | Método | Localização |
|---|---|---|
| Baseline | Random | `src/baselines/` |
| Baseline | Popularidade global | `src/baselines/` |
| Baseline | Popularidade personalizada | `src/baselines/` |
| Baseline | Filtragem colaborativa | `src/baselines/` |
| Bandit | epsilon-greedy | `src/agents/` |
| Bandit | UCB | `src/agents/` |
| Bandit | Thompson Sampling | `src/agents/` |
| RL | Q-Learning tabular | `src/agents/` |
| RL | DQN | `src/agents/` |
| RL | REINFORCE | `src/agents/` |

## Métricas de Avaliação

As métricas utilizadas foram:

- Recompensa acumulada média por episódio;
- Hit Rate@10;
- NDCG@10;
- Diversidade intra-lista;
- Cobertura do catálogo;
- Novidade.

## Resultados Principais

Os principais resultados experimentais mostram que:

- O REINFORCE obteve a maior recompensa acumulada média;
- O Thompson Sampling obteve o melhor NDCG@10;
- Os baselines personalizados apresentaram bom Hit Rate@10;
- Existe um compromisso entre recompensa, diversidade e cobertura.

Os ficheiros completos de métricas e gráficos encontram-se em:

```text
results/metrics/
results/plots/
```


## Autores

Grupo 4 (Diogo Azevedo, Eduardo Vilaça, Martim Ferreira, Oleksii Tantsura) — UC Aprendizagem por Reforço, 2025/2026.

- Hit Rate@K, NDCG@K
- Diversidade intra-lista
- Cobertura do catálogo
- Novidade e taxa de exploração
