# Modelo de risco de inadimplência com IA Generativa

Projeto de ponta a ponta em Machine Learning Engineering: um modelo de classificação de risco de crédito colocado em produção como API, com rastreamento de experimentos, integração contínua, monitoramento de drift e um agente de IA Generativa que explica as predições em linguagem natural.

> Status: **em andamento**

## Objetivo

Estimar a probabilidade de um cliente ficar inadimplente no mês seguinte e entregar essa estimativa de forma utilizável por outras aplicações e por pessoas:

- para sistemas, por meio de uma API REST empacotada em contêiner;
- para analistas, por meio de um agente que consulta o modelo e redige um parecer explicativo.

O foco não é só a métrica do modelo, mas o ciclo completo de MLOps: reprodutibilidade, testes, deploy, observabilidade e monitoramento.

## Base de dados

**Default of Credit Card Clients**, UCI Machine Learning Repository.

- Cerca de 30 mil clientes de cartão de crédito em Taiwan (2005)
- 23 variáveis explicativas: limite de crédito, dados demográficos, histórico de pagamentos, valores de fatura e de pagamentos dos últimos 6 meses
- Alvo binário: inadimplência no mês seguinte (taxa de cerca de 22%)
- Licença: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Fonte: Yeh, I. C. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

Os dados não são versionados no repositório. O script `scripts/download_data.py` baixa, padroniza e valida a base.

## Arquitetura planejada

```
                 ┌──────────────────────┐
                 │  Treino (Scikit-learn)│
                 │  + MLflow (tracking)  │
                 └──────────┬───────────┘
                            │ modelo serializado
                            ▼
┌──────────┐  requisição  ┌──────────────────────┐   log    ┌──────────────────┐
│ Cliente  │ ───────────► │ API FastAPI (Docker) │ ───────► │ Predições (logs) │
└──────────┘ ◄─────────── └──────────┬───────────┘          └────────┬─────────┘
              probabilidade          ▲                               │
                                     │ tool use                      ▼
                          ┌──────────┴───────────┐        ┌──────────────────┐
                          │ Agente AWS Bedrock   │        │ Evidently        │
                          │ (Converse API)       │        │ (drift de dados) │
                          └──────────────────────┘        └──────────────────┘
```

1. **Modelo:** regressão logística como baseline e gradient boosting, avaliados por AUC, KS e calibração, com experimentos rastreados no MLflow.
2. **API:** FastAPI empacotada em Docker, com testes automatizados (pytest) e integração contínua no GitHub Actions.
3. **Agente:** agente no AWS Bedrock que usa o modelo como ferramenta (tool use via Converse API) e gera um parecer explicativo.
4. **Observabilidade:** registro das predições da API e monitoramento de drift com Evidently.

## Stack

Python 3.12 · Scikit-learn · MLflow · FastAPI · Docker · pytest · GitHub Actions · AWS Bedrock · Evidently

## Estrutura do repositório

```
credit-risk-ml-genai/
├── data/raw/          base baixada (não versionada)
├── data/processed/    bases tratadas (não versionadas)
├── notebooks/         análise exploratória
├── scripts/           tarefas pontuais (download da base)
├── src/               código do modelo e da API
├── tests/             testes automatizados
└── docs/images/       imagens do README
```

## Como reproduzir

Requisitos: Python 3.12 e Git.

```bash
git clone https://github.com/vickcanto/credit-risk-ml-genai.git
cd credit-risk-ml-genai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```

## Roadmap

- [ ] Repositório no GitHub e download da base
- [ ] Análise exploratória
- [ ] Baseline com regressão logística e gradient boosting, rastreados no MLflow
- [ ] Avaliação por AUC, KS e calibração
- [ ] API FastAPI
- [ ] Empacotamento em Docker
- [ ] Testes automatizados com pytest
- [ ] Integração contínua no GitHub Actions
- [ ] Registro das predições da API
- [ ] Monitoramento de drift com Evidently
- [ ] Agente no AWS Bedrock com tool use
- [ ] README final com resultados e prints

## Autora

Victória Canto · [LinkedIn](https://www.linkedin.com/in/victória-canto-49851a2b0)
