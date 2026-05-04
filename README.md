# Churn Prediction — MLP Pipeline End-to-End

> **Tech Challenge Fase 01** — Rede Neural para Previsão de Churn com Pipeline Profissional

## Descrição

Projeto completo de Machine Learning para prever o churn (cancelamento) de clientes em uma operadora de telecomunicações. Inclui desde a análise exploratória até a API de inferência em produção.

### Highlights
- **MLP (PyTorch)** com early stopping, batch normalization e class weighting
- **Baselines** (Scikit-Learn): DummyClassifier, Logistic Regression, Random Forest, Gradient Boosting
- **MLflow** para tracking de experimentos
- **FastAPI** para inferência real-time
- **Testes automatizados** (smoke, schema, API)
- **Logging estruturado** (sem print!)

## Arquitetura

```
├── src/
│   ├── api/          # FastAPI endpoints (/predict, /health)
│   ├── data/         # Carregamento e validação de dados
│   ├── features/     # Pipeline de pré-processamento
│   ├── models/       # MLP (PyTorch) + Baselines (sklearn)
│   └── utils/        # Logging, reprodutibilidade
├── data/
│   ├── raw/          # Dataset original (não versionado)
│   └── processed/    # Dados processados (não versionado)
├── models/           # Artefatos treinados (.pth, .joblib)
├── notebooks/        # EDA + Treinamento interativo
├── tests/            # Pytest (smoke, schema, API)
├── docs/             # Model Card, Monitoramento, Arquitetura
├── pyproject.toml    # Single source of truth
└── Makefile          # Comandos úteis
```

## Setup

### Pré-requisitos
- Python 3.10+
- pip

### Instalação

```bash
# Clone o repositório
git clone <url-do-repositório>
cd churn-prediction

# Instale as dependências
make install
# ou: pip install -e ".[dev]"
```

### Dataset

Baixe o dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) e coloque em `data/raw/telco_churn.csv`.

```bash
# Estrutura esperada
data/raw/telco_churn.csv
```

## Execução

### Treinamento Completo

```bash
# Treina baselines + MLP com MLflow tracking
make train
# ou: python -m src.models.train
```

### API de Inferência

```bash
# Inicia a API (requer modelo treinado)
make run
# ou: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints:**
- `GET /health` — Health check
- `POST /predict` — Predição de churn
- `GET /docs` — Documentação interativa (Swagger)

### Exemplo de Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 958.20
  }'
```

**Resposta:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7234,
  "risk_level": "alto"
}
```

### MLflow UI

```bash
make mlflow-ui
# Acesse http://localhost:5000
```

## Testes

```bash
# Todos os testes
make test

# Por categoria
make test-smoke    # Testes de fumaça
make test-schema   # Validação de schemas
make test-api      # Endpoints da API
```

## Linting

```bash
# Verificar
make lint

# Corrigir automaticamente
make format
```

## Deploy (GCP Cloud Run — Gratuito)

```bash
# Instalar gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project SEU_PROJECT_ID

# Habilitar APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# Build e deploy
gcloud builds submit --tag gcr.io/SEU_PROJECT_ID/churn-prediction
gcloud run deploy churn-prediction-api \
    --image gcr.io/SEU_PROJECT_ID/churn-prediction \
    --platform managed \
    --region southamerica-east1 \
    --allow-unauthenticated \
    --memory 512Mi --cpu 1 \
    --max-instances 3 --min-instances 0
```

> Detalhes completos em [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Documentação

- [Model Card](docs/MODEL_CARD.md) — Performance, limitações, vieses
- [Arquitetura de Deploy](docs/ARCHITECTURE.md) — Decisões técnicas e guia GCP Cloud Run
- [Plano de Monitoramento](docs/MONITORING.md) — Métricas, alertas, playbook

## Decisões Técnicas

| Decisão | Escolha | Motivo |
|---------|---------|--------|
| Framework DL | PyTorch | Flexibilidade, debugging, ecossistema |
| Preprocessamento | sklearn Pipeline | Reprodutibilidade, serialização |
| API | FastAPI | Async, validação automática, docs |
| Experiment Tracking | MLflow | Open-source, model registry |
| Linting | Ruff | Velocidade, all-in-one |
| Logging | Structlog | Estruturado, sem print() |
| Validação | Pydantic + Pandera | Type safety em runtime |

## Métricas Obtidas

| Modelo | AUC-ROC | F1 | Precision | Recall | Accuracy |
|--------|---------|-----|-----------|--------|----------|
| DummyClassifier | 0.5065 | 0.2762 | 0.2748 | 0.2776 | 0.6140 |
| Logistic Regression | 0.8455 | 0.6284 | 0.5168 | 0.8013 | 0.7485 |
| Random Forest | 0.8207 | 0.5379 | 0.6282 | 0.4702 | 0.7856 |
| Gradient Boosting | 0.8474 | 0.5879 | 0.6614 | 0.5291 | 0.8032 |
| **MLP (PyTorch)** | **0.8409** | **0.6128** | **0.5513** | **0.6898** | **0.7686** |

> Métricas reais obtidas com seed=42 no conjunto de teste (20% holdout).

## Equipe

- Grupo X — Tech Challenge FIAP Fase 01

## Licença

MIT
