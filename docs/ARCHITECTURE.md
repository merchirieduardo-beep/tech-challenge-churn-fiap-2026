# Arquitetura de Deploy — Churn Prediction

## Decisão: Real-Time (API REST)

### Justificativa

| Critério | Batch | Real-Time (escolhido) |
|----------|-------|----------------------|
| Latência | Horas | < 100ms |
| Freshness | Dados desatualizados | Predição instantânea |
| Caso de uso | Relatórios periódicos | Ação imediata na jornada |
| Complexidade | Menor | Moderada |
| Custo | Menor (compute sob demanda) | Maior (servidor always-on) |

**Motivo da escolha**: O caso de uso de retenção exige resposta rápida — quando um cliente acessa o portal ou liga para cancelar, a equipe precisa de score em tempo real para oferecer retenção personalizada.

## Arquitetura Geral

```
┌──────────────────────────────────────────────────────────────┐
│                      PRODUÇÃO                                 │
│                                                              │
│  ┌─────────┐     ┌─────────────┐     ┌──────────────────┐   │
│  │  Client  │────▶│  API Gateway │────▶│  FastAPI + Model  │  │
│  │ (CRM/App)│     │  (Kong/ALB)  │     │  (Container)      │  │
│  └─────────┘     └─────────────┘     └──────────────────┘   │
│                                              │               │
│                                              ▼               │
│                                       ┌────────────┐         │
│                                       │  MLflow    │         │
│                                       │  Registry  │         │
│                                       └────────────┘         │
│                                                              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                      TREINAMENTO                              │
│                                                              │
│  ┌──────────┐    ┌───────────┐    ┌─────────────────────┐   │
│  │  Dataset  │───▶│  Pipeline  │───▶│  MLflow Experiment  │  │
│  │  (S3/GCS) │    │  (sklearn) │    │  Tracking           │  │
│  └──────────┘    └───────────┘    └─────────────────────┘   │
│                        │                                     │
│                        ▼                                     │
│                 ┌─────────────┐                              │
│                 │  MLP Train   │                              │
│                 │  (PyTorch)   │                              │
│                 └─────────────┘                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Componentes

### 1. API de Inferência (FastAPI)
- **Endpoint principal**: `POST /predict`
- **Health check**: `GET /health`
- **Validação**: Pydantic schemas
- **Logging**: Structlog (JSON estruturado)
- **Middleware**: Latência tracking

### 2. Modelo (PyTorch MLP)
- Carregado na inicialização (lifespan)
- State dict + preprocessor (joblib)
- Inferência em CPU (suficiente para o volume)

### 3. Preprocessador (sklearn)
- ColumnTransformer serializado
- StandardScaler + OneHotEncoder
- Aplicado antes da inferência

### 4. MLflow
- Tracking de experimentos
- Model Registry para versionamento
- Artefatos persistidos (S3/GCS)

## Stack de Deploy (sugerida)

### Opção 1: AWS
```
ECR → ECS Fargate → ALB → Route 53
         │
         └── S3 (artefatos MLflow)
```

### Opção 2: GCP
```
Artifact Registry → Cloud Run → Load Balancer
                        │
                        └── GCS (artefatos MLflow)
```

### Opção 3: Azure
```
ACR → Azure Container Apps → Application Gateway
              │
              └── Blob Storage (artefatos MLflow)
```

### Deploy Escolhido: GCP Cloud Run (gratuito)
```
GitHub Repo → Artifact Registry → Cloud Run → URL pública (HTTPS)
                                      │
                                      └── Auto-scaling (0 a N instâncias)
```
**Motivo**: Tier gratuito generoso (2M requests/mês), escala para zero (custo zero quando inativo), deploy Docker nativo, HTTPS automático, integração com GCP.

## Containerização

O Dockerfile usa a variável `PORT` (injetada pelo Cloud Run automaticamente):

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY src/ src/
COPY models/ models/
ENV PORT=8080
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}
```

## Deploy no GCP Cloud Run (Passo a Passo)

### Pré-requisitos
1. Criar conta no GCP: https://cloud.google.com (free tier, sem cobrança)
2. Instalar Google Cloud CLI: https://cloud.google.com/sdk/docs/install
3. Criar um projeto GCP (ex: `churn-prediction-fiap`)

### Comandos de Deploy

```bash
# 1. Autenticar no GCP
gcloud auth login
gcloud config set project SEU_PROJECT_ID

# 2. Habilitar APIs necessárias
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# 3. Build e push da imagem (Cloud Build — gratuito 120 min/dia)
gcloud builds submit --tag gcr.io/SEU_PROJECT_ID/churn-prediction

# 4. Deploy no Cloud Run
gcloud run deploy churn-prediction-api \
    --image gcr.io/SEU_PROJECT_ID/churn-prediction \
    --platform managed \
    --region southamerica-east1 \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --max-instances 3 \
    --min-instances 0

# 5. URL pública gerada automaticamente:
#    https://churn-prediction-api-XXXXX-rj.a.run.app
```

### Tier Gratuito do Cloud Run
| Recurso | Limite Gratuito/Mês |
|---------|---------------------|
| Requests | 2.000.000 |
| CPU | 180.000 vCPU-segundos |
| Memória | 360.000 GiB-segundos |
| Networking | 1 GB de saída |

> Com min-instances=0, o serviço escala para zero e não consome nada quando inativo.

## Escalabilidade

- **Horizontal**: 2-5 réplicas com auto-scaling baseado em CPU/latência
- **Startup**: Modelo carregado uma vez na inicialização (~2s)
- **Memory**: ~500MB por réplica (modelo + preprocessor + runtime)
- **CPU**: 1 vCPU por réplica é suficiente para ~200 req/s

## CI/CD Pipeline

```
Git Push → Lint + Tests → Build Image → Push Registry → Deploy (staging) → Smoke Tests → Deploy (prod)
```
