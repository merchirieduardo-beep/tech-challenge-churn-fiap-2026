# Model Card — Churn Prediction MLP

## Informações Gerais

| Campo | Valor |
|-------|-------|
| **Nome do Modelo** | ChurnMLP |
| **Versão** | 1.0.0 |
| **Tipo** | Multi-Layer Perceptron (classificação binária) |
| **Framework** | PyTorch 2.x |
| **Data de Treinamento** | 2026 |
| **Responsáveis** | Grupo X — Tech Challenge FIAP |

## Objetivo

Prever a probabilidade de churn (cancelamento) de clientes de uma operadora de telecomunicações, permitindo ações proativas de retenção.

## Arquitetura

```
Input (46 features)
  → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64) → BatchNorm → ReLU → Dropout(0.2)
  → Linear(32) → BatchNorm → ReLU → Dropout(0.1)
  → Linear(1) → Sigmoid
```

### Hiperparâmetros
- **Otimizador**: Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early Stopping**: patience=15, min_delta=1e-4
- **Batch Size**: 64
- **Class Weight**: ~2.8 (compensação do desbalanceamento)
- **Inicialização**: Xavier Uniform

## Performance

### Métricas no Conjunto de Teste (20% holdout)

| Métrica | Valor |
|---------|-------|
| **AUC-ROC** | 0.8409 |
| **F1-Score** | 0.6128 |
| **Precision** | 0.5513 |
| **Recall** | 0.6898 |
| **Accuracy** | 0.7686 |

### Comparação com Baselines

| Modelo | AUC-ROC | F1 | Recall |
|--------|---------|-----|--------|
| DummyClassifier | 0.5065 | 0.2762 | 0.2776 |
| Logistic Regression | 0.8455 | 0.6284 | 0.8013 |
| Random Forest | 0.8207 | 0.5379 | 0.4702 |
| Gradient Boosting | 0.8474 | 0.5879 | 0.5291 |
| **MLP (PyTorch)** | **0.8409** | **0.6128** | **0.6898** |

## Dataset

- **Fonte**: Telco Customer Churn (IBM/Kaggle)
- **Tamanho**: ~7.043 registros
- **Features**: 19 (4 numéricas + 15 categóricas)
- **Target**: Binário (Churn: Yes/No)
- **Desbalanceamento**: ~26.5% positivos (churn)

## Limitações

1. **Dados estáticos**: O modelo não captura dinâmicas temporais de comportamento
2. **Dataset limitado**: ~7K registros pode não generalizar para bases muito maiores
3. **Sem features comportamentais**: Não inclui dados de uso (chamadas, dados, reclamações)
4. **Bias geográfico**: Dataset pode não representar mercados fora dos EUA
5. **Threshold fixo**: Requer calibração do threshold por contexto de negócio

## Vieses Identificados

- **Gênero**: Não há diferença significativa na performance entre gêneros
- **Idosos (SeniorCitizen)**: Performance ligeiramente inferior para idosos (taxa de churn mais alta neste grupo)
- **Tenure**: Melhor performance para clientes com tenure > 12 meses (mais dados de histórico)

## Cenários de Falha

1. **Clientes novos**: Alta incerteza para tenure < 3 meses (pouco histórico)
2. **Mudanças de mercado**: Não detecta churn motivado por fatores externos (novo concorrente)
3. **Fraude/Involuntário**: Não distingue churn voluntário de cancelamento por inadimplência
4. **Features ausentes**: Se features estiverem faltando, o modelo pode dar predições imprecisas

## Uso Ético

- **Uso pretendido**: Campanhas de retenção direcionadas
- **Uso NÃO recomendado**: Decisões automatizadas de cancelamento de serviço, precificação discriminatória
- **Transparência**: Clientes devem ser informados se ações foram baseadas em modelos preditivos

## Reprodutibilidade

- Seed global: 42
- Todas as seeds são fixadas (Python, NumPy, PyTorch, CUDA)
- Experimentos rastreados no MLflow
- Pipeline completo reproduzível via `make train`

## Monitoramento Recomendado

- Drift de features (PSI/KL divergence mensalmente)
- Degradação de AUC-ROC (alerta se < 0.75)
- Volume de predições por dia
- Distribuição das probabilidades preditas
