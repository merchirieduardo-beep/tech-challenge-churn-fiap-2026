# Plano de Monitoramento — Churn Prediction

## Visão Geral

Este documento define o plano de monitoramento para o modelo de previsão de churn em produção, cobrindo métricas, alertas e playbook de resposta a incidentes.

## Arquitetura de Monitoramento

```
[API FastAPI] → [Logs Estruturados] → [Aggregator (ex: ELK/Datadog)]
     ↓                                         ↓
[MLflow Tracking]                    [Dashboard de Métricas]
     ↓                                         ↓
[Model Registry]                     [Sistema de Alertas]
```

## Métricas Monitoradas

### 1. Métricas de Infraestrutura (SLOs)

| Métrica | Threshold | Frequência |
|---------|-----------|------------|
| Latência p50 | < 50ms | Real-time |
| Latência p95 | < 100ms | Real-time |
| Latência p99 | < 200ms | Real-time |
| Disponibilidade | ≥ 99.5% | Diário |
| Error Rate (5xx) | < 0.1% | Real-time |
| Throughput | > 100 req/min | Real-time |

### 2. Métricas de Modelo

| Métrica | Threshold | Frequência |
|---------|-----------|------------|
| AUC-ROC (offline) | ≥ 0.75 | Semanal |
| Precision@recall=0.7 | ≥ 0.50 | Semanal |
| Prediction distribution shift | PSI < 0.1 | Diário |
| Feature drift (KL divergence) | < 0.05 | Diário |
| Calibration error | < 0.05 | Semanal |

### 3. Métricas de Negócio

| Métrica | Threshold | Frequência |
|---------|-----------|------------|
| Taxa de retenção (campanha) | > 15% | Mensal |
| Custo por churn evitado | < R$ 200 | Mensal |
| Volume de clientes alto risco | < 30% do total | Semanal |
| False positive rate (campanhas) | < 40% | Mensal |

## Alertas

### Nível CRÍTICO (resposta imediata)
- API indisponível (> 30s sem resposta)
- Error rate > 5% por 5 minutos
- Modelo retornando mesma predição para todos os inputs (collapsed output)

### Nível ALTO (resposta em 1h)
- Latência p95 > 200ms por 10 minutos
- Feature drift detectado (PSI > 0.2)
- AUC-ROC caiu abaixo de 0.70

### Nível MÉDIO (resposta em 24h)
- Prediction distribution shift (PSI entre 0.1 e 0.2)
- Volume de predições fora do baseline (±50%)
- Degradação gradual de métricas (tendência negativa por 7 dias)

### Nível BAIXO (próxima sprint)
- Feature com missing rate > 5%
- Log de warnings crescentes
- Necessidade de re-treinamento preventivo

## Playbook de Resposta

### Incidente: Modelo Degradado (AUC-ROC < 0.75)

1. **Diagnóstico** (15 min)
   - Verificar se há data drift nas features principais
   - Comparar distribuição de predições atual vs. baseline
   - Checar se houve mudança nos dados de input

2. **Mitigação** (30 min)
   - Se drift confirmado: acionar re-treinamento com dados recentes
   - Se não: verificar integridade do pipeline de features
   - Fallback: reverter para modelo anterior (versão MLflow)

3. **Resolução** (1-2 dias)
   - Re-treinar com janela de dados atualizada
   - Validar novo modelo com A/B test
   - Atualizar Model Card com novas métricas

4. **Post-mortem** (1 semana)
   - Documentar causa raiz
   - Implementar monitoramento adicional se necessário
   - Atualizar thresholds de alerta

### Incidente: API Indisponível

1. **Diagnóstico** (5 min)
   - Verificar health check endpoint
   - Checar logs de erro
   - Verificar recursos (CPU, memória, disco)

2. **Mitigação** (10 min)
   - Restart do serviço
   - Se persistir: escalar horizontalmente
   - Ativar fallback (regra de negócio simples)

3. **Resolução**
   - Identificar causa (OOM, deadlock, dependency failure)
   - Fix e deploy

## Ferramentas Recomendadas

| Categoria | Ferramenta | Uso |
|-----------|------------|-----|
| Logging | Structlog + ELK | Logs estruturados |
| Métricas | Prometheus + Grafana | Dashboards |
| Alertas | PagerDuty / Opsgenie | Notificações |
| ML Monitoring | MLflow + Evidently | Drift detection |
| APM | Datadog / New Relic | Performance |

## Frequência de Re-treinamento

- **Agendado**: Mensal (com dados dos últimos 6 meses)
- **Trigger automático**: Quando AUC-ROC cai abaixo de 0.75
- **Manual**: Após mudanças significativas no produto/mercado

## Dashboard Sugerido

```
┌─────────────────────────────────────────────────────┐
│  CHURN PREDICTION — MONITORING DASHBOARD            │
├─────────────────────┬───────────────────────────────┤
│ API Health: ✅ UP    │  Latency p95: 45ms           │
│ Model Version: 1.0  │  Requests/min: 234           │
├─────────────────────┼───────────────────────────────┤
│ AUC-ROC: 0.83       │  Feature Drift: LOW          │
│ F1: 0.63            │  Prediction Shift: NORMAL    │
├─────────────────────┼───────────────────────────────┤
│ High Risk Clients   │  Retention Rate (campaign)   │
│ 24.3% (↓1.2%)      │  18.5% (↑2.1%)              │
└─────────────────────┴───────────────────────────────┘
```
