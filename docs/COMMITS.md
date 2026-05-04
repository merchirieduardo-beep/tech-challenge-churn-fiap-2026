# Guia de Commits — Histórico Limpo e Significativo

## Padrão de Commit Messages

Seguimos o padrão **Conventional Commits** adaptado com mensagens descritivas e ação clara.

### Formato

```
<tipo>: <descrição concisa da mudança>
```

### Tipos

| Tipo | Uso |
|------|-----|
| `feat` | Nova funcionalidade |
| `fix` | Correção de bug |
| `docs` | Documentação |
| `test` | Testes |
| `refactor` | Refatoração (sem mudar comportamento) |
| `chore` | Configuração, dependências, CI |
| `perf` | Otimização de performance |

---

## Commits Sugeridos (Ordem Cronológica)

Execute esses commits na ordem abaixo para construir um histórico limpo e progressivo:

### Etapa 1 — Setup e Estrutura

```bash
git init
git add pyproject.toml Makefile .gitignore README.md
git commit -m "chore: initialize project structure with pyproject.toml, Makefile, and .gitignore"

git add src/__init__.py src/utils/
git commit -m "feat: add structured logging and reproducibility utilities"

git add src/data/
git commit -m "feat: add data loading module with Pandera schema validation"

git add src/features/
git commit -m "feat: add preprocessing pipeline with custom sklearn transformers"
```

### Etapa 2 — Modelagem

```bash
git add src/models/baseline.py
git commit -m "feat: add baseline models (Dummy, LogReg, RF, GBM) with stratified CV"

git add src/models/mlp.py
git commit -m "feat: add PyTorch MLP with early stopping, batch norm, and class weighting"

git add src/models/train.py
git commit -m "feat: add full training pipeline with MLflow experiment tracking"
```

### Etapa 3 — API e Testes

```bash
git add src/api/
git commit -m "feat: add FastAPI inference API with /predict and /health endpoints"

git add tests/test_smoke.py
git commit -m "test: add smoke tests for MLP forward pass and early stopping"

git add tests/test_schema.py
git commit -m "test: add schema validation tests for Pydantic input/output"

git add tests/test_api.py tests/conftest.py
git commit -m "test: add API endpoint tests (health, predict, validation)"
```

### Etapa 4 — Documentação e Notebooks

```bash
git add notebooks/01_eda_baselines.ipynb
git commit -m "docs: add EDA notebook with ML Canvas and baseline analysis"

git add notebooks/02_mlp_training.ipynb
git commit -m "docs: add MLP training notebook with cost analysis and model comparison"

git add docs/MODEL_CARD.md
git commit -m "docs: add Model Card with performance, limitations, and bias analysis"

git add docs/ARCHITECTURE.md
git commit -m "docs: add deployment architecture documentation (real-time API)"

git add docs/MONITORING.md
git commit -m "docs: add monitoring plan with metrics, alerts, and incident playbook"

git add Dockerfile
git commit -m "chore: add Dockerfile for containerized deployment"
```

### Commit Final

```bash
git add -A
git commit -m "docs: finalize README with setup instructions, examples, and architecture"
```

---

## Dicas para Histórico Limpo

1. **Um commit = uma mudança lógica** — não misture features com fixes
2. **Mensagem no imperativo** — "add", "fix", "update" (não "added", "fixing")
3. **Não committar dados** — dataset fica no .gitignore
4. **Não committar artefatos** — .pth, .joblib ficam no .gitignore
5. **Commits atômicos** — cada commit deve compilar/funcionar sozinho

## Exemplo de Git Log Esperado

```
* docs: finalize README with setup instructions, examples, and architecture
* chore: add Dockerfile for containerized deployment
* docs: add monitoring plan with metrics, alerts, and incident playbook
* docs: add deployment architecture documentation (real-time API)
* docs: add Model Card with performance, limitations, and bias analysis
* docs: add MLP training notebook with cost analysis and model comparison
* docs: add EDA notebook with ML Canvas and baseline analysis
* test: add API endpoint tests (health, predict, validation)
* test: add schema validation tests for Pydantic input/output
* test: add smoke tests for MLP forward pass and early stopping
* feat: add FastAPI inference API with /predict and /health endpoints
* feat: add full training pipeline with MLflow experiment tracking
* feat: add PyTorch MLP with early stopping, batch norm, and class weighting
* feat: add baseline models (Dummy, LogReg, RF, GBM) with stratified CV
* feat: add preprocessing pipeline with custom sklearn transformers
* feat: add data loading module with Pandera schema validation
* feat: add structured logging and reproducibility utilities
* chore: initialize project structure with pyproject.toml, Makefile, and .gitignore
```
