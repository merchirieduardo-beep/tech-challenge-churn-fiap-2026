.PHONY: install lint test run train all clean help

help: ## Mostra comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependências do projeto
	pip install -e ".[dev]"

lint: ## Executa linting com ruff
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Formata código com ruff
	ruff check --fix src/ tests/
	ruff format src/ tests/

test: ## Executa testes com pytest
	pytest tests/ -v --cov=src --cov-report=term-missing

test-smoke: ## Executa apenas smoke tests
	pytest tests/ -v -m smoke

test-schema: ## Executa apenas schema tests
	pytest tests/ -v -m schema

test-api: ## Executa apenas API tests
	pytest tests/ -v -m api

train: ## Treina o modelo completo (baselines + MLP)
	python -m src.models.train

run: ## Inicia a API de inferência
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui: ## Inicia a interface do MLflow
	mlflow ui --host 0.0.0.0 --port 5000

all: install lint test ## Instala, lint e testa

clean: ## Remove artefatos temporários
	rm -rf __pycache__ .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
