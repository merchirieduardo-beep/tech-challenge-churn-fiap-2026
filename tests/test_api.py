"""API tests: verificam endpoints da FastAPI."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.mark.api
class TestAPIEndpoints:
    """Testes para os endpoints da API."""

    @pytest.fixture
    def client(self):
        """Client de teste da FastAPI."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Verifica que /health retorna status 200."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_predict_endpoint_without_model(self, client):
        """Verifica que /predict retorna 503 quando modelo não está carregado."""
        payload = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 24,
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
            "TotalCharges": 1919.40,
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 503

    def test_predict_endpoint_invalid_input(self, client):
        """Verifica que /predict rejeita input inválido."""
        payload = {
            "gender": "Male",
            "SeniorCitizen": 5,  # Inválido
            "tenure": -10,  # Inválido
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_endpoint_missing_fields(self, client):
        """Verifica que /predict rejeita input com campos faltantes."""
        payload = {"gender": "Male", "tenure": 24}

        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_health_response_structure(self, client):
        """Verifica a estrutura completa da resposta de health."""
        response = client.get("/health")
        data = response.json()

        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["version"], str)

    def test_latency_header_present(self, client):
        """Verifica que o header de latência está presente."""
        response = client.get("/health")

        assert "x-process-time-ms" in response.headers

    def test_docs_endpoint_available(self, client):
        """Verifica que a documentação OpenAPI está acessível."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_available(self, client):
        """Verifica que o schema OpenAPI está acessível."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert "/predict" in data["paths"]
        assert "/health" in data["paths"]
