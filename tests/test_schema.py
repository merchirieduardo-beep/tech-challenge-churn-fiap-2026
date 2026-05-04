"""Schema tests: verificam validação de dados com Pandera e Pydantic."""

import pandas as pd
import pytest
from pydantic import ValidationError
from src.api.schemas import PredictionInput


@pytest.mark.schema
class TestSchemaValidation:
    """Testes de validação de schema para input/output."""

    def test_valid_prediction_input(self):
        """Verifica que input válido é aceito."""
        input_data = PredictionInput(
            gender="Male",
            SeniorCitizen=0,
            Partner="Yes",
            Dependents="No",
            tenure=24,
            PhoneService="Yes",
            MultipleLines="No",
            InternetService="Fiber optic",
            OnlineSecurity="No",
            OnlineBackup="Yes",
            DeviceProtection="No",
            TechSupport="No",
            StreamingTV="Yes",
            StreamingMovies="No",
            Contract="Month-to-month",
            PaperlessBilling="Yes",
            PaymentMethod="Electronic check",
            MonthlyCharges=79.85,
            TotalCharges=1919.40,
        )

        assert input_data.tenure == 24
        assert input_data.MonthlyCharges == 79.85

    def test_invalid_senior_citizen_value(self):
        """Verifica que SeniorCitizen fora do range é rejeitado."""
        with pytest.raises(ValidationError):
            PredictionInput(
                gender="Male",
                SeniorCitizen=2,  # Inválido
                Partner="Yes",
                Dependents="No",
                tenure=24,
                PhoneService="Yes",
                MultipleLines="No",
                InternetService="Fiber optic",
                OnlineSecurity="No",
                OnlineBackup="Yes",
                DeviceProtection="No",
                TechSupport="No",
                StreamingTV="Yes",
                StreamingMovies="No",
                Contract="Month-to-month",
                PaperlessBilling="Yes",
                PaymentMethod="Electronic check",
                MonthlyCharges=79.85,
                TotalCharges=1919.40,
            )

    def test_negative_tenure_rejected(self):
        """Verifica que tenure negativo é rejeitado."""
        with pytest.raises(ValidationError):
            PredictionInput(
                gender="Female",
                SeniorCitizen=0,
                Partner="No",
                Dependents="No",
                tenure=-1,  # Inválido
                PhoneService="Yes",
                MultipleLines="No",
                InternetService="DSL",
                OnlineSecurity="Yes",
                OnlineBackup="No",
                DeviceProtection="Yes",
                TechSupport="Yes",
                StreamingTV="No",
                StreamingMovies="No",
                Contract="Two year",
                PaperlessBilling="No",
                PaymentMethod="Mailed check",
                MonthlyCharges=45.50,
                TotalCharges=1090.00,
            )

    def test_negative_charges_rejected(self):
        """Verifica que MonthlyCharges negativo é rejeitado."""
        with pytest.raises(ValidationError):
            PredictionInput(
                gender="Male",
                SeniorCitizen=0,
                Partner="Yes",
                Dependents="Yes",
                tenure=36,
                PhoneService="Yes",
                MultipleLines="Yes",
                InternetService="DSL",
                OnlineSecurity="Yes",
                OnlineBackup="Yes",
                DeviceProtection="Yes",
                TechSupport="Yes",
                StreamingTV="Yes",
                StreamingMovies="Yes",
                Contract="One year",
                PaperlessBilling="Yes",
                PaymentMethod="Bank transfer (automatic)",
                MonthlyCharges=-10.0,  # Inválido
                TotalCharges=1000.00,
            )

    def test_missing_required_field_rejected(self):
        """Verifica que campo obrigatório ausente é rejeitado."""
        with pytest.raises(ValidationError):
            PredictionInput(
                gender="Male",
                # SeniorCitizen ausente
                Partner="Yes",
                Dependents="No",
                tenure=12,
                PhoneService="Yes",
                MultipleLines="No",
                InternetService="DSL",
                OnlineSecurity="No",
                OnlineBackup="No",
                DeviceProtection="No",
                TechSupport="No",
                StreamingTV="No",
                StreamingMovies="No",
                Contract="Month-to-month",
                PaperlessBilling="Yes",
                PaymentMethod="Electronic check",
                MonthlyCharges=50.0,
                TotalCharges=600.0,
            )

    def test_prediction_input_to_dataframe(self):
        """Verifica que o input pode ser convertido para DataFrame."""
        input_data = PredictionInput(
            gender="Female",
            SeniorCitizen=1,
            Partner="No",
            Dependents="No",
            tenure=1,
            PhoneService="No",
            MultipleLines="No phone service",
            InternetService="DSL",
            OnlineSecurity="No",
            OnlineBackup="No",
            DeviceProtection="No",
            TechSupport="No",
            StreamingTV="No",
            StreamingMovies="No",
            Contract="Month-to-month",
            PaperlessBilling="Yes",
            PaymentMethod="Electronic check",
            MonthlyCharges=29.85,
            TotalCharges=29.85,
        )

        df = pd.DataFrame([input_data.model_dump()])
        assert len(df) == 1
        assert "tenure" in df.columns
        assert df["tenure"].iloc[0] == 1
