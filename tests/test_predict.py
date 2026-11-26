import numpy as np
import pandas as pd
import pytest

from src.features import MODEL_FEATURES, RAW_FEATURES
from src.predict import predict_dataframe


class DummyModel:
    """Modelo simples de teste: usa o teor alcoólico para gerar scores previsíveis."""

    def predict_proba(self, X):
        positive_scores = (X["alcohol"].to_numpy(dtype=float) / 20).clip(0, 1)
        return np.column_stack([1 - positive_scores, positive_scores])


def sample_input():
    base = {
        "fixed acidity": 7.0,
        "volatile acidity": 0.5,
        "citric acid": 0.2,
        "residual sugar": 2.0,
        "chlorides": 0.08,
        "free sulfur dioxide": 10.0,
        "total sulfur dioxide": 40.0,
        "density": 0.996,
        "pH": 3.3,
        "sulphates": 0.6,
        "alcohol": 10.5,
    }

    row_low = base.copy()
    row_low["alcohol"] = 8.0   # score de teste = 0,40

    row_medium = base.copy()
    row_medium["alcohol"] = 11.0  # score de teste = 0,55

    row_high = base.copy()
    row_high["alcohol"] = 14.0  # score de teste = 0,70

    return pd.DataFrame([
        {"sample_id": "LOW", **row_low},
        {"sample_id": "MEDIUM", **row_medium},
        {"sample_id": "HIGH", **row_high},
    ])


def model_bundle():
    return {
        "model": DummyModel(),
        "raw_feature_columns": RAW_FEATURES,
        "model_feature_columns": MODEL_FEATURES,
    }


def test_predict_dataframe_returns_scores_and_labels():
    result = predict_dataframe(sample_input(), model_bundle(), threshold=0.50)

    assert result["score_modelo_alta_qualidade"].tolist() == pytest.approx(
        [0.40, 0.55, 0.70]
    )
    assert result["classificacao_prevista"].tolist() == [
        "Baixa/Média qualidade",
        "Alta qualidade",
        "Alta qualidade",
    ]
    assert result["sample_id"].tolist() == ["LOW", "MEDIUM", "HIGH"]


def test_threshold_changes_class_without_changing_scores():
    data = sample_input()
    result_050 = predict_dataframe(data, model_bundle(), threshold=0.50)
    result_060 = predict_dataframe(data, model_bundle(), threshold=0.60)

    # Alterar o threshold não deve alterar os scores calculados pelo modelo.
    assert result_050["score_modelo_alta_qualidade"].tolist() == pytest.approx(
        result_060["score_modelo_alta_qualidade"].tolist()
    )

    # A amostra MEDIUM (score 0,55) muda de classe entre os dois thresholds.
    assert result_050["target_previsto"].tolist() == [0, 1, 1]
    assert result_060["target_previsto"].tolist() == [0, 0, 1]


def test_predict_dataframe_rejects_unapproved_threshold():
    with pytest.raises(ValueError, match="Threshold inválido"):
        predict_dataframe(sample_input(), model_bundle(), threshold=0.55)


def test_predict_dataframe_rejects_missing_input_column():
    data = sample_input().drop(columns=["alcohol"])

    with pytest.raises(ValueError, match="Faltam colunas obrigatórias"):
        predict_dataframe(data, model_bundle(), threshold=0.50)
