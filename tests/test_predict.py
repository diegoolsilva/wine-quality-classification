import numpy as np
import pickle
import pandas as pd
import pytest

from src.features import MODEL_FEATURES, RAW_FEATURES
from src.predict import load_model_bundle, predict_dataframe, resolve_threshold


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

    # A amostra MEDIUM (score 0,55) muda de classe entre 0,50 e 0,60.
    assert result_050["target_previsto"].tolist() == [0, 1, 1]
    assert result_060["target_previsto"].tolist() == [0, 0, 1]


def test_predict_dataframe_accepts_custom_threshold():
    # O programa não fica restrito aos três exemplos 0,50 / 0,60 / 0,70.
    result = predict_dataframe(sample_input(), model_bundle(), threshold=0.56)
    assert result["target_previsto"].tolist() == [0, 0, 1]
    assert result["threshold_aplicado"].tolist() == [0.56, 0.56, 0.56]


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_predict_dataframe_rejects_threshold_outside_zero_one(threshold):
    with pytest.raises(ValueError, match="Threshold inválido"):
        predict_dataframe(sample_input(), model_bundle(), threshold=threshold)


def test_predict_dataframe_rejects_missing_input_column():
    data = sample_input().drop(columns=["alcohol"])
    with pytest.raises(ValueError, match="Faltam colunas obrigatórias"):
        predict_dataframe(data, model_bundle(), threshold=0.50)


def test_resolve_threshold_uses_model_recommendation_when_not_provided():
    bundle = model_bundle()
    bundle["recommended_threshold"] = 0.27
    assert resolve_threshold(None, bundle) == pytest.approx(0.27)


def test_resolve_threshold_allows_manual_override():
    bundle = model_bundle()
    bundle["recommended_threshold"] = 0.27
    assert resolve_threshold(0.50, bundle) == pytest.approx(0.50)


def test_resolve_threshold_requires_recommendation_or_manual_value():
    with pytest.raises(ValueError, match="não contém um threshold recomendado"):
        resolve_threshold(None, model_bundle())


@pytest.mark.parametrize("model_columns", [RAW_FEATURES, MODEL_FEATURES])
def test_load_model_bundle_accepts_11_or_13_feature_artifact(tmp_path, model_columns):
    bundle = {
        "model": DummyModel(),
        "raw_feature_columns": RAW_FEATURES,
        "model_feature_columns": model_columns,
    }
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(bundle, model_file)

    loaded_bundle = load_model_bundle(model_path)
    assert loaded_bundle["model_feature_columns"] == model_columns
