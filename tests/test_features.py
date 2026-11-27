import pandas as pd
import pytest

from src.features import RAW_FEATURES, add_domain_features


def sample_input():
    """Cria uma amostra mínima válida com as 11 variáveis de entrada."""
    values = {
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
    return pd.DataFrame([{column: values[column] for column in RAW_FEATURES}])


def test_add_domain_features_creates_expected_columns():
    result = add_domain_features(sample_input())

    assert "free_sulfur_ratio" in result.columns
    assert "volatile_to_fixed_acidity" in result.columns
    assert result.loc[0, "free_sulfur_ratio"] == pytest.approx(0.25)
    assert result.loc[0, "volatile_to_fixed_acidity"] == pytest.approx(0.5 / 7.0)


def test_add_domain_features_rejects_missing_columns():
    data = sample_input().drop(columns=["alcohol"])

    with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
        add_domain_features(data)


def test_add_domain_features_rejects_zero_denominator():
    data = sample_input()
    data.loc[0, "total sulfur dioxide"] = 0

    with pytest.raises(ValueError, match="total sulfur dioxide.*zero|denominador zero"):
        add_domain_features(data)


def test_add_domain_features_rejects_free_sulfur_above_total():
    data = sample_input()
    data.loc[0, "free sulfur dioxide"] = 50.0
    data.loc[0, "total sulfur dioxide"] = 40.0

    with pytest.raises(ValueError, match="free sulfur dioxide.*maior que.*total sulfur dioxide"):
        add_domain_features(data)
