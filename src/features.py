"""Funções de preparação das variáveis físico-químicas do projeto Wine Quality."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Colunas que devem chegar no arquivo de entrada, na ordem usada no treinamento.
RAW_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Novas variáveis criadas durante o Feature Engineering.
ENGINEERED_FEATURES = [
    "free_sulfur_ratio",
    "volatile_to_fixed_acidity",
]

MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


def add_domain_features(data: pd.DataFrame) -> pd.DataFrame:
    """Retorna as 11 variáveis originais mais as 2 features derivadas.

    A função é utilizada tanto no notebook de treinamento quanto no script
    de predição. Assim, a engenharia de atributos permanece consistente.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("O argumento 'data' precisa ser um pandas DataFrame.")

    missing_columns = [column for column in RAW_FEATURES if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Não foi possível criar as features. Colunas obrigatórias ausentes: "
            + ", ".join(missing_columns)
        )

    result = data.loc[:, RAW_FEATURES].copy()

    # Conversão explícita para números; valores inválidos são identificados abaixo.
    for column in RAW_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    invalid_columns = result.columns[result.isna().any()].tolist()
    if invalid_columns:
        raise ValueError(
            "Existem valores ausentes ou não numéricos nas colunas: "
            + ", ".join(invalid_columns)
        )

    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Os dados contêm valores infinitos ou não finitos.")

    if (result["total sulfur dioxide"] == 0).any():
        raise ValueError(
            "A coluna 'total sulfur dioxide' contém zero; não é possível calcular "
            "free_sulfur_ratio com denominador zero."
        )

    if (result["fixed acidity"] == 0).any():
        raise ValueError(
            "A coluna 'fixed acidity' contém zero; não é possível calcular "
            "volatile_to_fixed_acidity com denominador zero."
        )

    result["free_sulfur_ratio"] = (
        result["free sulfur dioxide"] / result["total sulfur dioxide"]
    )
    result["volatile_to_fixed_acidity"] = (
        result["volatile acidity"] / result["fixed acidity"]
    )

    return result.loc[:, MODEL_FEATURES]
