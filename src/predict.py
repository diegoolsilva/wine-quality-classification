"""Gera previsões de qualidade de vinhos a partir de um arquivo CSV.

Execute a partir da raiz do projeto:
    python -m src.predict --input data/examples/novos_vinhos_exemplo.csv

Para informar manualmente outro threshold:
    python -m src.predict --input data/examples/novos_vinhos_exemplo.csv --threshold 0.50

Quando --threshold não é informado, o script usa o valor recomendado salvo no Pickle.

ATENÇÃO: carregue somente arquivos pickle criados por você e de origem confiável.
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import pandas as pd

from src.features import MODEL_FEATURES, RAW_FEATURES, add_domain_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "wine_quality_model.pkl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "predictions" / "predictions.csv"


def load_model_bundle(model_path: Path) -> dict:
    """Carrega o pacote contendo o modelo e seus metadados."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em '{model_path}'. Execute primeiro a seção "
            "de exportação do modelo no notebook."
        )

    # Nunca carregue pickle recebido de terceiros ou de origem desconhecida.
    with model_path.open("rb") as model_file:
        model_bundle = pickle.load(model_file)

    required_keys = {"model", "raw_feature_columns", "model_feature_columns"}
    missing_keys = required_keys - set(model_bundle)
    if missing_keys:
        raise ValueError(
            "O arquivo pickle não contém os metadados esperados: "
            + ", ".join(sorted(missing_keys))
        )

    if model_bundle["raw_feature_columns"] != RAW_FEATURES:
        raise ValueError(
            "As colunas de entrada registradas no modelo não correspondem à versão "
            "atual de src/features.py. Verifique se os arquivos pertencem à mesma versão."
        )

    supported_feature_sets = (RAW_FEATURES, MODEL_FEATURES)
    if model_bundle["model_feature_columns"] not in supported_feature_sets:
        raise ValueError(
            "As features registradas no modelo não correspondem aos conjuntos "
            "suportados por src/features.py (11 originais ou 13 com Feature Engineering)."
        )

    return model_bundle


def prepare_input_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Valida o CSV e devolve as features prontas e os identificadores das amostras."""
    if data.empty:
        raise ValueError("O arquivo de entrada não contém nenhuma amostra.")

    if data.columns.duplicated().any():
        duplicated = data.columns[data.columns.duplicated()].tolist()
        raise ValueError("O arquivo possui nomes de colunas repetidos: " + ", ".join(duplicated))

    # sample_id é opcional e serve apenas para identificar a amostra no resultado.
    allowed_columns = set(RAW_FEATURES) | {"sample_id"}
    missing_columns = sorted(set(RAW_FEATURES) - set(data.columns))
    unknown_columns = sorted(set(data.columns) - allowed_columns)

    if missing_columns:
        raise ValueError(
            "Faltam colunas obrigatórias no CSV: " + ", ".join(missing_columns)
        )
    if unknown_columns:
        raise ValueError(
            "O CSV contém colunas não esperadas: " + ", ".join(unknown_columns)
            + ". Remova colunas como quality/target; elas não são necessárias para prever."
        )

    if "sample_id" in data.columns:
        sample_ids = data["sample_id"].astype("string")
        if sample_ids.isna().any() or sample_ids.str.strip().eq("").any():
            raise ValueError("A coluna 'sample_id' possui identificadores vazios.")
        if sample_ids.duplicated().any():
            raise ValueError("A coluna 'sample_id' deve possuir identificadores únicos.")
    else:
        sample_ids = pd.Series(
            [f"amostra_{number:04d}" for number in range(1, len(data) + 1)],
            name="sample_id",
        )

    raw_features = data.loc[:, RAW_FEATURES].copy()
    model_features = add_domain_features(raw_features)
    return model_features, sample_ids.reset_index(drop=True)


def predict_dataframe(
    data: pd.DataFrame,
    model_bundle: dict,
    threshold: float,
) -> pd.DataFrame:
    """Gera score e classe prevista para cada linha recebida."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Threshold inválido. Informe um número entre 0 e 1, inclusive.")
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ValueError("Threshold inválido. Informe um número entre 0 e 1, inclusive.")
    threshold = float(threshold)

    model_features, sample_ids = prepare_input_data(data)
    expected_columns = model_bundle["model_feature_columns"]
    model_features = model_features.loc[:, expected_columns]

    model = model_bundle["model"]
    if not hasattr(model, "predict_proba"):
        raise TypeError("O modelo salvo não disponibiliza o método predict_proba().")

    scores = model.predict_proba(model_features)[:, 1]
    predictions = (scores >= threshold).astype(int)

    results = data.loc[:, RAW_FEATURES].copy().reset_index(drop=True)
    results.insert(0, "sample_id", sample_ids)
    results["score_modelo_alta_qualidade"] = scores
    results["threshold_aplicado"] = threshold
    results["target_previsto"] = predictions
    results["classificacao_prevista"] = pd.Series(predictions).map(
        {0: "Baixa/Média qualidade", 1: "Alta qualidade"}
    ).to_numpy()

    return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classifica amostras de vinho usando o modelo treinado no projeto."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Caminho do arquivo CSV com as 11 variáveis físico-químicas.",
    )
    parser.add_argument(
        "--threshold",
        required=False,
        type=float,
        default=None,
        help="Threshold entre 0 e 1. Se omitido, utiliza o threshold recomendado salvo no Pickle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Caminho do CSV de saída (padrão: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Caminho do modelo pickle (padrão: {DEFAULT_MODEL_PATH}).",
    )
    return parser.parse_args()



def resolve_threshold(user_threshold: float | None, model_bundle: dict) -> float:
    """Usa o threshold informado pelo usuário ou a recomendação salva no modelo."""
    if user_threshold is not None:
        return user_threshold

    recommended_threshold = model_bundle.get("recommended_threshold")
    if recommended_threshold is None:
        raise ValueError(
            "O arquivo do modelo não contém um threshold recomendado. "
            "Informe --threshold explicitamente ou recrie o Pickle executando a seção 8 do notebook atualizado."
        )
    return recommended_threshold


def main() -> None:
    args = parse_arguments()

    if not args.input.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {args.input}")

    input_data = pd.read_csv(args.input)
    model_bundle = load_model_bundle(args.model)
    effective_threshold = resolve_threshold(args.threshold, model_bundle)
    predictions = predict_dataframe(input_data, model_bundle, effective_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False, encoding="utf-8-sig")

    positive_count = int(predictions["target_previsto"].sum())
    print(f"Amostras processadas: {len(predictions)}")
    print(f"Classificadas como alta qualidade: {positive_count}")
    print(f"Threshold aplicado: {effective_threshold:.2f}")
    print(f"Arquivo de resultados salvo em: {args.output.resolve()}")


if __name__ == "__main__":
    main()
