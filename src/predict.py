"""Gera previsões de qualidade de vinhos a partir de um arquivo CSV.

Execute a partir da raiz do projeto:
    python -m src.predict --input data/examples/novos_vinhos_exemplo.csv --threshold 0.50

ATENÇÃO: carregue somente arquivos pickle criados por você e de origem confiável.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.features import MODEL_FEATURES, RAW_FEATURES, add_domain_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "wine_quality_model.pkl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "predictions" / "predictions.csv"
ALLOWED_THRESHOLDS = (0.50, 0.60, 0.70)


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

    if model_bundle["model_feature_columns"] != MODEL_FEATURES:
        raise ValueError(
            "As features do modelo não correspondem à versão atual de src/features.py."
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
    if threshold not in ALLOWED_THRESHOLDS:
        raise ValueError(
            f"Threshold inválido. Utilize um dos cenários avaliados: {ALLOWED_THRESHOLDS}."
        )

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
        required=True,
        type=float,
        choices=ALLOWED_THRESHOLDS,
        help="Cenário avaliado: 0.50 (padrão), 0.60 (intermediário) ou 0.70 (conservador).",
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


def main() -> None:
    args = parse_arguments()

    if not args.input.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {args.input}")

    input_data = pd.read_csv(args.input)
    model_bundle = load_model_bundle(args.model)
    predictions = predict_dataframe(input_data, model_bundle, args.threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False, encoding="utf-8-sig")

    positive_count = int(predictions["target_previsto"].sum())
    print(f"Amostras processadas: {len(predictions)}")
    print(f"Classificadas como alta qualidade: {positive_count}")
    print(f"Threshold aplicado: {args.threshold:.2f}")
    print(f"Arquivo de resultados salvo em: {args.output.resolve()}")


if __name__ == "__main__":
    main()
