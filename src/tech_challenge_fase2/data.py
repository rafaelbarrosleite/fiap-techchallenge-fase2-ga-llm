"""Carga, validacao e divisao sem vazamento do dataset."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    EXPECTED_DATASET_SHA256,
    PHASE1_VALIDATION_SIZE_WITHIN_DEVELOPMENT,
    RANDOM_STATE,
    TEST_SIZE,
)


@dataclass(frozen=True)
class DevelopmentTestSplit:
    """Conjunto de desenvolvimento e teste final, preservando indices."""

    X_development: pd.DataFrame
    X_test: pd.DataFrame
    y_development: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class Phase1Split:
    """Divisao 60/20/20 usada originalmente na Fase 1."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def file_sha256(path: Path) -> str:
    """Calcula o SHA-256 sem carregar o arquivo inteiro na memoria."""

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(
    path: Path,
    *,
    require_expected_hash: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega e valida a mesma base da Fase 1.

    A coluna alvo e codificada como maligno=1 e benigno=0. `id` e a coluna
    vazia exportada pelo CSV sao removidas antes da modelagem.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset nao encontrado: {path}")

    observed_hash = file_sha256(path)
    if require_expected_hash and observed_hash != EXPECTED_DATASET_SHA256:
        raise ValueError(
            "O hash do dataset difere da copia auditada da Fase 1: "
            f"{observed_hash}"
        )

    frame = pd.read_csv(path)
    required_columns = {"id", "diagnosis", "Unnamed: 32"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Colunas esperadas ausentes: {sorted(missing_columns)}")
    if frame.shape != (569, 33):
        raise ValueError(f"Dimensao inesperada do CSV: {frame.shape}")
    if not frame["Unnamed: 32"].isna().all():
        raise ValueError("A coluna 'Unnamed: 32' deixou de ser totalmente vazia.")
    if frame["id"].duplicated().any():
        raise ValueError("Existem identificadores duplicados no dataset.")
    if frame.duplicated().any():
        raise ValueError("Existem linhas completas duplicadas no dataset.")
    if set(frame["diagnosis"].unique()) != {"B", "M"}:
        raise ValueError("O alvo deve conter somente os valores 'B' e 'M'.")

    model_frame = frame.drop(columns=["id", "Unnamed: 32"])
    if model_frame.drop(columns=["diagnosis"]).isna().any().any():
        raise ValueError("Existem valores ausentes nas variaveis preditoras.")

    y = model_frame["diagnosis"].map({"B": 0, "M": 1}).astype("int8")
    X = model_frame.drop(columns=["diagnosis"])
    if X.shape != (569, 30):
        raise ValueError(f"Matriz de atributos inesperada: {X.shape}")
    return X, y


def split_development_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DevelopmentTestSplit:
    """Reserva o teste final antes de qualquer ajuste ou otimizacao."""

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DevelopmentTestSplit(X_development, X_test, y_development, y_test)


def reproduce_phase1_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = RANDOM_STATE,
) -> Phase1Split:
    """Reproduz exatamente os dois cortes estratificados do notebook original."""

    first = split_development_test(X, y, random_state=random_state)
    X_train, X_validation, y_train, y_validation = train_test_split(
        first.X_development,
        first.y_development,
        test_size=PHASE1_VALIDATION_SIZE_WITHIN_DEVELOPMENT,
        random_state=random_state,
        stratify=first.y_development,
    )
    return Phase1Split(
        X_train,
        X_validation,
        first.X_test,
        y_train,
        y_validation,
        first.y_test,
    )

