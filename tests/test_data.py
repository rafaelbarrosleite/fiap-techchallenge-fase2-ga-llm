from pathlib import Path

import pandas as pd

from tech_challenge_fase2.config import (
    DEFAULT_DATA_PATH,
    EXPECTED_DATASET_SHA256,
)
from tech_challenge_fase2.data import (
    file_sha256,
    load_dataset,
    reproduce_phase1_split,
    split_development_test,
)


def test_dataset_is_the_audited_phase1_copy() -> None:
    X, y = load_dataset(DEFAULT_DATA_PATH)

    assert file_sha256(DEFAULT_DATA_PATH) == EXPECTED_DATASET_SHA256
    assert X.shape == (569, 30)
    assert y.value_counts().to_dict() == {0: 357, 1: 212}
    assert "id" not in X.columns
    assert "Unnamed: 32" not in X.columns
    assert not X.isna().any().any()


def test_final_test_is_stratified_reproducible_and_disjoint() -> None:
    X, y = load_dataset(DEFAULT_DATA_PATH)
    first = split_development_test(X, y)
    second = split_development_test(X, y)

    assert len(first.X_development) == 455
    assert len(first.X_test) == 114
    assert set(first.X_development.index).isdisjoint(first.X_test.index)
    pd.testing.assert_index_equal(first.X_test.index, second.X_test.index)
    assert first.y_test.value_counts().to_dict() == {0: 72, 1: 42}


def test_phase1_split_reproduces_60_20_20() -> None:
    X, y = load_dataset(DEFAULT_DATA_PATH)
    split = reproduce_phase1_split(X, y)

    assert (len(split.X_train), len(split.X_validation), len(split.X_test)) == (
        341,
        114,
        114,
    )
    all_indices = (
        set(split.X_train.index)
        | set(split.X_validation.index)
        | set(split.X_test.index)
    )
    assert len(all_indices) == len(X)
    assert set(split.X_train.index).isdisjoint(split.X_validation.index)
    assert set(split.X_train.index).isdisjoint(split.X_test.index)
    assert set(split.X_validation.index).isdisjoint(split.X_test.index)


def test_load_rejects_an_unexpected_file(tmp_path: Path) -> None:
    unexpected = tmp_path / "data.csv"
    unexpected.write_text("id,diagnosis,Unnamed: 32\n1,M,\n", encoding="utf-8")

    try:
        load_dataset(unexpected)
    except ValueError as error:
        assert "hash" in str(error).lower()
    else:
        raise AssertionError("Um dataset nao auditado deveria ser rejeitado.")

