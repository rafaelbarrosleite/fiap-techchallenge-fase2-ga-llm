"""Metricas de classificacao com foco na classe maligna."""

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y_true: pd.Series,
) -> dict[str, Any]:
    """Avalia um classificador ja ajustado sem expor linhas individuais."""

    y_pred = estimator.predict(X)
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("O baseline exige predict_proba para calcular ROC-AUC.")
    y_score = estimator.predict_proba(X)[:, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_malignant": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_malignant": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_malignant": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": matrix.tolist(),
        "true_negatives": true_negative,
        "false_positives": false_positive,
        "false_negatives_malignant": false_negative,
        "true_positives_malignant": true_positive,
    }

