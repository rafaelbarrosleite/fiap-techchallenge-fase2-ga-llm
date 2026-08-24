import pandas as pd
from sklearn.dummy import DummyClassifier

from tech_challenge_fase2.evaluation import evaluate_classifier


def test_evaluation_reports_false_negatives_and_all_required_metrics() -> None:
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 1, 1])
    estimator = DummyClassifier(strategy="constant", constant=0).fit(X, y)

    result = evaluate_classifier(estimator, X, y)

    assert result["confusion_matrix"] == [[2, 0], [2, 0]]
    assert result["false_negatives_malignant"] == 2
    assert result["recall_malignant"] == 0.0
    assert {
        "accuracy",
        "precision_malignant",
        "recall_malignant",
        "f1_malignant",
        "roc_auc",
    }.issubset(result)

