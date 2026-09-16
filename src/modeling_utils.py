import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


TARGET = "DEFAULT_PAYMENT_NEXT_MONTH"

CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]


def load_data(path):
    df = pd.read_csv(path)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Leakage checks
    assert TARGET not in X.columns
    assert "HIGH_RISK" not in X.columns

    for col in X.columns:
        assert not X[col].equals(y), \
            f"Target leakage detected: {col}"

    return X, y


def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def build_preprocessor(X, scale_numeric=True):
    numerical_features = [
        col
        for col in X.columns
        if col not in CATEGORICAL_FEATURES
    ]

    if scale_numeric:
        numerical_transformer = StandardScaler()
    else:
        numerical_transformer = "passthrough"

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_transformer,
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )


def evaluate_model(
    model,
    X,
    y,
    model_name
):
    y_pred = model.predict(X)

    y_prob = model.predict_proba(X)[:, 1]

    return {
        "Model": model_name,

        "Accuracy": accuracy_score(
            y,
            y_pred
        ),

        "Precision": precision_score(
            y,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y,
            y_pred,
            zero_division=0
        ),

        "ROC_AUC": roc_auc_score(
            y,
            y_prob
        ),

        "PR_AUC": average_precision_score(
            y,
            y_prob
        )
    }