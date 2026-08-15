"""Train and evaluate the five required classifiers for Steel Plates Faults."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Data" / "steel_plates_faults_clean.csv"
MODELS_DIR = PROJECT_DIR / "model"
RESULTS_DIR = PROJECT_DIR / "results"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
RANDOM_STATE = 42
TARGET_COLUMN = "fault_type"


def calculate_metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Return all required metrics using macro averages for the seven classes."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities, labels=model.classes_, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "F1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }


def main() -> None:
    """Train, evaluate, and save all requested classification models."""
    data = pd.read_csv(DATA_PATH)
    x = data.drop(columns=TARGET_COLUMN)
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(TEST_DATA_PATH, index=False)

    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB()),
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=400,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    results: list[dict[str, str | float]] = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        metrics = calculate_metrics(model, x_test, y_test)
        filename = model_name.lower().replace("-", "_").replace(" ", "_") + ".joblib"
        joblib.dump(model, MODELS_DIR / filename)
        results.append({"ML Model Name": model_name, **metrics})

    metrics_table = pd.DataFrame(results)
    metrics_table.iloc[:, 1:] = metrics_table.iloc[:, 1:].round(4)
    metrics_table.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()
