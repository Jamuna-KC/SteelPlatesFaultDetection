"""Interactive Streamlit dashboard for Steel Plates Fault Detection."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
COMPARISON_PATH = PROJECT_DIR / "results" / "model_comparison.csv"
TARGET_COLUMN = "fault_type"
FEATURE_COLUMNS = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity",
    "Maximum_of_Luminosity", "Length_of_Conveyer", "TypeOfSteel_A300",
    "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
    "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
    "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


st.set_page_config(page_title="Steel Plate Fault Detection", page_icon="🔩", layout="wide")


@st.cache_resource
def load_model(model_file: str):
    """Load one trained model once per Streamlit session."""
    return joblib.load(MODEL_DIR / model_file)


@st.cache_data
def load_default_test_data() -> pd.DataFrame:
    """Load the exact held-out test data used during model evaluation."""
    return pd.read_csv(TEST_DATA_PATH)


def calculate_metrics(model, features: pd.DataFrame, labels: pd.Series) -> dict[str, float | str]:
    """Calculate the six assignment metrics on labelled test data."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    try:
        auc = roc_auc_score(
            labels,
            probabilities,
            labels=model.classes_,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auc = "Not available: upload data containing all fault classes."

    return {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": auc,
        "Precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "Recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "F1 Score": f1_score(labels, predictions, average="macro", zero_division=0),
        "MCC Score": matthews_corrcoef(labels, predictions),
    }


def main() -> None:
    """Render the interactive evaluation dashboard."""
    st.title("Machine Learning Assignment-2: Steel Plate Fault Detection")
    st.caption("Compare five machine-learning classifiers for seven steel-surface fault categories.")
    st.info(
        "**Project purpose:** This application identifies the surface defect category of a steel plate "
        "from its measurable surface and geometric properties.\n\n"
        "**Machine-learning task:** Multiclass classification.\n\n"
        "**Possible fault categories:** `Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, "
        "`Dirtiness`, `Bumps`, and `Other_Faults`.\n\n"
        "**How prediction works:** The selected model evaluates 27 measurements, including defect size, "
        "position, perimeter, luminosity, steel type, thickness, and shape-related indices.\n\n"
        "Upload a CSV following the expected format below, then choose a model and click **Evaluate test data**."
    )
    with st.expander("Expected CSV columns", expanded=False):
        st.caption(
            "All 27 feature columns are required and must be numeric. "
            "Include `fault_type` to display metrics and the confusion matrix."
        )
        schema = pd.DataFrame({
            "Column": FEATURE_COLUMNS + [TARGET_COLUMN],
            "Required": ["Yes"] * len(FEATURE_COLUMNS) + ["Optional - labels"],
        })
        st.dataframe(schema, hide_index=True, use_container_width=True, height=420)

    with st.sidebar:
        st.header("Evaluation controls")
        model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))
        uploaded_file = st.file_uploader("Upload test data (CSV)", type="csv")
        evaluate_clicked = st.button("Evaluate test data", type="primary", use_container_width=True)

    if not evaluate_clicked:
        st.subheader("Ready to evaluate")
        st.write(
            "Choose a classifier, optionally upload a test CSV, then click **Evaluate test data**. "
            "If no CSV is uploaded, the included held-out test set will be used."
        )
        return

    try:
        data = pd.read_csv(uploaded_file) if uploaded_file is not None else load_default_test_data()
    except Exception as error:
        st.error(f"The CSV could not be read: {error}")
        return

    model = load_model(MODEL_FILES[model_name])
    expected_features = list(model.feature_names_in_)
    missing_features = sorted(set(expected_features) - set(data.columns))
    if missing_features:
        st.error("The CSV is missing required feature columns: " + ", ".join(missing_features))
        return

    features = data[expected_features]
    predictions = model.predict(features)
    labels_available = TARGET_COLUMN in data.columns

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Selected model")
        st.markdown(
            f"""
            <div style="background: linear-gradient(90deg, #063970, #2596be);
                        border-left: 6px solid #0284c7; border-radius: 8px;
                        padding: 0.8rem 1rem; margin-bottom: 1rem;">
                <div style="color: #d4f0fa; font-size: 0.82rem; font-weight: 700;
                            letter-spacing: 0.06em; text-transform: uppercase;">
                    Active classifier
                </div>
                <div style="color: #ffffff; font-size: 1.35rem; font-weight: 700; margin-top: 0.2rem;">
                    {model_name}
                </div>
                <div style="color: #d4f0fa; margin-top: 0.25rem;">
                    Evaluated on <strong>{len(data):,}</strong> rows
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(data.head(10), use_container_width=True)
    with right:
        st.subheader("Prediction counts")
        st.dataframe(pd.Series(predictions, name="Predicted fault").value_counts().rename_axis("Fault type"))

    if not labels_available:
        st.warning(
            "Predictions are shown, but this CSV has no `fault_type` column. "
            "Upload labelled test data to display metrics and the confusion matrix."
        )
        output = data.copy()
        output["predicted_fault_type"] = predictions
        st.download_button(
            "Download predictions",
            output.to_csv(index=False).encode("utf-8"),
            file_name="steel_fault_predictions.csv",
            mime="text/csv",
        )
        return

    labels = data[TARGET_COLUMN]
    metrics = calculate_metrics(model, features, labels)
    st.subheader("Evaluation metrics")
    metric_columns = st.columns(6)
    for column, (metric_name, metric_value) in zip(metric_columns, metrics.items()):
        column.metric(metric_name, metric_value if isinstance(metric_value, str) else f"{metric_value:.4f}")

    matrix_column, report_column = st.columns(2)
    with matrix_column:
        st.subheader("Confusion matrix")
        class_names = list(model.classes_)
        matrix = confusion_matrix(labels, predictions, labels=class_names)
        figure, axis = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axis)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        axis.set_title(f"{model_name} - Confusion Matrix")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(figure)
        plt.close(figure)

    with report_column:
        st.subheader("Classification report")
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.subheader("Training comparison")
    st.caption("The table below was generated from the same stratified held-out test split.")
    st.dataframe(pd.read_csv(COMPARISON_PATH), use_container_width=True)


if __name__ == "__main__":
    main()
