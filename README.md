# Steel Plate Fault Detection

## Problem Statement

Manufacturing quality-control teams need to identify surface defects on steel plates quickly and consistently. This project uses supervised machine learning to classify a steel plate into one of seven defect categories from 27 numeric surface, geometric, luminosity, and steel-related measurements.

The project implements and compares five classification models, then provides an interactive Streamlit app for evaluating a selected model on labelled test data.

## Dataset Description

The project uses the **Steel Plates Faults** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/198/steel%2Bplates%2Bfaults).

- **Task:** Multiclass classification
- **Instances:** 1,941 steel-plate records
- **Features:** 27 numeric predictor features
- **Target:** `fault_type`
- **Classes:** `Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, and `Other_Faults`
- **Missing values:** None in the supplied source data

The original `Faults.NNA` file stores the target as seven one-hot columns. `prepare_data.py` converts those columns into the single readable `fault_type` target used by the project.

## GitHub Repository Link
`https://github.com/Jamuna-KC/SteelPlatesFaultDetection`

## Models Used and Evaluation

All five models were trained using the same 75:25 stratified train-test split (`random_state=42`). Precision, recall, F1, and AUC are macro averaged to give every fault class equal weight despite the class imbalance.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7222 | 0.9386 | 0.7632 | 0.7113 | 0.7316 | 0.6404 |
| Decision Tree | 0.6996 | 0.8254 | 0.7017 | 0.7078 | 0.6965 | 0.6137 |
| K-Nearest Neighbors | 0.7593 | 0.9283 | 0.7498 | 0.7622 | 0.7540 | 0.6912 |
| Gaussian Naive Bayes | 0.6091 | 0.9225 | 0.6406 | 0.7025 | 0.6323 | 0.5456 |
| Random Forest (Ensemble) | **0.7901** | **0.9635** | **0.8225** | **0.8212** | **0.8183** | **0.7343** |

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong linear baseline with high AUC, but it is less able to model complex non-linear interactions among the plate measurements. |
| Decision Tree | Produces interpretable decision rules and reasonably balanced class performance, but a single tree is more sensitive to variations in the training data. |
| K-Nearest Neighbors | Performs well after feature scaling because nearby defect patterns are informative; its macro recall is higher than Logistic Regression and Decision Tree. |
| Gaussian Naive Bayes | Achieves a good AUC but the lowest accuracy, F1, and MCC. The feature-independence assumption is restrictive for correlated geometric and luminosity measurements. |
| Random Forest (Ensemble) | Gives the best score for every reported metric. Combining many decision trees captures non-linear feature interactions while reducing the instability of a single decision tree. |
| Overall Winner | **Random Forest** is the best-performing model for this dataset, with 0.7901 accuracy, 0.9635 macro AUC, 0.8183 macro F1, and 0.7343 MCC. |

## Streamlit Application Features

The deployed app includes the assignment-required features:

- CSV test-data upload
- Model-selection dropdown
- Evaluate test data button
- Accuracy, AUC, precision, recall, F1, and MCC display
- Confusion matrix and classification report
- Prediction download for an uploaded CSV without labels

## Project Structure

```text
SteelPlatesFaultDetection/
├── app.py
├── prepare_data.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── Data/
│   ├── Faults.NNA
│   ├── Faults27x7_var
│   └── steel_plates_faults_clean.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── k_nearest_neighbors.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── random_forest.joblib
└── results/
    └── model_comparison.csv
```

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prepare_data.py
python train_models.py
streamlit run app.py
```
## Live Streamlit Apllication link:
https://steelplatesfaultdetection-wl23nb4uyaeu5mkv9wqobc.streamlit.app/

