from pathlib import Path
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from data_preprocessing import load_and_prepare_data


# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
REPORTS_PATH = PROJECT_ROOT / "reports"

# Load data and create train/test sets
X, y = load_and_prepare_data(DATA_PATH)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create a validation set for threshold selection
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full,
)

# Train a model for threshold selection
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)

# Select the threshold that maximizes precision with recall >= 80%
val_scores = rf_model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_scores)

valid_indices = [
    i for i, threshold in enumerate(thresholds)
    if recall[i] >= 0.80
]

if not valid_indices:
    raise ValueError("No threshold achieved the required validation recall.")

best_index = max(valid_indices, key=lambda i: precision[i])
selected_threshold = thresholds[best_index]

# Retrain on the full training set
rf_model.fit(X_train_full, y_train_full)

# Evaluate on the untouched test set
test_scores = rf_model.predict_proba(X_test)[:, 1]
y_pred = (test_scores >= selected_threshold).astype(int)

print(f"Selected threshold: {selected_threshold:.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-score: {f1_score(y_test, y_pred):.3f}")
print(f"PR-AUC: {average_precision_score(y_test, test_scores):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["No Failure", "Failure"],
    zero_division=0,
))

# Save test predictions and evaluation results
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

pd.DataFrame({
    "Actual Failure": y_test,
    "Predicted Failure": y_pred,
    "Failure Probability": test_scores,
}).to_csv(REPORTS_PATH / "tuned_rf_predictions.csv", index=True)

pd.DataFrame([{
    "Threshold": selected_threshold,
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred),
    "PR-AUC": average_precision_score(y_test, test_scores),
}]).to_csv(REPORTS_PATH / "tuned_rf_metrics.csv", index=False)


# Save the trained model and its decision threshold
MODEL_PATH = PROJECT_ROOT / "reports" / "tuned_random_forest.joblib"

joblib.dump(
    {
        "model": rf_model,
        "threshold": selected_threshold,
        "feature_names": list(X.columns),
    },
    MODEL_PATH,
)

print(f"\nModel saved to: {MODEL_PATH}")
