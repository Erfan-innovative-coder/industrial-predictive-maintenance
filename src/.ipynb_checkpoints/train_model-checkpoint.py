from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_preprocessing import load_and_prepare_data


# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
REPORTS_PATH = PROJECT_ROOT / "reports"

# Load and prepare data
X, y = load_and_prepare_data(DATA_PATH)

# Split data while preserving the failure rate
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

# Train and evaluate models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "PR-AUC (Average Precision)": average_precision_score(
            y_test, y_scores
        ),
    })

# Save model comparison results
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(REPORTS_PATH / "model_comparison.csv", index=False)

print(results_df.round(3).to_string(index=False))
