from pathlib import Path

import joblib
import pandas as pd


# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "reports" / "tuned_random_forest.joblib"

# Load the saved model and its decision threshold
saved_artifact = joblib.load(MODEL_PATH)
model = saved_artifact["model"]
threshold = saved_artifact["threshold"]
feature_names = saved_artifact["feature_names"]


def predict_failure(machine_data):
    """Predict machine failure from sensor measurements."""
    input_df = pd.DataFrame([machine_data])

    # Ensure features match the model's training order
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    print("Input features:", input_df.to_dict(orient="records")[0])
    failure_probability = model.predict_proba(input_df)[0, 1]
    prediction = int(failure_probability >= threshold)

    return {
        "failure_probability": round(float(failure_probability), 4),
        "prediction": prediction,
        "result": "Failure predicted" if prediction else "No failure predicted",
    }



from data_preprocessing import load_and_prepare_data

if __name__ == "__main__":
    X, y = load_and_prepare_data(
        PROJECT_ROOT / "data" / "ai4i2020.csv"
    )

    # Select an actual failure record from the dataset
    failure_sample = X.loc[y == 1].iloc[[0]]
    actual_label = int(y.loc[failure_sample.index[0]])

    probability = model.predict_proba(
        failure_sample[feature_names]
    )[0, 1]

    prediction = int(probability >= threshold)

    print("Actual label:", actual_label)
    print("Failure probability:", round(float(probability), 4))
    print("Predicted label:", prediction)
    print("Threshold:", threshold)

