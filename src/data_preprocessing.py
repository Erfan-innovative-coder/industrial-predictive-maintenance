import pandas as pd


def load_and_prepare_data(file_path):
    """Load the AI4I dataset and prepare features and target."""
    df = pd.read_csv(file_path)

    # Remove identifiers and failure-mode columns to prevent leakage
    columns_to_drop = [
        "UDI", "Product ID", "TWF", "HDF",
        "PWF", "OSF", "RNF"
    ]

    X = df.drop(columns=columns_to_drop + ["Machine failure"])
    y = df["Machine failure"]

    # Convert machine type into numerical features
    X = pd.get_dummies(X, columns=["Type"], dtype=int)

    return X, y
