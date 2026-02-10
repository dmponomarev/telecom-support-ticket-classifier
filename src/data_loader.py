import pandas as pd

def load_data(path="data/support_tickets_clean.csv"):
    """
    Loads and validates the customer support ticket dataset.
    """
    df = pd.read_csv(path, encoding="latin1")

    # Required structure checks
    required_columns = {"text", "category"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")

    # Basic cleaning
    df = df.dropna(subset=["text", "category"])
    df["category"] = df["category"].str.strip().str.lower()

    # Category validation
    expected_categories = {"billing", "network", "device", "contract", "other"}
    actual_categories = set(df["category"].unique())

    if actual_categories != expected_categories:
        raise ValueError(
            f"Incorrect categories detected. "
            f"Expected: {expected_categories}, got: {actual_categories}"
        )

    # Dataset size validation
    expected_size = 400
    if len(df) != expected_size:
        raise ValueError(
            f"Expected {expected_size} records, got {len(df)}"
        )
    return df
