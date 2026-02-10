from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from .preprocessing import clean_text

def train_and_evaluate(df):
    """Train and evaluate an NLP-based support ticket classification model."""

    # Apply text preprocessing
    X = df["text"].apply(clean_text)
    y = df["category"]

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Build ML pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save trained model
    joblib.dump(pipeline, "outputs/model.joblib")

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "classification_report": report,
        "categories": sorted(df["category"].unique())
    }
