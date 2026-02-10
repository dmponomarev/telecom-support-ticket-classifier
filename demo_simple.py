"""
Simple demo script for customer support ticket classification.
Works in any environment: terminal, PyCharm, Jupyter Notebook.
"""

import sys
import os

# Add project root to Python path to enable imports from src/
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import joblib
from src.preprocessing import clean_text

def main():
    print(" Loading model...")

    # Check if trained model exists
    model_path = os.path.join(project_root, "outputs", "model.joblib")
    if not os.path.exists(model_path):
        print(f" Error: Model file not found at {model_path}")
        print(" Please run: python main.py first")
        return

    # Load trained pipeline (includes TF-IDF vectorizer + Logistic Regression)
    pipeline = joblib.load(model_path)
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["clf"]

    print(" Model loaded successfully!\n")

    # Classification function
    def classify_ticket(text):
        """Preprocess text and predict category."""
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        return model.predict(vec)[0]

    # Show examples
    print(" Example predictions:")
    examples = [
        "My internet is very slow",
        "Gebühren auf meiner Rechnung sind zu hoch",
        "SIM-Karte funktioniert nicht",
        "Ich möchte meinen Vertrag kündigen"
    ]
    for ex in examples:
        result = classify_ticket(ex)
        print(f"   Ticket: '{ex}' → Category: {result}")

    # Main interactive loop
    print("\n Enter a support ticket (English or German) to classify:")

    while True:
        try:
            user_input = input("\n> ")
            if not user_input.strip():
                print("️ Empty input. Please try again.")
                continue

            # Predict and display result
            result = classify_ticket(user_input)
            print(f"\n Predicted category: {result}")

            # Ask to continue
            while True:
                choice = input("\n Classify another ticket? (y/n): ").strip().lower()
                if choice in ('y', 'yes'):
                    break
                elif choice in ('n', 'no'):
                    print("\n Goodbye!")
                    return
                else:
                    print(" Please enter 'y' or 'n'")

        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            return

if __name__ == "__main__":
    main()