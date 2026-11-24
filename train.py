import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_save_model():
    print("1. Loading dataset (Iris)...")
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"Dataset shape: {X.shape}, Classes: {list(iris.target_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("2. Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

    # Save model artifact in the script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(base_dir, "model.joblib")
    joblib.dump(model, model_filename)
    print(f"3. Model successfully saved to '{model_filename}'")

if __name__ == "__main__":
    train_and_save_model()
