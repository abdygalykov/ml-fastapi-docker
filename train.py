import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay
)

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def train_and_track_model():
    """
    Train a RandomForest classifier on the Iris dataset,
    track experiments with MLflow (parameters, metrics, artifacts),
    and register the resulting model in the MLflow Model Registry.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(base_dir, "model.joblib")

    # 1. Load Dataset
    print("=" * 60)
    print("Step 1: Loading Iris Dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    target_names = [str(cls) for cls in iris.target_names]
    print(f"Dataset Shape: {X.shape} samples, {len(feature_names)} features")
    print(f"Target Classes: {target_names}")

    # Hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "criterion": "gini",
        "random_state": 42,
        "test_size": 0.2,
        "stratify": True,
        "algorithm": "RandomForestClassifier"
    }

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y if params["stratify"] else None
    )

    # 2. Train Model
    print("\n" + "=" * 60)
    print("Step 2: Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        criterion=params["criterion"],
        random_state=params["random_state"]
    )
    model.fit(X_train, y_train)

    # 3. Evaluate Model
    print("\n" + "=" * 60)
    print("Step 3: Evaluating Model Performance...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")

    metrics = {
        "accuracy": float(acc),
        "f1_score_macro": float(f1_macro),
        "f1_score_weighted": float(f1_weighted),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro)
    }

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Macro): {prec_macro:.4f}")
    print(f"Recall (Macro): {rec_macro:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

    # Save local model artifact
    joblib.dump(model, model_filename)
    print(f"Model successfully saved locally to: {model_filename}")

    # Generate Confusion Matrix Artifact
    cm_path = os.path.join(base_dir, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=target_names,
        cmap=plt.cm.Blues,
        ax=ax
    )
    plt.title("Iris Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Save classification report artifact
    report_path = os.path.join(base_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, y_pred, target_names=target_names))

    # 4. MLflow Experiment Tracking & Model Registry
    if MLFLOW_AVAILABLE:
        print("\n" + "=" * 60)
        print("Step 4: MLflow Experiment Tracking & Model Registration...")

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(base_dir, 'mlflow.db')}")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow Tracking URI: {tracking_uri}")

        experiment_name = "iris_classification_experiment"
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        print(f"Active Experiment: '{experiment_name}' (ID: {experiment.experiment_id})")

        model_name = "IrisClassifier"

        with mlflow.start_run(run_name="rf_hyperparams_v1") as run:
            run_id = run.info.run_id
            print(f"Started MLflow Run ID: {run_id}")

            # 4.1 Log Parameters
            print("Logging parameters to MLflow...")
            mlflow.log_params(params)

            # 4.2 Log Metrics
            print("Logging evaluation metrics to MLflow...")
            mlflow.log_metrics(metrics)

            # 4.3 Log Artifacts
            print("Logging artifacts to MLflow...")
            mlflow.log_artifact(model_filename, artifact_path="saved_model_file")
            mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
            mlflow.log_artifact(report_path, artifact_path="evaluation_reports")

            # 4.4 Log Model with skops_trusted_types & cloudpickle fallback
            input_example = X_test[:5]
            try:
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example,
                    registered_model_name=model_name,
                    skops_trusted_types=["sklearn.tree._tree.Tree"]
                )
            except (TypeError, Exception) as skops_err:
                print(f"Notice: skops validation fallback to cloudpickle ({skops_err})")
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example,
                    registered_model_name=model_name,
                    serialization_format="cloudpickle"
                )
            print(f"Model logged with artifact URI: {model_info.model_uri}")

            # 4.5 Register in MLflow Model Registry & Assign Metadata
            client = MlflowClient(tracking_uri=tracking_uri)
            try:
                model_versions = client.search_model_versions(f"name = '{model_name}'")
                if not model_versions:
                    reg = mlflow.register_model(model_info.model_uri, model_name)
                    latest_version = int(reg.version)
                else:
                    latest_version = max([int(mv.version) for mv in model_versions])

                # Update model description
                client.update_registered_model(
                    name=model_name,
                    description="RandomForest model for classifying Iris species (setosa, versicolor, virginica)."
                )

                # Update version description & tags
                client.update_model_version(
                    name=model_name,
                    version=str(latest_version),
                    description=f"Version {latest_version}: trained with n_estimators={params['n_estimators']}, accuracy={acc*100:.2f}%"
                )
                client.set_model_version_tag(
                    name=model_name,
                    version=str(latest_version),
                    key="stage",
                    value="Production"
                )
                print(f"Successfully registered model '{model_name}' Version: {latest_version}")
            except Exception as reg_err:
                print(f"Model registry metadata update notice: {reg_err}")

        print("=" * 60)
        print("MLflow Tracking & Registration Completed Successfully!")
        print("To view the MLflow UI, execute:")
        print(f"  mlflow ui --backend-store-uri {tracking_uri} --port 5000")
        print("=" * 60)
    else:
        print("\n[WARNING] MLflow is not installed in the current environment.")


if __name__ == "__main__":
    train_and_track_model()