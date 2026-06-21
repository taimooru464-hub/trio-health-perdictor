"""
train.py
--------
Main training script for Health Trio Predictor.

Trains three models:
  1. KNN (from scratch)          → Diabetes Prediction
  2. Decision Tree (from scratch) → Heart Disease Prediction
  3. Naive Bayes (from scratch)  → Student Stress Prediction

Each model is:
  - Trained from scratch AND compared to sklearn baseline
  - Tracked with MLflow (params, metrics, artifacts)
  - Evaluated with accuracy, F1, confusion matrix

Usage:
    python train.py

Requirements:
    pip install numpy pandas scikit-learn mlflow matplotlib pyyaml joblib
"""

import os
import sys
import pickle
import random
import numpy as np
import yaml
import mlflow
import mlflow.sklearn

# Sklearn baselines for comparison
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.naive_bayes import GaussianNB

# Our custom models
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_diabetes, load_heart, load_stress
from src.models.knn_scratch import KNNClassifier
from src.models.decision_tree_scratch import DecisionTreeClassifier
from src.models.naive_bayes_scratch import GaussianNaiveBayes
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_comparison_bar


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seeds(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Save model ────────────────────────────────────────────────────────────────

def save_model(model, name: str, folder: str = "models"):
    """Pickle a trained model to disk."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved → {path}")
    return path


# ── Train: Diabetes (KNN) ──────────────────────────────────────────────────────

def train_diabetes(config: dict):
    """Train KNN from scratch + sklearn KNN on Diabetes dataset."""
    print("\n" + "█"*50)
    print("  DIABETES PREDICTION — KNN from Scratch")
    print("█"*50)

    X_train, X_test, y_train, y_test, scaler = load_diabetes(config)
    k = config["diabetes"]["k"]
    seed = config["general"]["random_seed"]

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # ── Custom KNN ──
    with mlflow.start_run(run_name="Diabetes_CustomKNN"):
        model = KNNClassifier(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, "Custom KNN — Diabetes")

        mlflow.log_param("model", "KNN_scratch")
        mlflow.log_param("k", k)
        mlflow.log_param("random_seed", seed)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_score", metrics["f1"])

        cm_path = "outputs/diabetes_custom_cm.png"
        plot_confusion_matrix(y_test, y_pred, "Diabetes — Custom KNN",
                              cm_path, class_names=["No Diabetes", "Diabetes"])
        mlflow.log_artifact(cm_path)

        custom_acc = metrics["accuracy"]
        custom_f1 = metrics["f1"]

    # ── Sklearn KNN (baseline) ──
    with mlflow.start_run(run_name="Diabetes_SklearnKNN"):
        sk_model = KNeighborsClassifier(n_neighbors=k)
        sk_model.fit(X_train, y_train)
        sk_pred = sk_model.predict(X_test)

        sk_metrics = compute_metrics(sk_pred, y_test, "Sklearn KNN — Diabetes")

        mlflow.log_param("model", "KNN_sklearn")
        mlflow.log_param("k", k)
        mlflow.log_metric("accuracy", sk_metrics["accuracy"])
        mlflow.log_metric("f1_score", sk_metrics["f1"])

    # ── Comparison chart ──
    plot_comparison_bar(
        {"Custom KNN": custom_acc, "Sklearn KNN": sk_metrics["accuracy"]},
        metric="accuracy",
        title="Diabetes — Custom vs Sklearn KNN",
        save_path="outputs/diabetes_comparison.png"
    )

    # Save models
    save_model(model, "diabetes_knn")
    save_model(scaler, "diabetes_scaler")
    print("[DONE] Diabetes training complete.\n")


# ── Train: Heart Disease (Decision Tree) ───────────────────────────────────────

def train_heart(config: dict):
    """Train Decision Tree from scratch + sklearn DT on Heart Disease dataset."""
    print("\n" + "█"*50)
    print("  HEART DISEASE PREDICTION — Decision Tree")
    print("█"*50)

    X_train, X_test, y_train, y_test, scaler = load_heart(config)
    max_depth = config["heart"]["max_depth"]
    seed = config["general"]["random_seed"]

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # ── Custom Decision Tree ──
    with mlflow.start_run(run_name="Heart_CustomDT"):
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, "Custom Decision Tree — Heart")

        mlflow.log_param("model", "DecisionTree_scratch")
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_seed", seed)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_score", metrics["f1"])

        cm_path = "outputs/heart_custom_cm.png"
        plot_confusion_matrix(y_test, y_pred, "Heart Disease — Custom DT",
                              cm_path, class_names=["No Disease", "Disease"])
        mlflow.log_artifact(cm_path)

        custom_acc = metrics["accuracy"]

    # ── Sklearn Decision Tree (baseline) ──
    with mlflow.start_run(run_name="Heart_SklearnDT"):
        sk_model = SklearnDT(max_depth=max_depth, random_state=seed)
        sk_model.fit(X_train, y_train)
        sk_pred = sk_model.predict(X_test)

        sk_metrics = compute_metrics(sk_pred, y_test, "Sklearn DT — Heart")

        mlflow.log_param("model", "DecisionTree_sklearn")
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", sk_metrics["accuracy"])
        mlflow.log_metric("f1_score", sk_metrics["f1"])

    # ── Comparison chart ──
    plot_comparison_bar(
        {"Custom DT": custom_acc, "Sklearn DT": sk_metrics["accuracy"]},
        metric="accuracy",
        title="Heart Disease — Custom vs Sklearn DT",
        save_path="outputs/heart_comparison.png"
    )

    save_model(model, "heart_dt")
    save_model(scaler, "heart_scaler")
    print("[DONE] Heart Disease training complete.\n")


# ── Train: Student Stress (Naive Bayes) ────────────────────────────────────────

def train_stress(config: dict):
    """Train Naive Bayes from scratch + sklearn NB on Student Stress dataset."""
    print("\n" + "█"*50)
    print("  STUDENT STRESS PREDICTION — Naive Bayes")
    print("█"*50)

    X_train, X_test, y_train, y_test, scaler = load_stress(config)
    seed = config["general"]["random_seed"]

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # ── Custom Naive Bayes ──
    with mlflow.start_run(run_name="Stress_CustomNB"):
        model = GaussianNaiveBayes()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, "Custom Naive Bayes — Stress")

        mlflow.log_param("model", "NaiveBayes_scratch")
        mlflow.log_param("random_seed", seed)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_score", metrics["f1"])

        cm_path = "outputs/stress_custom_cm.png"
        plot_confusion_matrix(y_test, y_pred, "Student Stress — Custom NB",
                              cm_path, class_names=["Low", "Medium", "High"])
        mlflow.log_artifact(cm_path)

        custom_acc = metrics["accuracy"]

    # ── Sklearn Naive Bayes (baseline) ──
    with mlflow.start_run(run_name="Stress_SklearnNB"):
        sk_model = GaussianNB()
        sk_model.fit(X_train, y_train)
        sk_pred = sk_model.predict(X_test)

        sk_metrics = compute_metrics(sk_pred, y_test, "Sklearn NB — Stress")

        mlflow.log_param("model", "NaiveBayes_sklearn")
        mlflow.log_metric("accuracy", sk_metrics["accuracy"])
        mlflow.log_metric("f1_score", sk_metrics["f1"])

    # ── Comparison chart ──
    plot_comparison_bar(
        {"Custom NB": custom_acc, "Sklearn NB": sk_metrics["accuracy"]},
        metric="accuracy",
        title="Student Stress — Custom vs Sklearn NB",
        save_path="outputs/stress_comparison.png"
    )

    save_model(model, "stress_nb")
    save_model(scaler, "stress_scaler")
    print("[DONE] Student Stress training complete.\n")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = load_config("config.yaml")
    set_seeds(config["general"]["random_seed"])

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    print("\n" + "="*50)
    print("   HEALTH TRIO PREDICTOR — Training Pipeline")
    print("="*50)

    train_diabetes(config)
    train_heart(config)
    train_stress(config)

    print("\n" + "="*50)
    print("   ALL MODELS TRAINED SUCCESSFULLY!")
    print("   Run: mlflow ui   →   open localhost:5000")
    print("="*50 + "\n")
