import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data():
    """Load training and test data"""
    with open("files/grading/x_train.pkl", "rb") as file:
        x_train = pickle.load(file)

    with open("files/grading/y_train.pkl", "rb") as file:
        y_train = pickle.load(file)

    with open("files/grading/x_test.pkl", "rb") as file:
        x_test = pickle.load(file)

    with open("files/grading/y_test.pkl", "rb") as file:
        y_test = pickle.load(file)

    return x_train, y_train, x_test, y_test


def create_pipeline(x_train):
    """Create the pipeline with required components"""
    # Identify categorical and numerical columns
    categorical_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = x_train.select_dtypes(include=["number"]).columns.tolist()

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    # Create pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_classif)),
            ("scaler", MinMaxScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    return pipeline


def train_model(x_train, y_train):
    """Train the model with GridSearchCV"""
    pipeline = create_pipeline(x_train)

    param_grid = {
        "selector__k": [10, 15, 20],
        "classifier__C": [0.1, 1, 10],
        "classifier__solver": ["liblinear", "lbfgs"],
    }

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1
    )

    grid_search.fit(x_train, y_train)

    return grid_search


def save_model(model):
    """Save the trained model"""
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pickle.dump(model, file)


def calculate_metrics(model, x_train, y_train, x_test, y_test):
    """Calculate and save metrics"""
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics = []

    # Train metrics
    metrics.append(
        {
            "type": "metrics",
            "dataset": "train",
            "precision": float(precision_score(y_train, y_train_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
            "recall": float(recall_score(y_train, y_train_pred)),
            "f1_score": float(f1_score(y_train, y_train_pred)),
        }
    )

    # Test metrics
    metrics.append(
        {
            "type": "metrics",
            "dataset": "test",
            "precision": float(precision_score(y_test, y_test_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
            "recall": float(recall_score(y_test, y_test_pred)),
            "f1_score": float(f1_score(y_test, y_test_pred)),
        }
    )

    # Confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    metrics.append(
        {
            "type": "cm_matrix",
            "dataset": "train",
            "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
            "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])},
        }
    )

    cm_test = confusion_matrix(y_test, y_test_pred)
    metrics.append(
        {
            "type": "cm_matrix",
            "dataset": "test",
            "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
            "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])},
        }
    )

    return metrics


def save_metrics(metrics):
    """Save metrics to JSON file"""
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        for metric in metrics:
            file.write(json.dumps(metric) + "\n")


def main():
    """Main function"""
    # Load data
    x_train, y_train, x_test, y_test = load_data()

    # Train model
    model = train_model(x_train, y_train)

    # Save model
    save_model(model)

    # Calculate and save metrics
    metrics = calculate_metrics(model, x_train, y_train, x_test, y_test)
    save_metrics(metrics)

    print("Model trained and saved successfully!")
    print(f"Best score: {model.best_score_:.4f}")
    print(f"Best params: {model.best_params_}")


if __name__ == "__main__":
    main()
