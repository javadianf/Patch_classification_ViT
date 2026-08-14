# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Javadian. All rights reserved.

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(eval_pred):
    """
    Metric callback for the HuggingFace Trainer.

    Balanced accuracy is reported alongside plain accuracy because the test set
    keeps the original class imbalance, where a model that never predicts grade
    3 still scores well on accuracy. Balanced accuracy is the mean per-class
    recall, so it collapses if the minority grade is missed.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def evaluate_model(trainer, dataset):
    predictions = trainer.predict(dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    return y_pred, predictions.label_ids


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def plot_training_history(trainer, data_form):
    """Log the training and validation loss curves to MLflow."""
    log_history = trainer.state.log_history

    train_logs = [
        log for log in log_history if "loss" in log and "eval_loss" not in log
    ]
    eval_logs = [log for log in log_history if "eval_loss" in log]

    if not train_logs:
        print("No training logs found")
        return

    train_steps = [log.get("step", 0) for log in train_logs]
    train_loss = [log.get("loss", 0) for log in train_logs]

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label="Training Loss", marker="o")

    if eval_logs:
        eval_steps = [log.get("step", 0) for log in eval_logs]
        eval_loss = [log.get("eval_loss", 0) for log in eval_logs]
        plt.plot(
            eval_steps, eval_loss, label="Validation Loss", marker="s", color="orange"
        )

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss - {data_form}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    mlflow.log_figure(plt.gcf(), f"plots/training_history_{data_form}.png")
    plt.close()


def plot_confusion_matrix(
    y_true, y_pred, id2label, split_name="test", data_form="patches"
):
    cm = confusion_matrix(y_true, y_pred)
    label_names = [str(id2label[i]) for i in sorted(id2label.keys())]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title(f"Confusion Matrix - {split_name.capitalize()} Set ({data_form})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    mlflow.log_figure(plt.gcf(), f"plots/confusion_matrix_{split_name}_{data_form}.png")
    plt.close()


def save_metrics_to_mlflow(metrics, split_name="test"):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(metrics, f, indent=4)
        temp_path = f.name

    mlflow.log_artifact(temp_path, f"metrics/{split_name}")
    Path(temp_path).unlink()


def print_metrics(metrics, split_name="Test", data_form="patches"):
    print(f"{split_name.upper()} SET METRICS ({data_form})")
    print(f"Accuracy:          {metrics.get('accuracy', 0.0):.4f}")
    print(f"Balanced Accuracy: {metrics.get('balanced_accuracy', 0.0):.4f}")
    print(f"Precision:         {metrics.get('precision', 0.0):.4f}")
    print(f"Recall:            {metrics.get('recall', 0.0):.4f}")
    print(f"F1 Score:          {metrics.get('f1', 0.0):.4f}")


def print_classification_report(
    y_true, y_pred, id2label, split_name="Test", data_form="patches"
):
    print(f"{split_name} classification report ({data_form}):")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[str(id2label[i]) for i in sorted(id2label.keys())],
        )
    )


def evaluate_and_log(
    trainer,
    dataset,
    id2label,
    split_name="test",
    data_form="patches",
    save_metrics=False,
):
    """Run predictions on one split, print the report, and log everything to MLflow."""
    print(f"Evaluating on {split_name} set, data form {data_form}")

    predictions, labels = evaluate_model(trainer, dataset)
    metrics = calculate_metrics(labels, predictions)

    print_metrics(metrics, split_name, data_form)

    if save_metrics:
        save_metrics_to_mlflow(metrics, split_name)

    plot_confusion_matrix(labels, predictions, id2label, split_name, data_form)
    print_classification_report(labels, predictions, id2label, split_name, data_form)

    mlflow.log_metrics(
        {
            f"{split_name}_accuracy": metrics["accuracy"],
            f"{split_name}_balanced_accuracy": metrics["balanced_accuracy"],
            f"{split_name}_precision": metrics["precision"],
            f"{split_name}_recall": metrics["recall"],
            f"{split_name}_f1": metrics["f1"],
        }
    )

    return metrics
