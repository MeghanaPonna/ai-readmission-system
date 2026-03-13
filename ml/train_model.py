"""
train_model.py
--------------
Trains Logistic Regression, Random Forest, and XGBoost classifiers,
compares their performance, and saves the best model + artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve
)

from preprocessing import preprocess, ARTIFACTS_DIR

# ── Config ─────────────────────────────────────────────────────────────────────
PLOTS_DIR    = os.path.join(os.path.dirname(__file__), "plots")
RANDOM_STATE = 42


# ── Model definitions ──────────────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5,   # handles class imbalance
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


# ── Evaluation helper ──────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),                   4),
    }, y_pred, y_proba


# ── Plotting utilities ─────────────────────────────────────────────────────────
def plot_model_comparison(results: dict):
    """Bar chart comparing all models across all metrics."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models  = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4f86c6", "#f4a261", "#2a9d8f"]
    for i, (model_name, color) in enumerate(zip(models, colors)):
        vals = [results[model_name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[train] Saved → {path}")


def plot_confusion_matrix(model, name, X_test, y_test):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Readmit", "Readmit"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"cm_{name.replace(' ', '_').lower()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[train] Saved → {path}")


def plot_roc_curves(roc_data: dict):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    colors = ["#4f86c6", "#f4a261", "#2a9d8f"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[train] Saved → {path}")


def plot_feature_importance(model, feature_names: list, model_name: str,
                             top_n: int = 20):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(indices)),
            importances[indices],
            color="#2a9d8f", alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                 fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR,
                        f"feature_importance_{model_name.replace(' ','_').lower()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[train] Saved → {path}")


# ── Main training loop ─────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  AI Patient Readmission — Model Training")
    print("=" * 60)

    # 1. Preprocess
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        save_artifacts=True
    )

    # 2. Train all models
    models   = get_models()
    results  = {}
    roc_data = {}
    trained  = {}

    for name, model in models.items():
        print(f"\n[train] Training {name} …")
        model.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained[name] = model

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, metrics["roc_auc"])

        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

        plot_confusion_matrix(model, name, X_test, y_test)
        plot_feature_importance(model, feature_names, name)

    # 3. Select best model by ROC-AUC
    best_name  = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = trained[best_name]
    print(f"\n[train] Best model: {best_name} "
          f"(ROC-AUC = {results[best_name]['roc_auc']:.4f})")

    # 4. Save artefacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"[train] Model saved → {model_path}")

    results_path = os.path.join(ARTIFACTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump({"best_model": best_name, "metrics": results}, f, indent=2)
    print(f"[train] Results saved → {results_path}")

    # 5. Generate plots
    plot_model_comparison(results)
    plot_roc_curves(roc_data)

    print("\n[train] Done!")
    return best_model, results, feature_names


if __name__ == "__main__":
    train()
