"""
evaluate_model.py
-----------------
Loads the saved best model and produces a detailed evaluation report
including a classification report, ROC-AUC, and all key plots.
Run after train_model.py has completed.
"""

import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from preprocessing import preprocess, ARTIFACTS_DIR

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_model():
    path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at {path}. Run train_model.py first."
        )
    return joblib.load(path)


def load_results():
    path = os.path.join(ARTIFACTS_DIR, "results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def plot_precision_recall(y_test, y_proba, model_name: str):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#e76f51", lw=2,
            label=f"PR Curve (AP={ap:.3f})")
    ax.axhline(y_test.mean(), linestyle="--", color="gray", lw=1,
               label=f"Baseline (prevalence={y_test.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "precision_recall_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved → {path}")


def plot_threshold_analysis(y_test, y_proba):
    """Show how precision/recall/F1 vary with classification threshold."""
    thresholds = np.linspace(0.1, 0.9, 80)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#4f86c6", lw=2)
    ax.plot(thresholds, recalls,    label="Recall",    color="#f4a261", lw=2)
    ax.plot(thresholds, f1s,        label="F1 Score",  color="#2a9d8f", lw=2)
    best_t = thresholds[np.argmax(f1s)]
    ax.axvline(best_t, linestyle="--", color="gray", lw=1.5,
               label=f"Best F1 threshold ≈ {best_t:.2f}")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "threshold_analysis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Saved → {path}")


def evaluate():
    print("=" * 60)
    print("  AI Patient Readmission — Model Evaluation")
    print("=" * 60)

    # Load model
    model = load_model()
    saved = load_results()
    model_name = saved.get("best_model", "Best Model")
    print(f"\n[evaluate] Loaded model: {model_name}")

    # Re-run preprocessing (test split is deterministic due to fixed seed)
    _, X_test, _, y_test, feature_names = preprocess(save_artifacts=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Classification report ──────────────────────────────────────────────────
    print("\n── Classification Report ──────────────────────────────────────")
    print(classification_report(y_test, y_pred,
                                 target_names=["No Readmit", "Readmit"]))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score : {auc:.4f}")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    print("\n── Confusion Matrix ────────────────────────────────────────────")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Readmit", "Readmit"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "best_model_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[evaluate] Saved → {path}")

    # ── Extra plots ────────────────────────────────────────────────────────────
    plot_precision_recall(y_test, y_proba, model_name)
    plot_threshold_analysis(y_test, y_proba)

    # ── Saved metrics summary ──────────────────────────────────────────────────
    if "metrics" in saved:
        print("\n── All-Model Metrics Summary ───────────────────────────────")
        header = f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} "
        header += f"{'F1':>8} {'AUC':>8}"
        print(header)
        print("-" * len(header))
        for name, m in saved["metrics"].items():
            marker = " ◄ best" if name == model_name else ""
            print(f"{name:<25} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
                  f"{m['recall']:>8.4f} {m['f1']:>8.4f} "
                  f"{m['roc_auc']:>8.4f}{marker}")

    print("\n[evaluate] Evaluation complete.")


if __name__ == "__main__":
    evaluate()
