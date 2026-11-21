from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

METRIC_KEYS = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "Confusion Matrix": "confusion_matrix",
}

def compute_metrics(y_true, y_pred, selected_metrics: List[str]) -> Dict[str, Any]:
    """Compute metrics based on what user selected in the UI."""
    selected = set(selected_metrics or [])

    results: Dict[str, Any] = {}

    # Treat FDIA label as positive class (1)
    if "Accuracy" in selected:
        results["Accuracy"] = float(accuracy_score(y_true, y_pred))
    if "Precision" in selected:
        results["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    if "Recall" in selected:
        results["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    if "F1" in selected:
        results["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if "Confusion Matrix" in selected:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        # format as [[tn, fp],[fn,tp]]
        results["Confusion Matrix"] = cm.tolist()

    return results
