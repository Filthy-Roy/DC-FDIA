from __future__ import annotations
from typing import Dict, Any, List

from .pre_processing import prepare_train_test
from .metrics import compute_metrics

from .algorithms.logistic_regression import LogisticRegressionAlgo
from .algorithms.svm import SVMAlgo
from .algorithms.random_forest import RandomForestAlgo
from .algorithms.mlp import MLPAlgo

_ALGO_REGISTRY = {
    "Logistic Regression": LogisticRegressionAlgo,
    "SVM": SVMAlgo,
    "Random Forest": RandomForestAlgo,
    "MLP": MLPAlgo,
}

def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point called by the UI.

    Expected config keys:
      - real_file: str
      - fdia_file: str
      - train_pct: float
      - test_pct: float
      - selected_algos: List[str]
      - selected_buses: List[str]
      - selected_metrics: List[str]
    """
    data = prepare_train_test(
        real_csv_path=config["real_file"],
        fdia_csv_path=config["fdia_file"],
        selected_buses=config.get("selected_buses", []),
        train_percent=float(config["train_pct"]),
        test_percent=float(config["test_pct"]),
    )

    algo_names: List[str] = config.get("selected_algos", [])
    if not algo_names:
        raise ValueError("Select at least one algorithm.")

    results: Dict[str, Any] = {
        "n_train": int(len(data.y_train)),
        "n_test": int(len(data.y_test)),
        "n_features": int(len(data.feature_columns)),
        "algorithms": {},
    }

    for algo_name in algo_names:
        if algo_name not in _ALGO_REGISTRY:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

        algo = _ALGO_REGISTRY[algo_name]()  # default constructor
        model_result = algo.run(data.X_train, data.y_train, data.X_test)
        metrics = compute_metrics(data.y_test, model_result.y_pred, config.get("selected_metrics", []))

        results["algorithms"][algo_name] = {
            "metrics": metrics,
        }

    return results
