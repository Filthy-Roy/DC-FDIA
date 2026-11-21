from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from .base import ModelResult

class MLPAlgo:
    name = "MLP"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> ModelResult:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=300,
                random_state=self.random_state
            )),
        ])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return ModelResult(y_pred=np.asarray(y_pred), model=clf)
