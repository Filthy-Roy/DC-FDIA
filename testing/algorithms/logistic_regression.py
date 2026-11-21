from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .base import ModelResult

class LogisticRegressionAlgo:
    name = "Logistic Regression"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> ModelResult:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=self.random_state)),
        ])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return ModelResult(y_pred=np.asarray(y_pred), model=clf)
