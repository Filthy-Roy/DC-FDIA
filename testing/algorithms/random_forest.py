from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base import ModelResult

class RandomForestAlgo:
    name = "Random Forest"

    def __init__(self, random_state: int = 42, n_estimators: int = 300):
        self.random_state = random_state
        self.n_estimators = n_estimators

    def run(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> ModelResult:
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return ModelResult(y_pred=np.asarray(y_pred), model=clf)
