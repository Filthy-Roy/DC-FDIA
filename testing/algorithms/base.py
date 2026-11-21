from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Any, Dict
import numpy as np
import pandas as pd

@dataclass
class ModelResult:
    y_pred: np.ndarray
    model: Any

class Algorithm(Protocol):
    name: str
    def run(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> ModelResult:
        ...
