from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: List[str]


def _parse_bus_numbers(selected_buses: List[str]) -> List[int]:
    nums: List[int] = []
    for b in selected_buses:
        digits = "".join(ch for ch in b if ch.isdigit())
        if digits:
            nums.append(int(digits))
    return sorted(set(nums))


def select_columns_by_buses(df: pd.DataFrame, selected_buses: List[str]) -> List[str]:
    bus_nums = set(_parse_bus_numbers(selected_buses))

    candidate_cols = [c for c in df.columns if c != "Label"]
    numeric_cols = [
        c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])
    ]

    if not bus_nums:
        return numeric_cols

    def has_bus(col: str) -> bool:
        digits = "".join(ch for ch in col if ch.isdigit())
        return bool(digits) and int(digits) in bus_nums

    return [c for c in numeric_cols if has_bus(c)]


def drop_zero_only_columns(
    real_df: pd.DataFrame,
    fdia_df: pd.DataFrame,
    label_col: str = "Label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop columns (excluding label) that are ALL zeros in the authentic dataset.
    """

    feature_df = real_df.drop(columns=[label_col], errors="ignore")

    zero_only_cols = feature_df.columns[(feature_df == 0).all()].tolist()

    # Drop from both datasets
    real_df = real_df.drop(columns=zero_only_cols, errors="ignore")
    fdia_df = fdia_df.drop(columns=zero_only_cols, errors="ignore")

    # 🔍 DEBUG PRINTS (you can comment these later)
    print("Dropped zero-only columns:", len(zero_only_cols))
    print("Authentic dataset shape after drop:", real_df.shape)
    print("FDIA dataset shape after drop:", fdia_df.shape)

    return real_df, fdia_df


def load_and_label(real_csv_path: str, fdia_csv_path: str) -> pd.DataFrame:
    real_df = pd.read_csv(real_csv_path)
    fdia_df = pd.read_csv(fdia_csv_path)

    # Ensure labels
    if "Label" not in real_df.columns:
        real_df = real_df.copy()
        real_df["Label"] = 0

    fdia_df = fdia_df.copy()
    fdia_df["Label"] = 1

    # 🔹 DROP ZERO-ONLY COLUMNS
    real_df, fdia_df = drop_zero_only_columns(
        real_df, fdia_df, label_col="Label"
    )

    combined = pd.concat([real_df, fdia_df], ignore_index=True)
    return combined


def prepare_train_test(
    real_csv_path: str,
    fdia_csv_path: str,
    selected_buses: List[str],
    train_percent: float,
    test_percent: float,
    random_state: int = 42,
) -> PreparedData:

    if train_percent <= 0 or test_percent <= 0:
        raise ValueError("Train/Test percentages must be > 0.")
    if abs((train_percent + test_percent) - 100.0) > 1e-6:
        raise ValueError("Train% and Test% must sum to 100.")

    df = load_and_label(real_csv_path, fdia_csv_path)

    feature_cols = select_columns_by_buses(df, selected_buses)
    if not feature_cols:
        raise ValueError(
            "No feature columns matched after preprocessing."
        )

    X = df[feature_cols]
    y = df["Label"].astype(int)

    test_size = test_percent / 100.0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=feature_cols,
    )
