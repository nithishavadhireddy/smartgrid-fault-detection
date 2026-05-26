"""
data_preprocessing.py

Loads raw grid telemetry, handles outliers, scales features.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path)
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols: list, threshold: float = 1.5) -> pd.DataFrame:
    """Remove rows where any feature column is outside IQR bounds."""
    mask = pd.Series([True] * len(df), index=df.index)
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        iqr = Q3 - Q1
        lower = Q1 - threshold * iqr
        upper = Q3 + threshold * iqr
        mask &= df[col].between(lower, upper)
    removed = (~mask).sum()
    if removed:
        logger.info("Removed %d outlier rows via IQR method", removed)
    return df[mask].reset_index(drop=True)


def scale_features(X_train: np.ndarray, X_test: np.ndarray,
                   method: str = "standard", save_path: str = None):
    """Fit scaler on train, transform both splits."""
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if save_path:
        joblib.dump(scaler, save_path)
        logger.info("Scaler saved to %s", save_path)
    return X_train_scaled, X_test_scaled, scaler


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with column median, drop rows still missing after that."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < before:
        logger.warning("Dropped %d rows with remaining NaNs", before - len(df))
    return df
