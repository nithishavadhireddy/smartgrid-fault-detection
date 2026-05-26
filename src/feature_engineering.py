"""
feature_engineering.py

Adds rolling statistics and lag features to grid telemetry data.
"""

import numpy as np
import pandas as pd


SIGNAL_COLS = [
    "voltage_a", "voltage_b", "voltage_c",
    "current_a", "current_b", "current_c",
    "active_power", "reactive_power",
    "frequency", "power_factor",
]


def add_rolling_features(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    """Add rolling mean and std for each signal column."""
    windows = windows or [3, 5, 10]
    new_cols = {}
    for col in SIGNAL_COLS:
        if col not in df.columns:
            continue
        for w in windows:
            new_cols[f"{col}_roll_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            new_cols[f"{col}_roll_std_{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_lag_features(df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
    """Add lag features for each signal column."""
    lags = lags or [1, 2, 3]
    new_cols = {}
    for col in SIGNAL_COLS:
        if col not in df.columns:
            continue
        for lag in lags:
            new_cols[f"{col}_lag_{lag}"] = df[col].shift(lag).fillna(method="bfill")
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def drop_correlated_features(df: pd.DataFrame, threshold: float = 0.95,
                              target_col: str = "fault_label") -> pd.DataFrame:
    """Remove highly correlated feature columns (keep one from each correlated pair)."""
    feature_cols = [c for c in df.columns if c != target_col]
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        print(f"Dropping {len(to_drop)} highly correlated features: {to_drop[:5]}...")
    return df.drop(columns=to_drop)


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = add_rolling_features(df, config.get("rolling_windows", [3, 5, 10]))
    df = add_lag_features(df, config.get("lag_features", [1, 2, 3]))
    df = drop_correlated_features(
        df,
        threshold=config.get("drop_correlated_threshold", 0.95),
        target_col=config.get("target_col", "fault_label"),
    )
    return df
