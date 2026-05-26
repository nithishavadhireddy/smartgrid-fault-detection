"""
train.py

Entry point: loads data, engineers features, trains all models, prints results.

Usage:
    python src/train.py --config config/model_config.yaml
"""

import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data, handle_missing, remove_outliers_iqr, scale_features
from feature_engineering import engineer_features
from models.xgboost_model import train_xgboost, evaluate as eval_xgb, save_model
from models.dnn_model import train_dnn, evaluate as eval_dnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load & preprocess
    df = load_data(config["data"]["raw_path"])
    df = handle_missing(df)

    target = config["data"]["target_col"]
    feature_cols = [c for c in df.columns if c != target]

    df = remove_outliers_iqr(df, feature_cols, config["preprocessing"]["outlier_threshold"])
    df = engineer_features(df, config["feature_engineering"])

    X = df.drop(columns=[target]).values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"],
        stratify=y, random_state=config["data"]["random_state"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    _, X_val_s, _ = scale_features(X_train, X_val)

    os.makedirs(config["output"]["model_dir"], exist_ok=True)

    # XGBoost
    logger.info("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, config["xgboost"])
    xgb_results = eval_xgb(xgb_model, X_test, y_test)
    save_model(xgb_model, os.path.join(config["output"]["model_dir"], "xgboost_model.pkl"))

    # DNN
    logger.info("Training DNN...")
    dnn_model = train_dnn(X_train_s, y_train, X_val_s, y_val, config["dnn"])
    dnn_results = eval_dnn(dnn_model, X_test_s, y_test)
    dnn_model.save(os.path.join(config["output"]["model_dir"], "dnn_model.h5"))

    print("\n=== Results Summary ===")
    print(f"XGBoost AUC: {xgb_results['auc']:.4f}")
    print(f"DNN AUC:     {dnn_results['auc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    args = parser.parse_args()
    main(args.config)
