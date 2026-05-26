"""
xgboost_model.py

XGBoost classifier for smart grid fault detection.
"""

import logging
from typing import Tuple

import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def build_xgboost(config: dict) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", 6),
        learning_rate=config.get("learning_rate", 0.05),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        min_child_weight=config.get("min_child_weight", 3),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def train_xgboost(X_train, y_train, X_val, y_val, config: dict) -> XGBClassifier:
    model = build_xgboost(config)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=config.get("early_stopping_rounds", 20),
        verbose=False,
    )
    logger.info("XGBoost best iteration: %d", model.best_iteration)
    return model


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("XGBoost AUC: %.4f", auc)
    logger.info("\n%s", classification_report(y_test, y_pred))
    return {"auc": auc, "report": report}


def cross_validate(X, y, config: dict, cv: int = 5) -> np.ndarray:
    model = build_xgboost(config)
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_obj, scoring="roc_auc", n_jobs=-1)
    logger.info("XGBoost CV AUC: %.4f +/- %.4f", scores.mean(), scores.std())
    return scores


def save_model(model, path: str):
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)
