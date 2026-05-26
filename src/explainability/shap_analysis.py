"""
shap_analysis.py

Generates SHAP feature importance plots for the trained XGBoost model.
"""

import argparse
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def load_model_and_data(model_path: str, data_path: str):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    return model, df


def run_shap_analysis(model, X: pd.DataFrame, output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)

    print("Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_summary_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Saved: {bar_path}")

    # Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.tight_layout()
    bee_path = os.path.join(output_dir, "shap_beeswarm.png")
    plt.savefig(bee_path, dpi=150)
    plt.close()
    print(f"Saved: {bee_path}")

    # Top 5 features
    mean_abs = np.abs(shap_values).mean(axis=0)
    top5 = pd.Series(mean_abs, index=X.columns).nlargest(5)
    print("\nTop 5 features by mean |SHAP|:")
    print(top5.to_string())

    return shap_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/saved/xgboost_model.pkl")
    parser.add_argument("--data", default="data/processed/X_test.csv")
    parser.add_argument("--output_dir", default="reports")
    args = parser.parse_args()

    model, df = load_model_and_data(args.model, args.data)
    run_shap_analysis(model, df, args.output_dir)
