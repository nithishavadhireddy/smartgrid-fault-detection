# smartgrid-fault-detection

Predictive fault detection for smart grids using classical ML models and a Deep Neural Network,
with SHAP-based explainability. Built on a dataset of 1,000 samples across 24 engineered features
derived from grid telemetry signals.

## Problem

Modern smart grids experience instability due to fluctuating renewable energy sources and varying
loads. Early fault detection reduces outage duration and prevents cascading failures. This project
frames the problem as a binary classification task: predict whether a grid segment will experience
a fault within the next monitoring window.

## Models Evaluated

| Model               | Accuracy | F1 (fault class) |
|---------------------|----------|------------------|
| Logistic Regression | 0.51     | 0.48             |
| Random Forest       | 0.61     | 0.59             |
| Gradient Boosting   | 0.63     | 0.61             |
| XGBoost             | 0.65     | 0.63             |
| Deep Neural Network | 0.54     | 0.52             |

XGBoost performed best overall. The DNN showed signs of overfitting on this dataset size and
would likely benefit from more data.

## Explainability

SHAP summary plots identify `voltage_deviation`, `load_variance`, and `frequency_delta` as the
top predictors across tree-based models.

## Project Structure

```
smartgrid-fault-detection/
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── models/
│   │   ├── xgboost_model.py
│   │   └── dnn_model.py
│   └── explainability/
│       └── shap_analysis.py
├── config/
│   └── model_config.yaml
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# Train all models
python src/train.py --config config/model_config.yaml

# Run SHAP analysis on XGBoost
python src/explainability/shap_analysis.py --model xgboost
```

## Dataset

The dataset contains 24 features extracted from smart grid telemetry:
voltage readings, current measurements, load metrics, frequency deviations,
harmonic distortions, and derived rolling statistics. Target label is binary
(0 = normal, 1 = fault).
