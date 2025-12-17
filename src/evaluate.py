# src/evaluate.py

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


def evaluate(y_true, y_pred):
    return {
        "R2": round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAE": round(mean_absolute_error(y_true, y_pred), 2)
    }
