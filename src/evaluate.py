"""Forecast evaluation helpers."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(actual, predicted) -> dict[str, float]:
	"""Calculate MAE and RMSE for equally sized prediction arrays."""
	actual_values = np.asarray(actual, dtype=float)
	predicted_values = np.asarray(predicted, dtype=float)
	if actual_values.size == 0 or actual_values.shape != predicted_values.shape:
		raise ValueError("Actual and predicted values must be non-empty and equal in shape")
	if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
		raise ValueError("Actual and predicted values must be finite")
	return {
		"mae": float(mean_absolute_error(actual_values, predicted_values)),
		"rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
	}


def evaluate_predictions(actual, predicted, baseline) -> dict[str, dict[str, float]]:
	"""Evaluate the model alongside a baseline forecast."""
	return {
		"baseline": regression_metrics(actual, baseline),
		"model": regression_metrics(actual, predicted),
	}
