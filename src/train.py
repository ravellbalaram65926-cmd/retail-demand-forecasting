"""Run the end-to-end retail demand forecasting pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

from src.data_cleaning import clean_sales
from src.data_ingestion import ingest_competition_data
from src.evaluate import evaluate_predictions
from src.feature_engineering import create_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def split_by_date(data: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Hold out the latest distinct dates, preserving chronological order."""
	unique_dates = pd.Index(data["date"].dropna().unique()).sort_values()
	if test_days < 1 or len(unique_dates) <= test_days:
		raise ValueError("test_days must be positive and leave at least one training date")
	cutoff = unique_dates[-test_days]
	training = data[data["date"] < cutoff].copy()
	testing = data[data["date"] >= cutoff].copy()
	if training.empty or testing.empty:
		raise ValueError("Date split produced an empty training or testing period")
	return training, testing


def _project_path(value: str) -> Path:
	path = Path(value)
	return path if path.is_absolute() else PROJECT_ROOT / path


def run_pipeline(config_path: Path = PROJECT_ROOT / "config" / "config.yaml") -> dict:
	"""Download data if necessary, train, evaluate, and save pipeline outputs."""
	with Path(config_path).open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)

	paths = config["paths"]
	features_config = config["features"]
	raw_dir = _project_path(paths["raw_dir"])
	processed_dir = _project_path(paths["processed_dir"])
	gold_dir = _project_path(paths["gold_dir"])
	processed_dir.mkdir(parents=True, exist_ok=True)
	gold_dir.mkdir(parents=True, exist_ok=True)

	daily = ingest_competition_data(
		raw_dir=raw_dir,
		competition=config["competition"]["slug"],
		sales_filename=paths["sales_file"],
		calendar_filename=paths["calendar_file"],
	)
	cleaned = clean_sales(
		daily,
		date_column=features_config["date_column"],
		target_column=features_config["target_column"],
		store_column=features_config["store_column"],
	)
	cleaned.to_csv(processed_dir / "sales_daily.csv", index=False)

	featured, feature_columns = create_features(
		cleaned,
		date_column=features_config["date_column"],
		target_column=features_config["target_column"],
		store_column=features_config["store_column"],
		lags=features_config["lags"],
		rolling_windows=features_config["rolling_windows"],
	)
	training, testing = split_by_date(featured, int(config["split"]["test_days"]))
	model_config = config["model"]
	model = RandomForestRegressor(
		n_estimators=int(model_config["n_estimators"]),
		min_samples_leaf=int(model_config["min_samples_leaf"]),
		random_state=int(model_config["random_state"]),
		n_jobs=-1,
	)
	model.fit(training[feature_columns], training[features_config["target_column"]])
	predictions = model.predict(testing[feature_columns])
	baseline_predictions = testing[f"lag_{min(features_config['lags'])}_sales"].to_numpy()
	metrics = evaluate_predictions(
		testing[features_config["target_column"]], predictions, baseline_predictions
	)

	output = testing[[features_config["date_column"], features_config["store_column"]]].copy()
	output["actual_sales"] = testing[features_config["target_column"]].to_numpy()
	output["baseline_prediction"] = baseline_predictions
	output["predicted_sales"] = predictions
	output.to_csv(processed_dir / "predictions.csv", index=False)
	metrics["split"] = {
		"train_start": str(training["date"].min().date()),
		"train_end": str(training["date"].max().date()),
		"test_start": str(testing["date"].min().date()),
		"test_end": str(testing["date"].max().date()),
		"train_rows": int(len(training)),
		"test_rows": int(len(testing)),
	}
	metrics_path = gold_dir / "evaluation_metrics.json"
	metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
	print(f"Saved cleaned data to {processed_dir / 'sales_daily.csv'}")
	print(f"Saved predictions to {processed_dir / 'predictions.csv'}")
	print(f"Saved evaluation results to {metrics_path}")
	print(json.dumps(metrics, indent=2))
	return metrics


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml"
	)
	args = parser.parse_args()
	run_pipeline(args.config)


if __name__ == "__main__":
	main()
