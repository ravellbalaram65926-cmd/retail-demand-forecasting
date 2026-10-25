import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data_cleaning import clean_sales
from src.data_ingestion import aggregate_m5_sales
from src.evaluate import evaluate_predictions, regression_metrics
from src.feature_engineering import create_features
from src.train import run_pipeline, split_by_date


def test_aggregate_m5_sales_maps_dates_and_sums_by_store(tmp_path):
	sales_path = tmp_path / "sales.csv"
	calendar_path = tmp_path / "calendar.csv"
	pd.DataFrame(
		{
			"store_id": ["CA_1", "CA_1", "TX_1"],
			"d_1": [1, 2, 3],
			"d_2": [4, 5, 6],
		}
	).to_csv(sales_path, index=False)
	pd.DataFrame(
		{"d": ["d_1", "d_2"], "date": ["2016-01-01", "2016-01-02"]}
	).to_csv(calendar_path, index=False)

	result = aggregate_m5_sales(sales_path, calendar_path)

	assert result.to_dict("records") == [
		{"store_id": "CA_1", "sales": 3, "date": pd.Timestamp("2016-01-01")},
		{"store_id": "CA_1", "sales": 9, "date": pd.Timestamp("2016-01-02")},
		{"store_id": "TX_1", "sales": 3, "date": pd.Timestamp("2016-01-01")},
		{"store_id": "TX_1", "sales": 6, "date": pd.Timestamp("2016-01-02")},
	]


def test_clean_sales_drops_invalid_and_duplicate_records():
	data = pd.DataFrame(
		{
			"date": [
				"2025-01-01",
				"2025-01-01",
				"invalid",
				"2025-01-02",
				"2025-01-03",
			],
			"store_id": ["A", "A", "A", "A", "A"],
			"sales": [10, 12, 4, -1, None],
		}
	)

	result = clean_sales(data)

	assert result[["date", "sales"]].to_dict("records") == [
		{"date": pd.Timestamp("2025-01-01"), "sales": 12}
	]


def test_features_use_only_prior_sales_and_are_separate_by_store():
	data = pd.DataFrame(
		{
			"date": list(pd.date_range("2025-01-01", periods=3).repeat(2)),
			"store_id": ["A", "B"] * 3,
			"sales": [10, 100, 20, 200, 30, 300],
		}
	)

	result, feature_columns = create_features(data, lags=[1], rolling_windows=[2])

	assert {"lag_1_sales", "rolling_mean_2_sales"}.issubset(feature_columns)
	assert result["lag_1_sales"].tolist() == [20, 200]
	assert result["rolling_mean_2_sales"].tolist() == [15, 150]


def test_split_by_date_holds_out_latest_days():
	data = pd.DataFrame(
		{"date": pd.date_range("2025-01-01", periods=10), "sales": range(10)}
	)

	training, testing = split_by_date(data, test_days=3)

	assert training["date"].max() < testing["date"].min()
	assert testing["date"].nunique() == 3
	assert testing["sales"].tolist() == [7, 8, 9]


def test_metrics_report_mae_rmse_and_baseline():
	metrics = evaluate_predictions([1, 3], [2, 2], [1, 1])

	assert metrics["model"] == {"mae": 1.0, "rmse": 1.0}
	assert metrics["baseline"] == {"mae": 1.0, "rmse": pytest.approx(2**0.5)}
	with pytest.raises(ValueError):
		regression_metrics([1], [1, 2])


def test_run_pipeline_saves_processed_data_predictions_and_metrics(tmp_path, monkeypatch):
	dates = pd.date_range("2025-01-01", periods=20)
	sales = pd.DataFrame(
		{
			"date": list(dates) * 2,
			"store_id": ["A"] * 20 + ["B"] * 20,
			"sales": list(range(20, 40)) + list(range(30, 50)),
		}
	)
	monkeypatch.setattr("src.train.ingest_competition_data", lambda **kwargs: sales)
	config_path = tmp_path / "config.yaml"
	config_path.write_text(
		yaml.safe_dump(
			{
				"competition": {"slug": "unused"},
				"paths": {
					"raw_dir": str(tmp_path / "raw"),
					"processed_dir": str(tmp_path / "processed"),
					"gold_dir": str(tmp_path / "gold"),
					"sales_file": "sales.csv",
					"calendar_file": "calendar.csv",
				},
				"features": {
					"store_column": "store_id",
					"date_column": "date",
					"target_column": "sales",
					"lags": [1],
					"rolling_windows": [3],
				},
				"split": {"test_days": 5},
				"model": {
					"n_estimators": 5,
					"min_samples_leaf": 1,
					"random_state": 42,
				},
			}
		),
		encoding="utf-8",
	)

	metrics = run_pipeline(config_path)
	processed_dir = tmp_path / "processed"
	prediction_data = pd.read_csv(processed_dir / "predictions.csv")
	saved_metrics = json.loads((tmp_path / "gold" / "evaluation_metrics.json").read_text())

	assert (processed_dir / "sales_daily.csv").is_file()
	assert len(prediction_data) == 10
	assert set(["actual_sales", "baseline_prediction", "predicted_sales"]).issubset(
		prediction_data.columns
	)
	assert saved_metrics["model"] == metrics["model"]
	assert np.isfinite(metrics["model"]["mae"])
