# Architecture and workflow

## Data source

The pipeline uses the Kaggle M5 Forecasting Accuracy competition. KaggleHub
downloads `sales_train_validation.csv` and `calendar.csv`; the source files are
copied unchanged into `data/raw/`. The item-level sales table is read with only
the store and daily sales columns, summed by store, and mapped from M5 day IDs
to calendar dates.

Kaggle access is configured outside the repository using the
`KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables. The account must have
accepted the competition rules. No credentials are written to disk by this
project.

## Pipeline stages

1. `src/data_ingestion.py` obtains local or Kaggle source files and aggregates
	item sales into daily store totals.
2. `src/data_cleaning.py` parses dates and numeric values, drops invalid or
	missing records, rejects negative sales, removes duplicate store/date rows,
	and sorts the result.
3. `src/feature_engineering.py` adds calendar fields, store codes, historical
	sales lags, and shifted rolling means. All target-derived features use only
	earlier observations within the same store.
4. `src/train.py` holds out the latest configured dates, trains a Random Forest,
	and compares it to a previous-day-sales baseline.
5. `src/evaluate.py` computes MAE and RMSE. The pipeline saves processed data,
	predictions, and metrics under `data/processed/` and `data/gold/`.

Configuration is centralized in `config/config.yaml`. The default evaluation
uses the latest 28 dates as the test period. The test suite uses synthetic data
and does not require Kaggle access.

## Outputs

- `data/processed/sales_daily.csv`: cleaned store-day data.
- `data/processed/predictions.csv`: actuals and forecasts for each test row.
- `data/gold/evaluation_metrics.json`: model/baseline MAE and RMSE, date split,
  and row counts.
