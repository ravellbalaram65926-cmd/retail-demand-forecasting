"""Create leakage-safe calendar and historical sales features."""

import pandas as pd


def create_features(
	data: pd.DataFrame,
	date_column: str = "date",
	target_column: str = "sales",
	store_column: str = "store_id",
	lags: tuple[int, ...] | list[int] = (1, 7),
	rolling_windows: tuple[int, ...] | list[int] = (7,),
) -> tuple[pd.DataFrame, list[str]]:
	"""Return rows with lagged features, dropping rows without history."""
	featured = data.copy()
	featured[date_column] = pd.to_datetime(featured[date_column])
	featured = featured.sort_values([store_column, date_column]).reset_index(drop=True)
	dates = featured[date_column]
	featured["day_of_week"] = dates.dt.day_of_week
	featured["month"] = dates.dt.month
	featured["day_of_month"] = dates.dt.day
	featured["store_code"] = featured[store_column].astype("category").cat.codes

	feature_columns = ["store_code", "day_of_week", "month", "day_of_month"]
	grouped_sales = featured.groupby(store_column, sort=False)[target_column]
	for lag in lags:
		column = f"lag_{lag}_sales"
		featured[column] = grouped_sales.shift(lag)
		feature_columns.append(column)

	for window in rolling_windows:
		column = f"rolling_mean_{window}_sales"
		featured[column] = featured.groupby(store_column, sort=False)[target_column].transform(
			lambda values: values.shift(1).rolling(window=window, min_periods=window).mean()
		)
		feature_columns.append(column)

	featured = featured.dropna(subset=feature_columns).reset_index(drop=True)
	return featured, feature_columns
