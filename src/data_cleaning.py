"""Validation and cleaning for aggregated daily sales data."""

import pandas as pd


def clean_sales(
	data: pd.DataFrame,
	date_column: str = "date",
	target_column: str = "sales",
	store_column: str = "store_id",
) -> pd.DataFrame:
	"""Drop unusable rows and duplicate store/date observations."""
	required = {date_column, target_column, store_column}
	missing_columns = required.difference(data.columns)
	if missing_columns:
		raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

	cleaned = data.copy()
	cleaned[date_column] = pd.to_datetime(cleaned[date_column], errors="coerce")
	cleaned[target_column] = pd.to_numeric(cleaned[target_column], errors="coerce")
	cleaned = cleaned.dropna(subset=[date_column, target_column, store_column])
	cleaned = cleaned[cleaned[target_column] >= 0]
	cleaned = cleaned.drop_duplicates(subset=[store_column, date_column], keep="last")
	return cleaned.sort_values([store_column, date_column]).reset_index(drop=True)
