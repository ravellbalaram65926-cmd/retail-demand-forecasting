"""Download and aggregate the M5 competition data."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def download_competition_data(competition: str, raw_dir: Path) -> Path:
	"""Download competition files with KaggleHub and copy them into raw_dir."""
	try:
		import kagglehub

		download_dir = Path(kagglehub.competition_download(competition))
	except Exception as exc:
		raise RuntimeError(
			"Kaggle download failed. Configure KAGGLE_USERNAME and KAGGLE_KEY "
			"as Codespaces secrets, and accept the competition rules on Kaggle."
		) from exc

	if not download_dir.is_dir():
		raise FileNotFoundError(f"KaggleHub returned no download directory: {download_dir}")

	raw_dir.mkdir(parents=True, exist_ok=True)
	downloaded_files = [path for path in download_dir.rglob("*") if path.is_file()]
	if not downloaded_files:
		raise FileNotFoundError(f"No files found in Kaggle download: {download_dir}")

	for source in downloaded_files:
		destination = raw_dir / source.name
		if source.resolve() != destination.resolve():
			shutil.copy2(source, destination)
	return raw_dir


def aggregate_m5_sales(sales_path: Path, calendar_path: Path) -> pd.DataFrame:
	"""Aggregate M5's wide item-level sales table into daily store totals."""
	columns = pd.read_csv(sales_path, nrows=0).columns.tolist()
	day_columns = [column for column in columns if column.startswith("d_")]
	if "store_id" not in columns or not day_columns:
		raise ValueError("Sales file must contain store_id and d_# columns")

	sales = pd.read_csv(
		sales_path,
		usecols=["store_id", *day_columns],
		dtype={"store_id": "category", **{column: np.int16 for column in day_columns}},
	)
	totals_by_store = sales.groupby("store_id", observed=True)[day_columns].sum()
	wide_totals = totals_by_store.reset_index()
	daily = wide_totals.melt(
		id_vars="store_id", var_name="date_id", value_name="sales"
	)

	calendar = pd.read_csv(calendar_path, usecols=["d", "date"])
	calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
	date_lookup = calendar.dropna(subset=["date"]).set_index("d")["date"]
	daily["date"] = daily["date_id"].map(date_lookup)
	if daily["date"].isna().any():
		missing = daily.loc[daily["date"].isna(), "date_id"].unique()
		raise ValueError(f"Calendar is missing M5 date IDs: {missing[:5].tolist()}")

	return daily.drop(columns="date_id").sort_values(
		["store_id", "date"]
	).reset_index(drop=True)


def ingest_competition_data(
	raw_dir: Path,
	competition: str = "m5-forecasting-accuracy",
	sales_filename: str = "sales_train_validation.csv",
	calendar_filename: str = "calendar.csv",
) -> pd.DataFrame:
	"""Use local raw files when present, otherwise download the competition."""
	raw_dir = Path(raw_dir)
	sales_path = raw_dir / sales_filename
	calendar_path = raw_dir / calendar_filename
	if not sales_path.is_file() or not calendar_path.is_file():
		download_competition_data(competition, raw_dir)
	if not sales_path.is_file() or not calendar_path.is_file():
		raise FileNotFoundError(
			f"Expected {sales_filename} and {calendar_filename} in {raw_dir}"
		)
	return aggregate_m5_sales(sales_path, calendar_path)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
	parser.add_argument("--competition", default="m5-forecasting-accuracy")
	args = parser.parse_args()
	daily = ingest_competition_data(args.raw_dir, args.competition)
	print(f"Aggregated {len(daily):,} store-day rows into {args.raw_dir}")


if __name__ == "__main__":
	main()
