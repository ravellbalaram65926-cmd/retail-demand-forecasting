# Data layout

- `raw/`: original Kaggle M5 files (`sales_train_validation.csv` and
	`calendar.csv`). These files are ignored by Git because they are large and
	subject to Kaggle's competition terms.
- `processed/`: aggregated, cleaned daily store sales and holdout predictions.
	Generated files are ignored by Git.
- `gold/`: evaluation metrics produced by the training pipeline. Metrics are
	small and may be versioned with the project.

Run `python -m src.train` from the repository root to populate the output
directories. Configure Kaggle credentials as environment secrets; do not save
them in this directory.
