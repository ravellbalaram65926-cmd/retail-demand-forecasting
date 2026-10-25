# Retail Demand Forecasting

End-to-end retail demand forecasting on the Kaggle M5 Forecasting Accuracy dataset.
The pipeline ingests item-level data, aggregates it to store-day sales, builds
leakage-safe time features, trains a Random Forest model, compares it with a
previous-day baseline, and saves predictions and MAE/RMSE results.

## Setup

Use Python 3.10 or newer, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

To download the competition data, accept the M5 competition rules on Kaggle and
add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Codespaces repository or environment
secrets. KaggleHub reads these environment variables automatically. Never put
credentials in this repository or in the notebook. Locally, set the same
variables in your shell before running the pipeline.

The download step can also use files already placed in `data/raw/`:
`sales_train_validation.csv` and `calendar.csv`.

## Run

From the repository root, run the complete pipeline:

```bash
python -m src.train
```

The competition files are downloaded to `data/raw/` if they are not already
there. Aggregated and cleaned data is written to `data/processed/sales_daily.csv`,
holdout predictions to `data/processed/predictions.csv`, and evaluation metrics
to `data/gold/evaluation_metrics.json`. Adjust paths, features, model settings,
and the date-based holdout length in `config/config.yaml`.

To run only ingestion and aggregation:

```bash
python -m src.data_ingestion
```

## Tests

Run the offline test suite (it uses small synthetic fixtures and does not need
Kaggle credentials or the competition download):

```bash
python -m pytest -q
```

GitHub Actions runs the same tests on pushes and pull requests.

## Workflow

1. Download M5 sales and calendar files into `data/raw/`.
2. Aggregate item sales by store and day, then remove invalid and duplicate rows.
3. Create calendar, lag, and rolling-average features using only prior sales.
4. Train on earlier dates and hold out the latest configured number of dates.
5. Compare the model and previous-day baseline with MAE and RMSE.
6. Save cleaned data, predictions, and metrics to the configured output folders.

## Notebook prototype

The following sample-data prototype was already present in this README and is
preserved here for reference; use the configured pipeline above for the M5 data.

```python
import pandas as pd
plt.show()
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create sample sales data for testing
np.random.seed(42)

dates = pd.date_range(start="2025-01-01", periods=120, freq="D")

sales_data = pd.DataFrame({
    "date": dates,
    "store_id": 1,
    "item_id": 101,
    "sales": [
        max(0, int(50 + i * 0.2 + 10 * np.sin(i / 7) + np.random.normal(0, 5)))
        for i in range(len(dates))
    ]
})

print("Sample sales data:")
display(sales_data.head())

print("Number of rows:", len(sales_data))
# Work with a copy so the original data stays unchanged
df = sales_data.copy()

# Convert the date column to a proper date format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove rows with missing dates or sales
df = df.dropna(subset=["date", "sales"])

# Remove duplicate records
df = df.drop_duplicates(subset=["date", "store_id", "item_id"])

# Keep sales values that are zero or higher
df = df[df["sales"] >= 0]

# Sort the data by date
df = df.sort_values("date").reset_index(drop=True)

print("Data cleaning completed.")
print("Rows after cleaning:", len(df))
print("Missing values:")
print(df.isnull().sum())

display(df.head())
# Create calendar features
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day

# Use previous sales to help predict future sales
df["previous_day_sales"] = df["sales"].shift(1)

# Calculate the average sales from the previous seven days
df["average_sales_7_days"] = (
    df["sales"]
    .shift(1)
    .rolling(window=7)
    .mean()
)

# Remove rows that do not have enough historical data
df = df.dropna().reset_index(drop=True)

features = [
    "day_of_week",
    "month",
    "day_of_month",
    "previous_day_sales",
    "average_sales_7_days"
]

target = "sales"

print("Features created:")
print(features)

display(df[["date"] + features + [target]].head())
# Keep the records in date order
df = df.sort_values("date").reset_index(drop=True)

# Use the first 80% of the data for training
split_index = int(len(df) * 0.8)

train_data = df.iloc[:split_index]
test_data = df.iloc[split_index:]

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

# Train the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict sales for the test period
predictions = model.predict(X_test)

print("Model training completed.")
print("Training records:", len(train_data))
print("Testing records:", len(test_data))
# Measure how close the predictions are to actual sales
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Compare actual sales with predicted sales
results = pd.DataFrame({
    "date": test_data["date"].values,
    "actual_sales": y_test.values,
    "predicted_sales": predictions
})

display(results.head(10))

# Plot the actual and predicted sales
plt.figure(figsize=(12, 5))

plt.plot(
    results["date"],
    results["actual_sales"],
    label="Actual sales"
)

plt.plot(
    results["date"],
    results["predicted_sales"],
    label="Predicted sales"
)

plt.title("Actual vs Predicted Retail Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()