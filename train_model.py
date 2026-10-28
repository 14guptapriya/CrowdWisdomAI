
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = Path("data")
df = pd.read_csv(DATA / "btc_sentiment_hourly.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

# Features known after the current hourly candle closes.
# Shift sentiment by one hour to avoid using posts from the
# same hour that may have arrived after the prediction time.
sentiment_cols = [
    "post_count", "avg_sentiment", "positive_ratio", "negative_ratio",
    "total_likes", "total_reposts", "total_views", "avg_followers"
]
for col in sentiment_cols:
    df[col] = df[col].shift(1)

price_cols = [
    c for c in ["Open", "High", "Low", "Close", "Volume"]
    if c in df.columns
]
return_cols = ["return_1h", "return_6h", "return_24h"]

feature_cols = price_cols + return_cols + sentiment_cols
feature_cols = [c for c in feature_cols if c in df.columns]

df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols).reset_index(drop=True)

# Keep the final 20% as an untouched chronological test set.
split = int(len(df) * 0.8)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy().reset_index(drop=True)

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Features:", feature_cols)

results = []
predictions = pd.DataFrame({
    "timestamp": test["timestamp"].to_numpy(),
    "actual_current_close": test["Close"].to_numpy()
}, index=test.index)

# n+0 is the current observed close, so persistence is exact.
results.append({
    "horizon": "n+0",
    "model": "Current price (persistence)",
    "MAE": 0.0,
    "RMSE": 0.0,
    "MAPE_pct": 0.0
})

# Train a separate model for each future hour.
for h in range(1, 6):
    target = f"target_price_n+{h}"

    train_h = train.dropna(subset=[target])
    test_h = test.dropna(subset=[target])

    
    X_train = train_h[feature_cols]
    X_test = test_h[feature_cols]

    # Predict future percentage return instead of absolute price
    y_train = (
        train_h[target] / train_h["Close"] - 1
    )

    y_test = test_h[target].to_numpy()
    baseline = test_h["Close"].to_numpy()

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predicted_return = model.predict(X_test)
    pred = baseline * (1 + predicted_return)


    # Simple baseline: future price equals the current close.
    baseline = test_h["Close"].to_numpy()

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(
        np.abs((y_test - pred) /
               np.maximum(np.abs(y_test), 1e-8))
    ) * 100

    b_mae = mean_absolute_error(y_test, baseline)
    b_rmse = np.sqrt(mean_squared_error(y_test, baseline))

    results.append({
        "horizon": f"n+{h}",
        "model": "RandomForest",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_pct": mape,
        "Baseline_MAE": b_mae,
        "Baseline_RMSE": b_rmse
    })

    predictions.loc[test_h.index, f"actual_n+{h}"] = y_test
    predictions.loc[test_h.index, f"predicted_n+{h}"] = pred
    predictions.loc[test_h.index, f"baseline_n+{h}"] = baseline

    print(
        f"n+{h}: MAE={mae:.2f}, RMSE={rmse:.2f}, "
        f"MAPE={mape:.3f}% | baseline MAE={b_mae:.2f}"
    )

# Save results
metrics = pd.DataFrame(results)
metrics.to_csv(DATA / "evaluation_metrics.csv", index=False)
predictions.to_csv(DATA / "test_predictions.csv", index=False)

print("\nSaved:")
print("data/evaluation_metrics.csv")
print("data/test_predictions.csv")
print("\nMetrics summary:")
print(metrics.to_string(index=False))
