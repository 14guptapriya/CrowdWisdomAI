
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

DATA = Path("data")
df = pd.read_csv(DATA / "btc_sentiment_hourly.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

sentiment_cols = [
    "post_count", "avg_sentiment", "positive_ratio", "negative_ratio",
    "total_likes", "total_reposts", "total_views", "avg_followers"
]

for col in sentiment_cols:
    df[col] = df[col].shift(1)

feature_cols = [
    c for c in [
        "Open", "High", "Low", "Close", "Volume",
        "return_1h", "return_6h", "return_24h",
        *sentiment_cols
    ] if c in df.columns
]

df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols).reset_index(drop=True)

# Five chronological evaluation windows with expanding training history
n = len(df)
initial_train = int(n * 0.50)
test_size = int(n * 0.10)

all_predictions = []
metrics = []

for fold in range(5):
    train_end = initial_train + fold * test_size
    test_end = min(train_end + test_size, n)

    train = df.iloc[:train_end].copy()
    test = df.iloc[train_end:test_end].copy()

    if test.empty:
        continue

    target = "target_price_n+1"
    train = train.dropna(subset=[target])
    test = test.dropna(subset=[target])

    if train.empty or test.empty:
        continue

    # Predict percentage return, not absolute BTC price
    y_train = train[target] / train["Close"] - 1
    y_test = test[target] / test["Close"] - 1

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train[feature_cols], y_train)
    predicted_return = model.predict(test[feature_cols])

    result = pd.DataFrame({
        "timestamp": test["timestamp"].to_numpy(),
        "actual_return": y_test.to_numpy(),
        "predicted_return": predicted_return,
        "baseline_return": y_test.to_numpy() * 0
    })

    # Persistence baseline predicts zero return
    result["baseline_return"] = (
        test[target].to_numpy() / test["Close"].to_numpy() - 1
    )
    result["baseline_return"] = 0.0

    result["actual_price"] = test[target].to_numpy()
    result["predicted_price"] = (
        test["Close"].to_numpy() * (1 + predicted_return)
    )
    result["current_price"] = test["Close"].to_numpy()
    result["fold"] = fold + 1

    model_mae = mean_absolute_error(
        result["actual_price"], result["predicted_price"]
    )
    baseline_mae = mean_absolute_error(
        result["actual_price"], result["current_price"]
    )

    metrics.append({
        "fold": fold + 1,
        "train_rows": len(train),
        "test_rows": len(test),
        "model_MAE": model_mae,
        "baseline_MAE": baseline_mae
    })

    all_predictions.append(result)

pred = pd.concat(all_predictions, ignore_index=True)
pred = pred.sort_values("timestamp").reset_index(drop=True)

# Simple long-or-cash strategy using predicted next-hour returns
# 0.05% transaction cost per change in position
pred["position"] = (pred["predicted_return"] > 0).astype(float)
pred["turnover"] = pred["position"].diff().abs()
pred.loc[pred.index[0], "turnover"] = abs(pred["position"].iloc[0])

cost_rate = 0.0005
pred["strategy_return"] = (
    pred["position"] * pred["actual_return"]
    - cost_rate * pred["turnover"]
)

def summarize_returns(returns):
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return {"Sharpe": np.nan, "Sortino": np.nan, "Max_Drawdown_pct": np.nan}

    std = returns.std(ddof=1)
    sharpe = (
        returns.mean() / std * np.sqrt(24 * 365)
        if std > 0 else np.nan
    )

    downside = returns[returns < 0]
    downside_deviation = np.sqrt((downside ** 2).mean())
    sortino = (
        returns.mean() / downside_deviation * np.sqrt(24 * 365)
        if downside_deviation > 0 else np.nan
    )

    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1

    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown_pct": drawdown.min() * 100
    }

strategy_metrics = summarize_returns(pred["strategy_return"])

# Buy-and-hold baseline over the same forecast periods
buy_hold_metrics = summarize_returns(pred["actual_return"])

fold_metrics = pd.DataFrame(metrics)
fold_metrics.to_csv(DATA / "walk_forward_metrics.csv", index=False)
pred.to_csv(DATA / "walk_forward_predictions.csv", index=False)

summary = pd.DataFrame([
    {"strategy": "Model long-or-cash", **strategy_metrics},
    {"strategy": "Buy-and-hold benchmark", **buy_hold_metrics}
])
summary.to_csv(DATA / "trading_metrics.csv", index=False)

print("\nWALK-FORWARD RESULTS")
print(fold_metrics.to_string(index=False))
print("\nTRADING METRICS (hourly returns, annualized ratios)")
print(summary.to_string(index=False))
print("\nSaved:")
print("data/walk_forward_metrics.csv")
print("data/walk_forward_predictions.csv")
print("data/trading_metrics.csv")
