
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "btc_sentiment_hourly.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp")

# One figure with four aligned charts
fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

axes[0].plot(df["timestamp"], df["Close"], label="BTC Close")
axes[0].set_title("BTC Price")
axes[0].set_ylabel("Price (USD)")
axes[0].legend()

axes[1].plot(
    df["timestamp"], df["avg_sentiment"],
    label="Average X Sentiment"
)
axes[1].axhline(0, linestyle="--", linewidth=1)
axes[1].set_title("X Sentiment Over Time")
axes[1].set_ylabel("Sentiment")
axes[1].legend()

axes[2].bar(
    df["timestamp"], df["post_count"],
    width=0.03, label="X Post Volume"
)
axes[2].set_title("X Post Volume")
axes[2].set_ylabel("Posts/hour")
axes[2].legend()

pred_path = DATA / "test_predictions.csv"
if pred_path.exists():
    pred = pd.read_csv(pred_path)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)

    actual_col = "actual_n+1"
    predicted_col = "predicted_n+1"

    if actual_col in pred and predicted_col in pred:
        axes[3].plot(
            pred["timestamp"], pred[actual_col],
            label="Actual BTC price (n+1)"
        )
        axes[3].plot(
            pred["timestamp"], pred[predicted_col],
            label="Predicted BTC price (n+1)"
        )
        axes[3].set_title("One-Hour-Ahead Prediction vs Actual")
        axes[3].set_ylabel("Price (USD)")
        axes[3].legend()
else:
    axes[3].text(
        0.5, 0.5, "Run train_model.py to display predictions",
        ha="center", va="center", transform=axes[3].transAxes
    )
    axes[3].set_title("Model Predictions")

for ax in axes:
    ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "btc_sentiment_price_matrix.png", dpi=180)
plt.close(fig)

print("Saved: outputs/btc_sentiment_price_matrix.png")
