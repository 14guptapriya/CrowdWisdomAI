
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Load evaluation results
walk_path = DATA / "walk_forward_metrics.csv"
trade_path = DATA / "trading_metrics.csv"
news_path = DATA / "btc_news.csv"
outlook_path = OUT / "btc_sentiment_outlook.md"
news_summary_path = OUT / "btc_news_sentiment.md"

walk = pd.read_csv(walk_path) if walk_path.exists() else pd.DataFrame()
trade = pd.read_csv(trade_path) if trade_path.exists() else pd.DataFrame()
news = pd.read_csv(news_path) if news_path.exists() else pd.DataFrame()

# BTC and X sentiment overview
df = pd.read_csv(DATA / "btc_sentiment_hourly.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp")
latest = df.iloc[-1]
recent = df.tail(24)

avg_sentiment = recent["avg_sentiment"].mean()
post_count = int(recent["post_count"].sum())
price_change = (
    recent["Close"].iloc[-1] / recent["Close"].iloc[0] - 1
) * 100

if avg_sentiment >= 0.05:
    x_outlook = "Positive"
elif avg_sentiment <= -0.05:
    x_outlook = "Negative"
else:
    x_outlook = "Neutral / Mixed"

news_outlook = "Unavailable"
news_mean = np.nan

if not news.empty and "sentiment_score" in news:
    news_mean = news["sentiment_score"].mean()
    if news_mean >= 0.05:
        news_outlook = "Positive"
    elif news_mean <= -0.05:
        news_outlook = "Negative"
    else:
        news_outlook = "Neutral / Mixed"

report = f"""# BTC X Sentiment & Price Prediction — Evaluation Report

**Asset:** Bitcoin (BTC-USD)  
**Generated:** {pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")}  
**Latest market data:** {latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC")}

## 1. Executive summary

This project combines X post sentiment and engagement with hourly BTC price
features to investigate short-horizon price forecasting. Apify was used to
collect X posts, historical price data was merged hourly, and Tavily was used
to retrieve recent Bitcoin news.

- Latest BTC close: **${latest["Close"]:,.2f}**
- BTC price change over the latest 24 hourly observations: **{price_change:+.2f}%**
- Mean X sentiment: **{avg_sentiment:.3f} ({x_outlook})**
- X posts over latest 24 hourly observations: **{post_count:,}**
- Mean Tavily news sentiment: **{news_mean:.3f} ({news_outlook})**
- News articles analyzed: **{len(news)}**

Sentiment labels are heuristic summaries, not investment recommendations.

## 2. Data and feature engineering

- X post text, timestamps, likes, reposts, views and author followers.
- Hourly BTC OHLCV prices.
- VADER sentiment, positive/negative proportions and hourly post volume.
- Lagged price returns over 1, 6 and 24 hours.
- Future BTC close targets at n+0 through n+5.

X sentiment features are shifted by one hour in the model to reduce same-period
information leakage. Tavily news is analyzed separately and is not a model
input in the current experiment.

## 3. Walk-forward validation

"""

if not walk.empty:
    report += walk.to_string(index=False) + "\n\n"
    model_better = int((walk["model_MAE"] < walk["baseline_MAE"]).sum())
    report += (
        f"The model beats the current-price baseline in "
        f"{model_better} of {len(walk)} evaluation windows. "
        "Lower MAE is better.\n\n"
    )
else:
    report += "Walk-forward metrics are unavailable.\n\n"

report += """## 4. Trading simulation

"""

if not trade.empty:
    report += trade.to_string(index=False) + "\n\n"
else:
    report += "Trading metrics are unavailable.\n\n"

report += """The trading simulation is an experimental long-or-cash strategy.
Its results are historical, sensitive to transaction-cost and execution
assumptions, and not evidence of future profitability.

## 5. Recent news

"""

if not news.empty:
    for _, row in news.head(10).iterrows():
        report += (
            f"- **{row.get('title', 'Untitled')}** — "
            f"{row.get('sentiment_label', 'Unclassified')} sentiment. "
            f"Source: {row.get('url', '')}\n"
        )
else:
    report += "No Tavily news dataset was found.\n"

report += """
## 6. Visual outputs

- `outputs/btc_sentiment_price_matrix.png`
- `outputs/btc_sentiment_outlook.md`
- `outputs/btc_news_sentiment.md`

## 7. Limitations

1. X posts may not represent all market participants.
2. VADER is a general-purpose sentiment tool, not a finance-specific model.
3. A limited historical window may produce unstable evaluation metrics.
4. Current experiments have not established consistent improvement over the
   persistence baseline.
5. News sentiment is not included in the forecasting model.
6. Trading results do not include all real-world market frictions.

## Conclusion

This is an end-to-end research prototype for collecting, aligning and
evaluating social sentiment, news and BTC price data. Current results do not
justify claiming a reliable trading edge. Further feature validation,
finance-specific sentiment analysis, longer historical data and repeated
out-of-sample testing are required.
"""

path = OUT / "final_evaluation_report.md"
path.write_text(report, encoding="utf-8")

print(f"Saved: {path}")
print("\nReport preview:\n")
print(report)
