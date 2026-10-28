
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "btc_sentiment_hourly.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp")

latest = df.iloc[-1]
recent = df.tail(24)
previous = df.iloc[-48:-24]

sentiment = recent["avg_sentiment"].mean()
prev_sentiment = previous["avg_sentiment"].mean()
post_volume = recent["post_count"].sum()
prev_volume = previous["post_count"].sum()
price_change = (
    (recent["Close"].iloc[-1] / recent["Close"].iloc[0] - 1) * 100
)
sentiment_change = sentiment - prev_sentiment

if sentiment > 0.05:
    sentiment_label = "Positive"
elif sentiment < -0.05:
    sentiment_label = "Negative"
else:
    sentiment_label = "Neutral / Mixed"

if sentiment_change > 0.02:
    trend = "Improving"
elif sentiment_change < -0.02:
    trend = "Deteriorating"
else:
    trend = "Broadly stable"

if prev_volume > 0:
    volume_change = (post_volume / prev_volume - 1) * 100
    volume_text = f"{volume_change:+.1f}%"
else:
    volume_text = "Not available (no prior posts)"

metrics_file = DATA / "walk_forward_metrics.csv"
trading_file = DATA / "trading_metrics.csv"

trading_note = "Trading metrics not available."
if trading_file.exists():
    trading = pd.read_csv(trading_file)
    trading_note = trading.to_string(index=False)

report = f"""# BTC Sentiment Outlook
**Generated:** {pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")}  
**Data through:** {latest["timestamp"].strftime("%Y-%m-%d %H:%M UTC")}

## Executive summary
X sentiment over the latest 24 hourly bars is **{sentiment_label.lower()}**,
with a mean VADER score of **{sentiment:.3f}**. The sentiment trend is
**{trend.lower()}** compared with the preceding 24 hours. BTC's price changed
by **{price_change:+.2f}%** over the latest 24 hourly observations.

## Key indicators
- Latest BTC close: **${latest["Close"]:,.2f}**
- Mean sentiment (latest 24 bars): **{sentiment:.3f}**
- Mean sentiment (previous 24 bars): **{prev_sentiment:.3f}**
- Sentiment change: **{sentiment_change:+.3f}**
- X posts (latest 24 bars): **{int(post_volume):,}**
- Post volume change vs previous 24 bars: **{volume_text}**

## Interpretation
Sentiment scores summarize the language of collected X posts, not the
intentions of all market participants. Changes in post volume may reflect
attention or news events and do not necessarily imply a bullish or bearish
move. Price movement and sentiment can diverge.

## Model assessment
The Random Forest was evaluated using chronological walk-forward windows.
Its errors should be compared with the current-price baseline in
`data/walk_forward_metrics.csv`. The earlier experiment showed that the model
did not reliably beat that baseline. The simulated long-or-cash strategy
metrics are:

{trading_note}

These results are historical experiments, not financial advice or a
guarantee of future performance. Trading ratios are sensitive to the short
evaluation period, costs, and execution assumptions.

## Data limitations
The analysis depends on the collected X sample, VADER's general-purpose
sentiment lexicon, hourly aggregation, and available BTC price history.
Collected posts may not represent the entire X conversation. This report
does not include verified live-news sentiment unless a news source is
explicitly added.
"""

path = OUT / "btc_sentiment_outlook.md"
path.write_text(report, encoding="utf-8")
print(f"Saved: {path}")
print("\n" + report)
