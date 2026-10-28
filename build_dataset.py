
from pathlib import Path
import glob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA = Path("data")
START = pd.Timestamp("2026-04-27", tz="UTC")
END = pd.Timestamp("2026-10-25", tz="UTC")

# 1. Combine all weekly exports
files = sorted(glob.glob(str(DATA / "x_post_*.csv")))

if not files:
    raise FileNotFoundError("No x_post_*.csv files found inside data/")

posts = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in files],
    ignore_index=True
)

print("Weekly files:", len(files))
print("Raw posts:", len(posts))

# 2. Clean timestamps and duplicates
if "createdAt" not in posts.columns:
    raise ValueError(f"Missing createdAt column. Found: {list(posts.columns)}")

posts["createdAt"] = pd.to_datetime(
    posts["createdAt"], utc=True, errors="coerce"
)
posts = posts.dropna(subset=["createdAt"])
posts = posts[posts["createdAt"].ge(START) & posts["createdAt"].lt(END)]

if "id" in posts.columns:
    posts = posts.drop_duplicates(subset=["id"])
else:
    posts = posts.drop_duplicates(subset=["createdAt", "text"])

posts["text"] = posts.get("text", pd.Series("", index=posts.index)).fillna("").astype(str)

# Handle Apify column naming variations
def numeric_column(candidates):
    for col in candidates:
        if col in posts.columns:
            return pd.to_numeric(posts[col], errors="coerce").fillna(0)
    return pd.Series(0.0, index=posts.index)

posts["likes"] = numeric_column(["likeCount", "likes"])
posts["reposts"] = numeric_column(["repostCount", "retweetCount", "retweets"])
posts["views"] = numeric_column(["viewCount", "views"])
posts["followers"] = numeric_column([
    "author/followers", "author.followers", "followers"
])

# 3. Calculate sentiment for each post
analyzer = SentimentIntensityAnalyzer()
posts["sentiment"] = posts["text"].map(
    lambda text: analyzer.polarity_scores(text)["compound"]
)
posts["positive"] = (posts["sentiment"] > 0.05).astype(int)
posts["negative"] = (posts["sentiment"] < -0.05).astype(int)

# 4. Aggregate posts by UTC hour
posts["timestamp"] = posts["createdAt"].dt.floor("h")

hourly = posts.groupby("timestamp").agg(
    post_count=("sentiment", "size"),
    avg_sentiment=("sentiment", "mean"),
    positive_ratio=("positive", "mean"),
    negative_ratio=("negative", "mean"),
    total_likes=("likes", "sum"),
    total_reposts=("reposts", "sum"),
    total_views=("views", "sum"),
    avg_followers=("followers", "mean"),
).reset_index()

# 5. Load and clean hourly BTC prices
price = pd.read_csv(DATA / "btc_usd_hourly.csv")
price["timestamp"] = pd.to_datetime(
    price["timestamp"], utc=True, errors="coerce"
)
price = price.dropna(subset=["timestamp"])
price = price[
    price["timestamp"].ge(START) & price["timestamp"].lt(END)
]
price = price.drop_duplicates("timestamp").sort_values("timestamp")

if "Close" not in price.columns:
    raise ValueError(f"Missing Close price. Found: {list(price.columns)}")

# 6. Merge sentiment with price bars
dataset = price.merge(hourly, on="timestamp", how="left")
dataset["post_count"] = dataset["post_count"].fillna(0)

feature_cols = [
    "avg_sentiment", "positive_ratio", "negative_ratio",
    "total_likes", "total_reposts", "total_views", "avg_followers"
]
dataset[feature_cols] = dataset[feature_cols].fillna(0)

dataset = dataset.sort_values("timestamp").reset_index(drop=True)

# 7. Add historical returns and future price targets
dataset["return_1h"] = dataset["Close"].pct_change()
dataset["return_6h"] = dataset["Close"].pct_change(6)
dataset["return_24h"] = dataset["Close"].pct_change(24)

for h in range(6):
    dataset[f"target_price_n+{h}"] = dataset["Close"].shift(-h)

# Save results
posts.to_csv(DATA / "x_posts_180d.csv", index=False)
dataset.to_csv(DATA / "btc_sentiment_hourly.csv", index=False)

print("\nSaved cleaned posts:", len(posts))
print("BTC hourly rows:", len(price))
print("Merged dataset:", len(dataset))
print("Hours with posts:", int((dataset["post_count"] > 0).sum()))
print("Date range:", dataset["timestamp"].min(), "to", dataset["timestamp"].max())
print("Output: data/btc_sentiment_hourly.csv")
print("\nPreview:")
print(dataset.head())
