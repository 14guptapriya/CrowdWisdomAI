import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

st.set_page_config(page_title="CrowdWisdomAI | Pipeline Tester", page_icon="📈", layout="wide")
st.title("CrowdWisdomAI — Pipeline Tester")
st.caption("X sentiment + BTC market data + recent news. Research demo only; not financial advice.")

DATA = "data"
ACTOR_ID = "pro100chok~x-twitter-all-in-one-scraper"
SENTIMENT_COLS = [
    "post_count", "avg_sentiment", "positive_ratio", "negative_ratio",
    "total_likes", "total_reposts", "total_views", "avg_followers"
]
PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]
RETURN_COLS = ["return_1h", "return_6h", "return_24h"]
FEATURE_COLS = PRICE_COLS + RETURN_COLS + SENTIMENT_COLS
analyzer = SentimentIntensityAnalyzer()


def safe_float(value):
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def first_present(row, names, default=0):
    for name in names:
        value = row.get(name)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    return default


def fetch_apify_posts(token, start_date, end_date, max_posts):
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/run-sync-get-dataset-items"
    payload = {
        "scrapeType": "search",
        "targets": ["Bitcoin lang:en", "$BTC lang:en", "BTC-USD lang:en"],
        "searchType": "latest",
        "start": start_date.isoformat(),
        "end": (end_date + timedelta(days=1)).isoformat(),
        "maxItems": max_posts,
        "tweetLanguage": "en",
        "excludeReplies": True,
        "excludeReposts": True,
        "includeSearchTerms": True,
    }
    response = requests.post(
        url,
        params={"timeout": 120},
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=135,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected Apify response: {str(data)[:500]}")
    return pd.DataFrame(data)


def normalize_posts(posts):
    if posts.empty:
        return posts
    if "createdAt" not in posts.columns:
        raise ValueError(f"Apify output has no createdAt field. Fields: {list(posts.columns)}")
    posts = posts.copy()
    posts["createdAt"] = pd.to_datetime(posts["createdAt"], utc=True, errors="coerce")
    posts = posts.dropna(subset=["createdAt"])
    posts["text"] = posts.get("text", pd.Series("", index=posts.index)).fillna("").astype(str)
    posts["sentiment"] = posts["text"].map(lambda text: analyzer.polarity_scores(text)["compound"])
    posts["positive"] = (posts["sentiment"] > 0.05).astype(int)
    posts["negative"] = (posts["sentiment"] < -0.05).astype(int)
    def numeric_series(names):
        for name in names:
            if name in posts.columns:
                return pd.to_numeric(posts[name], errors="coerce").fillna(0)
        return pd.Series(0.0, index=posts.index)

    posts["likes"] = numeric_series(["likeCount", "likes"])
    posts["reposts"] = numeric_series(["repostCount", "retweetCount", "retweets"])
    posts["views"] = numeric_series(["viewCount", "views"])

    author_followers = posts.get("author")
    if author_followers is not None:
        followers = author_followers.map(
            lambda a: safe_float(a.get("followers", 0)) if isinstance(a, dict) else 0
        )
    else:
        followers = pd.to_numeric(posts.get("followers", 0), errors="coerce")
    posts["followers"] = pd.Series(followers, index=posts.index).fillna(0)
    return posts


def aggregate_posts(posts):
    if posts.empty:
        return pd.DataFrame(columns=["timestamp"] + SENTIMENT_COLS)
    posts = posts.copy()
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
    return hourly


def fetch_binance_prices(start_date, end_date):
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date + timedelta(days=1), tz="UTC")
    url = "https://api.binance.com/api/v3/klines"
    rows = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms:
        response = requests.get(
            url,
            params={"symbol": "BTCUSDT", "interval": "1h", "startTime": cursor, "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break
    if not rows:
        raise ValueError("Binance returned no hourly price candles for this date range.")
    cols = ["open_time", "Open", "High", "Low", "Close", "Volume", "close_time",
            "quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote", "ignore"]
    prices = pd.DataFrame(rows, columns=cols)
    prices["timestamp"] = pd.to_datetime(prices["open_time"], unit="ms", utc=True)
    for col in PRICE_COLS:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    return prices[["timestamp"] + PRICE_COLS].drop_duplicates("timestamp").sort_values("timestamp")


def make_live_dataset(prices, hourly_posts):
    data = prices.merge(hourly_posts, on="timestamp", how="left")
    for col in SENTIMENT_COLS:
        if col not in data:
            data[col] = 0.0
        data[col] = data[col].fillna(0.0)
    data = data.sort_values("timestamp").reset_index(drop=True)
    data["return_1h"] = data["Close"].pct_change()
    data["return_6h"] = data["Close"].pct_change(6)
    data["return_24h"] = data["Close"].pct_change(24)
    # Same leakage-control convention used by train_model.py.
    data[SENTIMENT_COLS] = data[SENTIMENT_COLS].shift(1)
    return data.replace([np.inf, -np.inf], np.nan)


def train_and_predict(live_data):
    historical = pd.read_csv(f"{DATA}/btc_sentiment_hourly.csv")
    historical["timestamp"] = pd.to_datetime(historical["timestamp"], utc=True, errors="coerce")
    historical = historical.sort_values("timestamp").reset_index(drop=True)
    missing = [c for c in FEATURE_COLS + ["target_price_n+1"] if c not in historical.columns]
    if missing:
        raise ValueError(f"Historical dataset is missing required columns: {missing}")

    # Mirror train_model.py: sentiment is shifted one hour, then train on future n+1 percentage return.
    historical[SENTIMENT_COLS] = historical[SENTIMENT_COLS].shift(1)
    historical[FEATURE_COLS] = historical[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    historical = historical.dropna(subset=FEATURE_COLS + ["target_price_n+1", "Close"])
    historical = historical[historical["Close"] != 0]
    X_train = historical[FEATURE_COLS]
    y_train = historical["target_price_n+1"] / historical["Close"] - 1

    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    candidates = live_data.dropna(subset=FEATURE_COLS).copy()
    if candidates.empty:
        raise ValueError("Not enough live price history to compute all features (need at least 25 hours).")
    latest = candidates.iloc[[-1]]
    predicted_return = float(model.predict(latest[FEATURE_COLS])[0])
    current_price = float(latest["Close"].iloc[0])
    predicted_price = current_price * (1 + predicted_return)
    return current_price, predicted_price, predicted_return, latest.iloc[0]["timestamp"], len(historical)


def fetch_tavily_news(api_key, max_results=10):
    from tavily import TavilyClient
    client = TavilyClient(api_key=api_key)
    response = client.search(
        query="Bitcoin BTC latest market news price developments",
        topic="news",
        time_range="week",
        max_results=max_results,
    )
    rows = []
    for item in response.get("results", []):
        text = f"{item.get('title', '')}. {item.get('content', '')}"
        score = analyzer.polarity_scores(text)["compound"]
        label = "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
        rows.append({
            "published_date": item.get("published_date", ""),
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "url": item.get("url", ""),
            "sentiment_score": score,
            "sentiment_label": label,
        })
    return pd.DataFrame(rows)


mode = st.sidebar.radio("Mode", ["Saved-data demo", "Live test"])
st.sidebar.caption("Live mode uses Apify, Binance public market data, and Tavily.")

if mode == "Saved-data demo":
    st.subheader("Saved evaluation results")
    metrics_path = f"{DATA}/evaluation_metrics.csv"
    pred_path = f"{DATA}/test_predictions.csv"
    trading_path = f"{DATA}/trading_metrics.csv"
    wf_path = f"{DATA}/walk_forward_metrics.csv"
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
        st.markdown("#### Forecast accuracy")
        st.dataframe(metrics, use_container_width=True)
    else:
        st.warning("Missing data/evaluation_metrics.csv. Run train_model.py first.")
    if os.path.exists(trading_path):
        st.markdown("#### Trading simulation")
        st.dataframe(pd.read_csv(trading_path), use_container_width=True)
    if os.path.exists(wf_path):
        st.markdown("#### Walk-forward evaluation")
        st.dataframe(pd.read_csv(wf_path), use_container_width=True)
    if os.path.exists(pred_path):
        pred = pd.read_csv(pred_path)
        if "timestamp" in pred:
            pred["timestamp"] = pd.to_datetime(pred["timestamp"], errors="coerce")
        numeric_cols = [c for c in pred.columns if c.startswith(("actual_n+1", "predicted_n+1", "baseline_n+1"))]
        if numeric_cols:
            st.markdown("#### Held-out n+1 predictions")
            chart_cols = [c for c in ["actual_n+1", "predicted_n+1", "baseline_n+1"] if c in pred.columns]
            st.line_chart(pred.set_index("timestamp")[chart_cols] if "timestamp" in pred else pred[chart_cols])
    chart_path = "outputs/btc_sentiment_price_matrix.png"
    if os.path.exists(chart_path):
        st.markdown("#### Existing sentiment/price chart")
        st.image(chart_path, use_container_width=True)
    news_path = f"{DATA}/btc_news.csv"
    if os.path.exists(news_path):
        st.markdown("#### Saved Tavily news sentiment")
        news = pd.read_csv(news_path)
        st.dataframe(news[[c for c in ["published_date", "title", "sentiment_label", "sentiment_score", "url"] if c in news]], use_container_width=True)
    st.info("Saved-data mode does not make new API requests. It displays your existing experiment outputs.")

else:
    st.subheader("Run a small live ingestion test")
    st.write("Use a short date range and a small post cap to control API usage.")
    today = date.today()
    default_end = today - timedelta(days=1)
    default_start = default_end - timedelta(days=2)
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start date (UTC)", value=default_start, max_value=default_end)
    with col2:
        end_date = st.date_input("End date (UTC)", value=default_end, min_value=start_date, max_value=default_end)
    with col3:
        max_posts = st.select_slider("Max posts per search target", options=[10, 20, 30, 50, 100], value=20)

    apify_default = os.getenv("APIFY_TOKEN", "")
    tavily_default = os.getenv("TAVILY_API_KEY", "")
    apify_token = st.text_input("Apify API token", value=apify_default, type="password", help="Stored only for this app session unless configured as a deployment secret.")
    tavily_key = st.text_input("Tavily API key (optional)", value=tavily_default, type="password")
    run_live = st.button("Run live ingestion + n+1 prediction", type="primary")

    if run_live:
        if not apify_token:
            st.error("Enter an Apify token or configure APIFY_TOKEN in your local .env / deployment secrets.")
        elif start_date > end_date:
            st.error("Start date must be on or before end date.")
        else:
            try:
                with st.spinner("Fetching X posts from Apify..."):
                    posts_raw = fetch_apify_posts(apify_token, start_date, end_date, max_posts)
                    posts = normalize_posts(posts_raw)
                st.success(f"Apify returned {len(posts_raw):,} rows; {len(posts):,} posts had valid timestamps.")
                if not posts.empty:
                    show_cols = [c for c in ["createdAt", "text", "likeCount", "repostCount", "viewCount", "sentiment"] if c in posts]
                    st.markdown("#### Sample X posts")
                    st.dataframe(posts[show_cols].head(20), use_container_width=True)
                    st.metric("Mean X sentiment", f"{posts['sentiment'].mean():.3f}")
                else:
                    st.warning("No posts returned for this date range. Widen the range or change the search.")
                with st.spinner("Fetching hourly BTC prices from Binance..."):
                    prices = fetch_binance_prices(start_date, end_date)
                st.success(f"Fetched {len(prices):,} hourly BTC/USDT candles.")
                hourly = aggregate_posts(posts)
                live_data = make_live_dataset(prices, hourly)
                current, predicted, predicted_return, forecast_time, training_rows = train_and_predict(live_data)

                st.markdown("### Latest n+1 model output")
                a, b, c = st.columns(3)
                a.metric("Latest BTC close (USDT)", f"${current:,.2f}")
                b.metric("Model n+1 estimate", f"${predicted:,.2f}", f"{predicted_return * 100:+.3f}%")
                c.metric("Historical training rows", f"{training_rows:,}")
                st.caption(f"Feature timestamp: {forecast_time} UTC. The estimate is for the next hourly step after the latest available candle.")
                st.line_chart(live_data.set_index("timestamp")[["Close"]].tail(72))
                if "timestamp" in live_data:
                    st.markdown("#### Recent hourly X sentiment")
                    sent_chart = live_data[["timestamp", "avg_sentiment", "post_count"]].tail(72).set_index("timestamp")
                    st.line_chart(sent_chart)

                st.markdown("#### Ingestion summary")
                st.write({
                    "X posts returned": int(len(posts_raw)),
                    "valid timestamp posts": int(len(posts)),
                    "hourly price candles": int(len(prices)),
                    "hours with posts": int((hourly["post_count"] > 0).sum()) if not hourly.empty else 0,
                    "Actor ID": ACTOR_ID,
                    "news included in prediction model": False,
                })
                st.warning("This is an experimental one-step estimate, not a validated trading signal. The historical model has underperformed the persistence baseline in your previous evaluation.")
            except requests.HTTPError as exc:
                body = exc.response.text[:1000] if exc.response is not None else str(exc)
                st.error(f"API request failed: {body}")
            except Exception as exc:
                st.exception(exc)

    st.divider()
    st.markdown("#### Optional: fetch current Tavily news")
    st.caption("News sentiment is supplementary and is not fed into the current Random Forest model.")
    if st.button("Fetch Tavily news"):
        key = tavily_key or os.getenv("TAVILY_API_KEY", "")
        if not key:
            st.error("Enter a Tavily key above or set TAVILY_API_KEY in .env / deployment secrets.")
        else:
            try:
                with st.spinner("Searching recent Bitcoin news..."):
                    news = fetch_tavily_news(key)
                if news.empty:
                    st.warning("Tavily returned no articles.")
                else:
                    st.metric("Articles returned", len(news))
                    st.metric("Average news sentiment", f"{news['sentiment_score'].mean():.3f}")
                    st.dataframe(news[["published_date", "title", "sentiment_label", "sentiment_score", "url"]], use_container_width=True)
            except Exception as exc:
                st.exception(exc)

st.divider()
st.caption("Data providers: Apify X scraper, Binance public klines API, Tavily news search. Outputs are for assessment/research, not financial advice.")
