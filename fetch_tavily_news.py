
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tavily import TavilyClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY missing. Check your .env file.")

client = TavilyClient(api_key=api_key)
analyzer = SentimentIntensityAnalyzer()

# Search recent news about Bitcoin
response = client.search(
    query="Bitcoin BTC latest market news price developments",
    topic="news",
    time_range="week",
    max_results=10
)

rows = []

for item in response.get("results", []):
    title = item.get("title", "")
    content = item.get("content", "")
    text = f"{title}. {content}".strip()
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    rows.append({
        "published_date": item.get("published_date", ""),
        "title": title,
        "content": content,
        "url": item.get("url", ""),
        "source": item.get("url", "").split("/")[2]
                  if item.get("url") else "",
        "sentiment_score": score,
        "sentiment_label": label
    })

if not rows:
    raise RuntimeError(
        "Tavily returned no news results. Try a broader query."
    )

df = pd.DataFrame(rows)
Path("data").mkdir(exist_ok=True)
df.to_csv("data/btc_news.csv", index=False)

positive = int((df["sentiment_label"] == "Positive").sum())
negative = int((df["sentiment_label"] == "Negative").sum())
neutral = int((df["sentiment_label"] == "Neutral").sum())
mean_score = df["sentiment_score"].mean()

if mean_score >= 0.05:
    outlook = "Positive"
elif mean_score <= -0.05:
    outlook = "Negative"
else:
    outlook = "Neutral / Mixed"

summary = f"""# Recent BTC News Sentiment

- Articles collected: {len(df)}
- Average headline/content sentiment: {mean_score:.3f}
- Overall news sentiment: {outlook}
- Positive articles: {positive}
- Negative articles: {negative}
- Neutral articles: {neutral}

## Recent headlines

"""

for _, row in df.head(10).iterrows():
    summary += (
        f"### {row['title']}\n"
        f"- Published: {row['published_date'] or 'Not provided'}\n"
        f"- Sentiment: {row['sentiment_label']} "
        f"({row['sentiment_score']:.3f})\n"
        f"- Source: {row['url']}\n\n"
    )

summary += (
    "\n*Sentiment uses VADER, a general-purpose lexicon, "
    "and is not a validated financial sentiment model. "
    "News sentiment does not predict price direction by itself.*\n"
)

Path("outputs").mkdir(exist_ok=True)
Path("outputs/btc_news_sentiment.md").write_text(
    summary, encoding="utf-8"
)

print(f"Articles collected: {len(df)}")
print(f"Average sentiment: {mean_score:.3f} ({outlook})")
print("Saved: data/btc_news.csv")
print("Saved: outputs/btc_news_sentiment.md")
print("\nRecent headlines:")
print(df[["title", "sentiment_label", "url"]].to_string(index=False))
