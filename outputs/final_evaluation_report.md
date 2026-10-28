# BTC X Sentiment & Price Prediction — Evaluation Report

**Asset:** Bitcoin (BTC-USD)  
**Generated:** 2026-10-28 20:21 UTC  
**Latest market data:** 2026-10-24 19:00 UTC

## 1. Executive summary

This project combines X post sentiment and engagement with hourly BTC price
features to investigate short-horizon price forecasting. Apify was used to
collect X posts, historical price data was merged hourly, and Tavily was used
to retrieve recent Bitcoin news.

- Latest BTC close: **$80,379.97**
- BTC price change over the latest 24 hourly observations: **-1.15%**
- Mean X sentiment: **0.000 (Neutral / Mixed)**
- X posts over latest 24 hourly observations: **0**
- Mean Tavily news sentiment: **0.279 (Positive)**
- News articles analyzed: **8**

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

 fold  train_rows  test_rows  model_MAE  baseline_MAE
    1        2157        431 125.574280    123.855568
    2        2588        431 210.603857    209.564512
    3        3019        431 190.346103    189.076022
    4        3450        431 209.908078    196.992913
    5        3881        431 188.578509    187.320312

The model beats the current-price baseline in 0 of 5 evaluation windows. Lower MAE is better.

## 4. Trading simulation

              strategy    Sharpe   Sortino  Max_Drawdown_pct
    Model long-or-cash -7.871986 -5.441900        -26.886948
Buy-and-hold benchmark  2.551798  2.782864         -8.564021

The trading simulation is an experimental long-or-cash strategy.
Its results are historical, sensitive to transaction-cost and execution
assumptions, and not evidence of future profitability.

## 5. Recent news

- **Latest Bitcoin News - (BTC) Future Outlook, Trends & Market Insights** — Positive sentiment. Source: https://coinmarketcap.com/cmc-ai/bitcoin/latest-updates
- **CoinDesk tracks live Bitcoin price and market cap across ...** — Positive sentiment. Source: https://www.facebook.com/quartznews/posts/coindesk-tracks-live-bitcoin-price-and-market-cap-across-all-major-markets-coind/1476860200976460
- **Bitcoin Price Forecast: BTC consolidates below $80,000 as Middle East tensions, tech earnings loom** — Negative sentiment. Source: https://www.tmgm.com/en/analysis/market-news/article/bitcoin-price-forecast-btc-consolidates-below-80-000-as-middle-east-tensions-tech-earnings-loom-202610270935
- **Bitcoin Traders Are Preparing for the Fed in 2 Very Different Ways** — Positive sentiment. Source: https://beincrypto.com/bitcoin-exchange-outflows-binance-shorts-fed
- **Bitcoin Price Today | Live Bitcoin Price, Chart & Market Cap** — Positive sentiment. Source: https://humb.io/price/btc
- **Anthropic Files for Nasdaq IPO** — Positive sentiment. Source: https://investingnews.com/cryptocurrency-market-recap
- **Bitcoin falls below $82,000 as investors retreat from risk assets** — Positive sentiment. Source: https://finance.yahoo.com/markets/article/bitcoin-falls-below-82000-as-investors-retreat-from-risk-assets-153142669.html
- **8.9K views · 44 reactions | Top crypto things you might have missed this week.

#xrp #ethereum #solana #bitcoin #crypto | Crypto Wendy O** — Negative sentiment. Source: https://www.facebook.com/CryptoWendyO/videos/top-crypto-things-you-might-have-missed-this-weekxrp-ethereum-solana-bitcoin-cry/1781349373155724

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
