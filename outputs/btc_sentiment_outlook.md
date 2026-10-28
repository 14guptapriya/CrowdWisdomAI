# BTC Sentiment Outlook
**Generated:** 2026-10-28 20:12 UTC  
**Data through:** 2026-10-24 19:00 UTC

## Executive summary
X sentiment over the latest 24 hourly bars is **neutral / mixed**,
with a mean VADER score of **0.000**. The sentiment trend is
**broadly stable** compared with the preceding 24 hours. BTC's price changed
by **-1.15%** over the latest 24 hourly observations.

## Key indicators
- Latest BTC close: **$80,379.97**
- Mean sentiment (latest 24 bars): **0.000**
- Mean sentiment (previous 24 bars): **0.000**
- Sentiment change: **+0.000**
- X posts (latest 24 bars): **0**
- Post volume change vs previous 24 bars: **Not available (no prior posts)**

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

              strategy    Sharpe   Sortino  Max_Drawdown_pct
    Model long-or-cash -7.871986 -5.441900        -26.886948
Buy-and-hold benchmark  2.551798  2.782864         -8.564021

These results are historical experiments, not financial advice or a
guarantee of future performance. Trading ratios are sensitive to the short
evaluation period, costs, and execution assumptions.

## Data limitations
The analysis depends on the collected X sample, VADER's general-purpose
sentiment lexicon, hourly aggregation, and available BTC price history.
Collected posts may not represent the entire X conversation. This report
does not include verified live-news sentiment unless a news source is
explicitly added.
