# CrowdWisdomAI — X Sentiment & BTC Price Prediction

An end-to-end machine learning pipeline that combines X (Twitter) sentiment, Bitcoin market data and recent financial news to explore short-horizon BTC price movements.

> **Research prototype only. Not financial advice.** Historical results do not establish future profitability.

## Overview

CrowdWisdomAI investigates whether social-media sentiment and engagement signals can improve short-term Bitcoin price predictions.

The pipeline:
- Collects Bitcoin-related X posts using Apify.
- Cleans and deduplicates posts and analyzes text sentiment using VADER.
- Aggregates sentiment and engagement features by UTC hour.
- Aligns social signals with hourly BTC market data.
- Trains Random Forest regression models for short-horizon return prediction.
- Evaluates predictions against a persistence baseline and uses walk-forward validation.
- Calculates experimental trading metrics and generates evaluation charts.
- Retrieves recent Bitcoin-related news through Tavily for supplementary sentiment analysis.
- Provides a Streamlit dashboard for saved results and live pipeline testing.

## Architecture

```text
X Posts ── Apify ── Cleaning & VADER Sentiment ──┐
                                                │
BTC Market Data ── Hourly Aggregation ──────────┤
                                                ▼
                                  Timestamp Alignment
                                                │
                                                ▼
                                   Feature Engineering
                                                │
                                                ▼
                                     Random Forest
                                                │
                                                ▼
                                  Forecasts & Evaluation
                                                │
                           ┌────────────────────┴─────────────┐
                           ▼                                  ▼
                     Streamlit App                     Evaluation Report

Tavily News ── Headline Sentiment ── Supplementary News Outlook
```

News sentiment is supplementary and is not included in the trained model's feature set.

## Technology Stack

- Python, Pandas, NumPy
- Scikit-learn — Random Forest regression
- VADER Sentiment — text sentiment analysis
- Apify — X post collection
- Binance public market-data API — BTC/USDT candles for live testing
- Tavily — recent news search
- Streamlit — interactive dashboard
- Matplotlib — visualization

## Data Sources

### X post collection

- **Actor name:** `pro100chok/x-twitter-all-in-one-scraper`
- **Actor ID:** `pro100chok~x-twitter-all-in-one-scraper`
- **Search targets:** `Bitcoin lang:en`, `$BTC lang:en`, `BTC-USD lang:en`
- **Collection period:** April 27–October 25, 2026
- **Collection method:** Weekly exports through the Apify interface
- **Language:** English
- **Sentiment method:** VADER compound sentiment score

The collected posts are timestamped in UTC, deduplicated and aggregated into hourly features.

### Bitcoin market data

Historical hourly BTC-USD data is aligned with hourly X sentiment. The live dashboard retrieves hourly BTC/USDT candles from Binance's public market-data endpoint.

### News

Tavily retrieves recent Bitcoin-related news. VADER analyzes the available headline and article text. News analysis is presented separately and does not directly influence the trained model.

## Project Structure

```text
CrowdWisdomAI/
├── app.py
├── build_dataset.py
├── train_model.py
├── walk_forward_evaluation.py
├── visualize_results.py
├── sentiment_outlook.py
├── fetch_tavily_news.py
├── generate_final_report.py
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── btc_sentiment_hourly.csv
│   ├── evaluation_metrics.csv
│   ├── test_predictions.csv
│   ├── walk_forward_metrics.csv
│   ├── walk_forward_predictions.csv
│   ├── trading_metrics.csv
│   └── btc_news.csv
└── outputs/
    ├── final_evaluation_report.md
    ├── btc_sentiment_outlook.md
    └── btc_sentiment_price_matrix.png
```

This is the intended structure; only include files that actually exist in the submitted repository. Raw post exports and local environment files are excluded from the public repository.

## Installation

Python 3.10 or later is recommended.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CrowdWisdomAI
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API credentials

Create a local `.env` file:

```text
APIFY_TOKEN=your_apify_token
TAVILY_API_KEY=your_tavily_api_key
```

Never commit `.env` or publish live credentials.

Apify is used for X collection, and Tavily is used for news retrieval. Exa was not used in this implementation.

## Running the Project

### Streamlit dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

- **Saved-data demo:** Inspect stored predictions, evaluation metrics, charts and available news results.
- **Live test:** Collect a limited set of recent X posts, retrieve market data, align features and generate a next-hour prediction using the historical training dataset.
- **News analysis:** Retrieve recent news and inspect supplementary sentiment.

Live collection requires valid API credentials and network access. The saved-data demo requires the corresponding processed datasets and result files.

### Build the historical dataset

```bash
python build_dataset.py
```

This script cleans and deduplicates X posts, calculates sentiment and engagement features, aggregates posts hourly, aligns them with BTC prices and creates future-price targets.

### Train the model

```bash
python train_model.py
```

The training script uses chronological train/test separation, shifts sentiment features by one hour to reduce look-ahead leakage, and trains separate Random Forest regressors for future return horizons. It saves evaluation metrics and test predictions.

### Walk-forward evaluation

```bash
python walk_forward_evaluation.py
```

This evaluates model performance across expanding chronological training windows.

### Generate visualizations and reports

```bash
python visualize_results.py
python sentiment_outlook.py
python fetch_tavily_news.py
python generate_final_report.py
```

The news script requires a valid Tavily API key. Run the evaluation scripts before generating the final report so the report reflects the latest results.

## Feature Engineering

The model uses:

- **Price features:** Open, High, Low, Close and Volume.
- **Return features:** 1-hour, 6-hour and 24-hour historical returns.
- **Sentiment features:** Average sentiment, positive and negative post ratios, and post count.
- **Engagement features:** Likes, reposts, views and author follower counts.

The target is the future percentage return over each prediction horizon. Predicted returns are converted back into estimated BTC prices.

The `n+0` value represents the current observed price, not a genuine forecast. The primary forecasting horizons are `n+1` through `n+5`.

## Evaluation Results

### Chronological holdout

| Horizon | Model MAE ($) | Persistence MAE ($) | Model MAPE |
|---|---:|---:|---:|
| n+1 | 201.91 | 192.28 | 0.242% |
| n+2 | 295.74 | 271.61 | 0.354% |
| n+3 | 387.70 | 335.96 | 0.464% |
| n+4 | 449.25 | 393.58 | 0.538% |
| n+5 | 503.10 | 443.07 | 0.602% |

The persistence baseline uses the latest observed price as the future-price estimate. On this holdout, the Random Forest model did not outperform that baseline at any reported horizon.

### Walk-forward validation

Five expanding-window evaluations were performed for the next-hour horizon.

| Fold | Model MAE ($) | Baseline MAE ($) |
|---|---:|---:|
| 1 | 125.57 | 123.86 |
| 2 | 210.60 | 209.56 |
| 3 | 190.35 | 189.08 |
| 4 | 209.91 | 196.99 |
| 5 | 188.58 | 187.32 |

The model did not outperform the persistence baseline in any of these five folds. This is an important negative result: the current features and model have not demonstrated incremental predictive value over the baseline.

### Experimental trading metrics

| Strategy | Sharpe Ratio | Sortino Ratio | Maximum Drawdown |
|---|---:|---:|---:|
| Model long-or-cash | -7.872 | -5.442 | -26.887% |
| Buy-and-hold | 2.552 | 2.783 | -8.564% |

The historical simulation uses a long-or-cash strategy based on predicted next-hour returns and assumes a transaction cost of 0.05% per position change. Sharpe and Sortino ratios are annualized using an hourly-bar convention of 24 × 365 periods per year.

These are experimental in-sample-period simulation results, not evidence of a profitable trading system. They are sensitive to the strategy, cost assumptions, period and implementation.

See [`outputs/final_evaluation_report.md`](outputs/final_evaluation_report.md) for the consolidated report, additional results and limitations.

## Limitations

- X posts may be incomplete or biased by search coverage, engagement and account characteristics.
- VADER is a general-purpose sentiment tool and may misinterpret sarcasm, financial jargon or context.
- Hourly aggregation can hide intrahour effects and differences in posting volume.
- Sentiment lagging reduces one source of leakage but does not guarantee a completely leakage-free pipeline.
- A chronological holdout and walk-forward evaluation are useful safeguards, but do not establish generalization across market regimes.
- The current model underperforms the persistence baseline and its experimental trading strategy performs poorly.
- News sentiment is supplementary and is not currently a model input.
- API limits, access permissions, market-data availability and third-party service changes may affect reproducibility.

## Reproducibility and Credentials

The public repository documents the actor identifier, data sources, model scripts and evaluation outputs. A reviewer can reproduce the processed-data workflow using the included datasets where redistribution is permitted, or recollect data using their own authorized credentials.

API tokens must be supplied separately through a secure channel. Do not place tokens in source code, Git history, public issue reports or screenshots.

## Disclaimer

This project is an educational machine-learning research prototype. Predictions and simulated trading metrics are not investment recommendations, and the model should not be used to make real-money trading decisions.