# Prediction Engine V2 Upgrade Plan

## Goal Description
Enhance the accuracy and depth of Nunno Finance's crypto predictions by integrating multi-dimensional data sources (Sentiment, On-Chain, Derivatives), implementing an AI Ensemble model (ML + LLM), and improving visual data representation. This moves the app from simple technical analysis to a professional-grade market intelligence platform.

## System Architecture Diagram

```mermaid
graph TD
    User[User / Frontend]
    
    subgraph "Data Ingestion Layer"
        Source_Price[Price Data API]
        Source_Social[Social Media APIs\n(Reddit, Twitter, Telegram)]
        Source_OnChain[On-Chain APIs\n(Glassnode/Dune)]
        Source_News[News Aggregators]
    end

    subgraph "Backend Services"
        TAS[Technical Analysis Service]
        Sent[Sentiment Service\n(New)]
        OnChain[On-Chain Service\n(New)]
        Deriv[Derivatives Service\n(Funding/OI)]
        News[News Service]
    end

    subgraph "Prediction Engine"
        ML_Model[ML Models\n(XGBoost / LSTM)]
        LLM_Analyst[LLM Reasoning\n(Strategy synthesis)]
        Ensemble[Ensemble Manager\n(Weighted Consensus)]
    end

    User -- Request Prediction --> Ensemble
    
    Source_Price --> TAS
    Source_Price --> Deriv
    Source_Social --> Sent
    Source_OnChain --> OnChain
    Source_News --> News

    TAS --> ML_Model
    Deriv --> ML_Model
    Sent --> ML_Model
    OnChain --> ML_Model

    TAS --> LLM_Analyst
    Deriv --> LLM_Analyst
    Sent --> LLM_Analyst
    OnChain --> LLM_Analyst
    News --> LLM_Analyst

    ML_Model -- "Technical Signals (Probabilities)" --> Ensemble
    LLM_Analyst -- "Strategic Insight + Confidence" --> Ensemble
    
    Ensemble -- "Final Prediction + Confidence Score" --> User
```

## User Review Required
> [!IMPORTANT]
> **API Access Costs**: Many on-chain and social sentiment APIs (Glassnode, Santiment, Twitter API) have significant costs. This plan assumes using available free tiers or free alternatives (e.g., web scraping for some social signals) where possible, but premium APIs may be needed for production stability.

> [!WARNING]
> **Performance Impact**: Gathering data from multiple new external sources (Reddit, Glassnode, etc.) will increase latency. We must implement aggressive caching and background data fetching to keep user response times fast.

## Proposed Changes

### 1. New Backend Services

#### [NEW] [sentiment_service.py](file:///c:/Users/CSC/Desktop/Nunno%20Streamlit/NunnoFinance/backend/services/sentiment_service.py)
- **Purpose**: Aggregates social sentiment.
- **Features**:
    - Reddit Scraper: r/CryptoCurrency, r/Bitcoin (mentions, sentiment score).
    - Google Trends: Pytrends integration for search volume.
    - Fear & Greed Index: Fetcher.
    - News Sentiment: Integration with existing `NewsService` to output a numerical score.

#### [NEW] [onchain_service.py](file:///c:/Users/CSC/Desktop/Nunno%20Streamlit/NunnoFinance/backend/services/onchain_service.py)
- **Purpose**: Fetches blockchain metrics.
- **Features**:
    - Active Addresses & Transaction Volume.
    - Exchange Inflow/Outflow (Reserves).
    - MVRV & NUPL (via free endpoints or calculated approximations).
    - Whale Alert tracking.

#### [NEW] [derivatives_service.py](file:///c:/Users/CSC/Desktop/Nunno%20Streamlit/NunnoFinance/backend/services/derivatives_service.py)
- **Purpose**: Tracks futures and options market data.
- **Features**:
    - Funding Rates (Aggregated).
    - Open Interest (OI) trends.
    - Long/Short Ratios.

### 2. Prediction Engine Core

#### [MODIFY] [betterpredictormodule.py](file:///c:/Users/CSC/Desktop/Nunno%20Streamlit/NunnoFinance/backend/services/betterpredictormodule.py)
- **Ensemble Logic**:
    - Create a class `PredictionEnsemble`.
    - Implement weighting logic: `Final_Score = (w1 * Tech_Signal) + (w2 * Sentiment) + (w3 * OnChain) + (w4 * LLM_Confidence)`.
    - Dynamic Weighting: Decrease technical weight if "Event Risk" (High News Speculation) is detected.

#### [MODIFY] [ml_predictor.py](file:///c:/Users/CSC/Desktop/Nunno%20Streamlit/NunnoFinance/backend/ml_predictor.py)
- **Feature Expansion**:
    - Add input columns for Sentiment Score, OI Change, and Funding Rate.
    - Retrain/Adjust model architecture to handle these new non-price features.

### 3. API & Data Handling

#### [MODIFY] [main.py](file:///c:/Users/CSC/Desktop/Nunno%20Streamlit/NunnoFinance/backend/main.py)
- **Endpoints**:
    - `GET /api/v1/sentiment/{symbol}`
    - `GET /api/v1/onchain/{symbol}`
    - `GET /api/v1/prediction/ensemble/{symbol}` (The combined master endpoint)

#### [NEW] [data_reliability.py](file:///c:/Users/CSC/Desktop/Nunno%20Streamlit/NunnoFinance/backend/utils/data_reliability.py)
- **Fallback System**:
    - Define hierarchy: API (Primary) -> Web Scrape (Secondary) -> Historical Avg/Neutral (Fallback).
    - Status reporting: "Used fallback data for Sentiment".

### 4. Frontend Visuals (Plan for Future UI Task)

- **Heatmaps**: React component using `react-heatmap-grid` to show confluence of indicators.
- **Sentiment Timeline**: Line chart overlaying Price vs. Social Volume.
- **Indicator Dashboard**: A grid of "gauges" for Fear/Greed, RSI, MVRV.

## Verification Plan

### Automated Tests
- **Unit Tests**:
    - Test `sentiment_service` returns valid scores (-1 to 1).
    - Test `onchain_service` handles API failures gracefully (Fallback check).
    - Test `Ensemble` logic ensures math correctness (weights sum to 1, output bounded).
- **Integration Tests**:
    - Run `test_enhanced_prediction.py` (to be created) which mocks all data sources and verifies the full ensemble pipeline produces a prediction.

### Manual Verification
- **Data Accuracy**: Compare app outputs for BTC/ETH against public dashboards (CoinGecko, Alternative.me, Glassnode free charts).
- **Latency Check**: Ensure the "Ensemble Prediction" returns within 5-8 seconds max.
