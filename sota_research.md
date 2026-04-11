# State-of-the-Art Research: Stock Price Prediction with News Sentiment

This document reviews top Kaggle notebooks and community approaches for stock prediction that combine OHLCV data with news or sentiment. Findings inform the design of [`stock_prediction.ipynb`](stock_prediction.ipynb).

**Sources analyzed:** five high-vote Kaggle notebooks (~1,100 community votes combined), covering GAN, LSTM, XGBoost, hybrid, and multi-step setups.

---

## 1. Analyzed notebooks

| # | Notebook | Votes | Model | NLP method | Horizon |
|---|----------|-------|-------|------------|---------|
| 1 | [Stock Prediction GAN + Twitter Sentiment](https://www.kaggle.com/code/equinxx/stock-prediction-gan-twitter-sentiment-analysis) | 447 | Conditional GAN (LSTM generator + Conv1D discriminator) | VADER (NLTK) | Next-day |
| 2 | [News Sentiment Based Trading Strategy](https://www.kaggle.com/code/shtrausslearning/news-sentiment-based-trading-strategy) | 143 | LogReg, SVM, RF, LSTM text classifier | Custom VADER, TextBlob, SpaCy embeddings | Event return (analysis) |
| 3 | [EDA and LSTM with Generator for Market and News](https://www.kaggle.com/code/dmitrypukhov/eda-and-lstm-with-generator-for-market-and-news) | 270 | 3-layer LSTM (128-64-32) | Pre-computed sentiment columns (no raw NLP) | 10-day direction |
| 4 | [XGBoost + LSTM for Netflix Stock](https://www.kaggle.com/code/mehmetakifciftci/xgboost-lstm-for-netflix-stock) | 75 | XGBoost + LSTM hybrid (0.6 / 0.4) | None (price-only) | Next-day |
| 5 | [Tesla Stock Forecasting Multi-Step Stacked LSTM](https://www.kaggle.com/code/guslovesmath/tesla-stock-forecasting-multi-step-stacked-lstm) | 151 | Stacked LSTM (40-20) | None (price-only) | Multi-step (10 days in code) |

---

## 2. Detailed analysis per notebook

### 2.1 GAN + Twitter sentiment (447 votes)

**Data:** Twitter stock tweets (per ticker) + Yahoo Finance OHLCV.

**NLP:** VADER per tweet; daily `groupby(date).mean()`; left-join to prices; `ffill()` for missing sentiment days.

**Price features:** MA(7,20), MACD, Bollinger (20), EMA, momentum; drop first 20 rows; `MinMaxScaler(-1, 1)` on all columns together.

**Training:** Adam 5e-4, 500 epochs, batch size 5, `predict_period=1`. Last 20 batches held out.

**Conditional GAN idea:** The discriminator sees **past window + candidate next value** (real next close vs generator output), so it learns whether the proposed step is plausible given history.

#### Neural architecture (Mermaid)

```mermaid
flowchart TB
  subgraph generator [Generator LSTM stack]
    direction TB
    gIn[Input_window_scaled]
    gL1["LSTM 1024 units return_sequences=True recurrent_dropout 0.3"]
    gL2["LSTM 512 return_sequences=True"]
    gL3["LSTM 256 return_sequences=True"]
    gL4["LSTM 128 return_sequences=True"]
    gL5["LSTM 64 return_sequences=False"]
    gD1[Dense 32]
    gD2[Dense 16]
    gD3[Dense 8]
    gOut[Dense output_dim linear next_step]
    gIn --> gL1 --> gL2 --> gL3 --> gL4 --> gL5 --> gD1 --> gD2 --> gD3 --> gOut
  end

  subgraph discriminator [Discriminator Conv1D stack]
    direction TB
    dIn["Input concat past_window plus next_scalar shape input_dim plus 1"]
    dC1["Conv1D 8 filters k3 s2 same LeakyReLU 0.01"]
    dC2["Conv1D 16 k3 s2"]
    dC3["Conv1D 32 k3 s2"]
    dC4["Conv1D 64 k3 s2"]
    dC5["Conv1D 128 k1 s2"]
    dL1[LeakyReLU]
    dFc1[Dense 220]
    dFc2["Dense 220 relu"]
    dSig[Dense 1 sigmoid]
    dIn --> dC1 --> dC2 --> dC3 --> dC4 --> dC5 --> dL1 --> dFc1 --> dFc2 --> dSig
  end

  gOut --> trainLoop[Adversarial_train_step]
  dSig --> trainLoop
```

**Training loop (conceptual):**

```mermaid
flowchart LR
  realY[Real_next_close]
  fakeY[Generator_output]
  hist[Past_window_features]
  dReal["D concat realY hist"]
  dFake["D concat fakeY hist"]
  hist --> dReal
  realY --> dReal
  hist --> dFake
  fakeY --> dFake
```

**Takeaways:** Strong generator capacity vs small data (overfitting risk); VADER is a weak financial signal; conditional D is the main modeling idea worth remembering.

---

### 2.2 News sentiment trading strategy (143 votes)

**Data:** yfinance (12 tickers), RSS/HTML headlines, expert labels, custom financial lexicon.

**Models:** Classical ML on **mean SpaCy token vectors** (LR, KNN, DT, SVM, RF); separate **Keras LSTM** on padded token ids for **binary sentiment** (not direct price regression).

#### LSTM text classifier (Mermaid)

```mermaid
flowchart TB
  subgraph textBranch [Keras sequence branch]
    direction TB
    tok[Tokenizer num_words 20000]
    seq[texts_to_sequences]
    pad[pad_sequences maxlen 50]
    emb[Embedding 20000 x 100 input_length 50]
    lstm[LSTM 100 dropout 0.2 recurrent_dropout 0.2]
    dense[Dense 1 sigmoid]
    tok --> seq --> pad --> emb --> lstm --> dense
  end

  subgraph classicalBranch [Sklearn on doc vectors]
    direction TB
    spacy[Mean_SpaCy_token_vectors]
    clf[LogReg KNN DT SVM RF]
    spacy --> clf
  end

  dense --> sentimentLabel[Predicted_sentiment_0_1]
  clf --> sentimentLabel2[Predicted_sentiment_class]
```

**Note:** `eventRet` (same-day return + previous-day return) is used for **scatter plots and correlation** vs sentiment, not as the LSTM target.

**Takeaways:** Custom VADER lexicon + expert labels + transfer to unlabeled headlines; random 90/10 split is **not** ideal for time series.

---

### 2.3 LSTM market + news (Two Sigma, 270 votes)

**Data:** Competition `market` + `news` tables; **no raw text** — numeric sentiment columns + rolling 10-day mean per asset.

**Input per timestep:** `StandardScaler` features = market numerics + calendar + **joined news numerics** (missing news → 0 after left join).

**Label:** `sign(returnsOpenNextMktres10)` → binary; loss: `binary_crossentropy`. Inference: `sigmoid * 2 - 1` for submission confidence.

#### Stacked LSTM (Mermaid)

```mermaid
flowchart TB
  subgraph inputPipe [Sequence input]
    direction TB
    win[Variable_length_window lookback 90 step 10]
    feats["Feature_vector per step market plus news numeric"]
    win --> feats
  end

  subgraph lstmCore [Keras Sequential]
    direction TB
    l1["LSTM 128 return_sequences=True input_shape None x F"]
    l2["LSTM 64 return_sequences=True"]
    l3["LSTM 32 return_sequences=False"]
    out["Dense 1 sigmoid"]
    l1 --> l2 --> l3 --> out
  end

  feats --> l1
  out --> post["Inference multiply by 2 subtract 1 maps to minus1 plus 1"]
```

**Callbacks:** `EarlyStopping(patience=5)`, `ReduceLROnPlateau(factor=0.1, patience=2)`.

**Takeaways:** Per-asset batches preserve time order; rolling news aggregates help sparse events; direction classification vs regression.

---

### 2.4 XGBoost + LSTM hybrid — Netflix (75 votes)

**Data:** NFLX OHLCV only. Isolation Forest removes outliers before features.

**Sequences:** length 10, 14 features per step → LSTM sees `(batch, 10, 14)`; XGBoost sees **flattened** `(batch, 140)`.

**Dual scalers:** one `MinMaxScaler` on **Close** (target), one on the 14 input columns.

#### Parallel branches + ensemble (Mermaid)

```mermaid
flowchart TB
  subgraph seqIn [Sequence construction]
    direction TB
    raw[OHLCV plus engineered features]
    scaleF[MinMaxScaler features]
    scaleY[MinMaxScaler close only target]
    jitter["Optional jitter duplicate batch noise 0.01"]
    raw --> scaleF
    raw --> scaleY
    scaleF --> win[Sliding_windows len 10]
    win --> jitter
  end

  subgraph lstmBranch [LSTM branch]
    direction TB
    l1["LSTM 64 return_sequences=True"]
    bn1[BatchNorm]
    dr1[Dropout 0.2]
    l2["LSTM 32"]
    bn2[BatchNorm]
    dr2[Dropout 0.2]
    d1["Dense 32 relu"]
    d2[Dense 1 linear]
    l1 --> bn1 --> dr1 --> l2 --> bn2 --> dr2 --> d1 --> d2
  end

  subgraph xgbBranch [XGBoost branch]
    direction TB
    flat[Flatten seq to 140 dims]
    xgb[XGBRegressor trees depth 5 etc]
    opt[Optuna optional hyperparam search]
    flat --> xgb --> opt
  end

  jitter --> l1
  win --> flat
  d2 --> blend["y_hat equals 0.6 LSTM plus 0.4 XGB"]
  opt --> blend
```

**Takeaways:** 0.6/0.4 weighting; jitter + dual scaling; Optuna tuned on test in original notebook (methodological caveat).

---

### 2.5 Tesla multi-step stacked LSTM (151 votes)

**Data:** yfinance TSLA; **only** scaled OHLCV (5 channels) — no extra indicators.

**Two strategies in one repo:**

1. **Vanilla:** single-step LSTM, **recursive** multi-step at inference (roll window forward).
2. **Direct multi-step:** one forward pass outputs `n_steps_out * 5` values (e.g. 10 days × 5 features).

#### Vanilla single-step LSTM (Mermaid)

```mermaid
flowchart LR
  subgraph vanilla [Single-step predictor]
    direction TB
    inV["Input shape batch x 32 x 5 scaled OHLCV"]
    lV["LSTM 80"]
    drV[Dropout 0.05]
    dV["Dense 5 linear one step all features"]
    inV --> lV --> drV --> dV
  end
```

#### Direct multi-step stacked LSTM (Mermaid)

```mermaid
flowchart TB
  subgraph multiStep [Direct multi-horizon]
    direction TB
    inM["Input batch x 252 x 5 one year lookback"]
    lM1["LSTM 40 return_sequences=True orthogonal init"]
    lM2["LSTM 20 return_sequences=False"]
    dM["Dense n_steps_out times 5 linear vectorized horizon"]
    reshape[Reshape to steps x5 for plotting]
    inM --> lM1 --> lM2 --> dM --> reshape
  end
```

**Training:** multi-step uses MSE on full vector; batch size 1024; vanilla uses MAE.

**Takeaways:** Long lookback (252) for multi-step; direct output avoids error compounding vs recursion; orthogonal init on first LSTM.

---

## 3. Cross-notebook comparison

### 3.1 NLP / sentiment

| Approach | Used by | Strengths | Weaknesses |
|----------|---------|-----------|------------|
| VADER vanilla | Notebook 1 | Fast, no training | Weak on finance jargon |
| VADER + custom lexicon | Notebook 2 | Domain-tuned | Still lexicon-based |
| TextBlob | Notebook 2 | Trivial API | Low quality |
| SpaCy mean vectors + sklearn | Notebook 2 | Semantic | Needs labels for best results |
| Pre-computed numeric news | Notebook 3 | Scalable | No raw-text nuance |
| **FinBERT (our pipeline)** | — | Context, finance pre-training | Heavier compute |

### 3.2 Model types vs our adoption

| Type | Notebook | Best for | Our plan |
|------|----------|----------|----------|
| GAN | 1 | Synthetic sequences | Skip at our scale |
| Classical on embeddings | 2 | Sentiment classification | Reference only |
| 3-layer LSTM classifier | 3 | Direction | Smaller LSTM variant |
| LSTM + XGBoost | 4 | Tabular + temporal | **Primary hybrid** |
| Stacked multi-step LSTM | 5 | Multi-day horizon | Optional extension |

### 3.3 Feature engineering matrix

(Same as prior doc: MA, RSI, MACD, Bollinger, momentum, volatility, calendar, cross-ticker, sentiment, embeddings, news volume — our pipeline aims to cover the union.)

### 3.4 Validation

Walk-forward validation is **not** used in these five notebooks; it remains the recommended approach for our time-series setup.

---

## 4. Techniques to adopt

- **Dual scalers** — separate target vs feature `MinMaxScaler`.
- **Jitter** — Gaussian noise on LSTM inputs for small *n*.
- **Outliers** — Isolation Forest and/or quantile clipping.
- **Rolling news** — 5–10 day mean of sentiment-derived features.
- **Ensemble** — start `0.6 * LSTM + 0.4 * XGB`, tune on validation only.
- **Per-asset batches** — avoid shuffling unrelated tickers inside one LSTM batch.
- **SHAP** — `TreeExplainer` on boosting models.
- **Orthogonal init** — first LSTM layer when stacking.

---

## 5. How our approach differs

| Aspect | Typical Kaggle pattern | Ours |
|--------|------------------------|------|
| Sentiment | VADER / rules | FinBERT + sentence embeddings |
| Validation | Hold-out or random split | Walk-forward |
| Features | Few indicators | Broad technical + NLP + cross-ticker |
| News timing | Date-only join | Hour-aware mapping to trading day |

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Small *n* per ticker | Regularization, jitter, ensemble, walk-forward |
| Hard naive baseline | Report directional accuracy + ablations |
| Noisy news | Rolling sentiment, `has_news`, FinBERT |
| Leakage | Strict time ordering; tune on val not test |
| Slow FinBERT | Batch inference, cache once |

---

## 7. Our target pipeline (Mermaid)

```mermaid
flowchart TB
  subgraph dataLayer [Data]
    price[price_csv]
    news[news_csv]
  end

  subgraph featLayer [Features]
    tech[Technical_and_cross_ticker]
    nlp[FinBERT_and_embeddings]
    roll[Rolling_news_agg]
  end

  subgraph splitLayer [Validation]
    wf[Walk_forward_split]
    dual[Dual_MinMax_scalers]
  end

  subgraph modelLayer [Models]
    xgb[XGBoost_or_LightGBM]
    lstm[LSTM_stack]
  end

  subgraph outLayer [Output]
    ens["Ensemble weighted blend"]
    eval[RMSE_MAE_direction_ablation]
  end

  price --> tech
  news --> nlp --> roll
  tech --> wf
  roll --> wf
  wf --> dual
  dual --> xgb
  dual --> lstm
  xgb --> ens
  lstm --> ens
  ens --> eval
```

---

## 8. References

1. Yukhymenko, H. (2022). *Stock Prediction GAN + Twitter Sentiment Analysis*. Kaggle. https://www.kaggle.com/code/equinxx/stock-prediction-gan-twitter-sentiment-analysis  
2. Shtrauss, A. (2025). *News Sentiment Based Trading Strategy*. Kaggle. https://www.kaggle.com/code/shtrausslearning/news-sentiment-based-trading-strategy  
3. Pukhov, D. (2018). *EDA and LSTM with Generator for Market and News*. Kaggle. https://www.kaggle.com/code/dmitrypukhov/eda-and-lstm-with-generator-for-market-and-news  
4. Cifci, A. (2025). *XGBoost LSTM for Netflix Stock*. Kaggle. https://www.kaggle.com/code/mehmetakifciftci/xgboost-lstm-for-netflix-stock  
5. GusLovesMath. (2024). *Tesla Stock Forecasting Multi-Step Stacked LSTM*. Kaggle. https://www.kaggle.com/code/guslovesmath/tesla-stock-forecasting-multi-step-stacked-lstm  
6. Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models*. arXiv:1908.10063.  
7. Oliveira, N. et al. (2017). *The impact of microblogging data for stock market prediction*. Expert Systems with Applications.
