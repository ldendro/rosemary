# Rosemary — Phase 1  
**Exploratory Research & Learning Phase**
**Link to confluence -> https://livanos.atlassian.net/wiki/spaces/~71202004476954f6b9455a9e921237c4d2aa1b/pages/65709/Phase+1**
---

## Overview

Phase 1 represents the **initial exploratory phase** in the development of *Rosemary*.  
The purpose of this phase is **learning, experimentation, and conceptual grounding**, not the creation of a production-ready trading system.

The files in this phase are primarily educational and **do not provide algorithmic or investment value**. Several components will be refactored or re-implemented in Phase 2, where the system is rebuilt with stronger abstractions and clearer separation of concerns.

This phase exists to understand *how* and *why* strategies behave — not to optimize performance.

---

## Purpose of This Phase

Phase 1 is designed to answer foundational questions such as:

- How do simple trend-following and mean-reversion strategies behave across time?
- When and why do strategies structurally fail?
- How should walk-forward testing be performed correctly?
- Which evaluation metrics matter most for robustness?
- What abstractions are worth preserving before scaling complexity?

The focus is on **insight generation**, not profitability.

---

## File Structure Philosophy

Each major file in Phase 1 follows a consistent learning-oriented structure:

1. **High-level description**  
   What the file does and why it exists.

2. **Key takeaway**  
   The most important market or trading insight derived from the file.

3. **Step-by-step code analysis**  
   A guided walkthrough explaining the logic and purpose of each section.

4. **Key insights & evaluation**  
   Observations on performance, failure modes, and regime sensitivity.

5. **Vocabulary section**  
   Definitions of important financial, statistical, and modeling terms.

---

## Important Disclaimer

> **Phase 1 is not a production or alpha-generating pipeline.**

This phase intentionally omits:

- Capital allocation logic
- Risk management frameworks
- Portfolio construction
- Execution modeling
- Slippage, fees, or live constraints

Any apparent performance should be interpreted **strictly as educational output**.

---

## How to Run Phase 1

Below is the intended execution flow for running experiments in Phase 1.

### 1. Data Ingestion

Fetch daily historical market data.

```bash
python scripts/fetch_daily.py
```

### 2. Build Features

Compute engineered features used by the models and backtests; writes
`data/curated/features_daily.parquet`.

```bash
python scripts/build_features.py
```

### 3. (Optional) Train Baseline Model

Train a simple linear baseline on the generated features and print
basic metrics (MSE, R², coefficients). Useful as a quick smoke test.

```bash
python scripts/train_baseline_model.py
```

### 4. Run Backtests (in order)

Run the provided toy backtests in the order below. These are intentionally
simple evaluations to validate alignment, signal logic, and PnL wiring.

- Baseline (simple long-if-positive prediction)

```bash
python scripts/backtest_baseline.py
```

- Baseline with transaction costs and simple sizing

```bash
python scripts/backtest_baseline_costs.py
```

- Baseline with minimum holding period (threshold + hold days)

```bash
python scripts/backtest_baseline_hold.py
```

- Baseline with trend filter (e.g., require ret_20d > 0)

```bash
python scripts/backtest_baseline_trend.py
```

- Baseline with trend + volatility filter

```bash
python scripts/backtest_baseline_trend_vol.py
```

- Mean-reversion rule-based backtest

```bash
python scripts/backtest_mean_reversion.py
```

- Walk-forward evaluation for the baseline/trend strategy

```bash
python scripts/backtest_walkforward.py
```

- Walk-forward evaluation for the mean-reversion strategy

```bash
python scripts/backtest_walkforward_meanrev.py
```

### Notes & Environment

- Run these commands from the `Phase 1` directory (this README's folder). The
   commands use the `scripts/` relative path.
- Activate your project's virtualenv before running (example):

```bash
source ../.venv/bin/activate
pip install -r ../requirements.txt
```

- If a script raises `FileNotFoundError` for `features_daily.parquet`, re-run
   `build_features.py` and confirm the path `data/curated/features_daily.parquet` exists.

- The backtests are educational and do not include full execution assumptions
   (market impact, slippage, advanced risk management). Interpret outputs as
   diagnostics rather than deployable signals.
