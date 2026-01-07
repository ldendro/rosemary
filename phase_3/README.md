# Phase 3 — Regime Control, Strategy Eligibility, and Allocation Robustness

## Overview

**Phase 3** focuses on the **control plane** of the trading system rather than signal generation or portfolio construction.

Where Phase 2 established that the Trend and Mean-Reversion strategies produce valid walk-forward signal streams, Phase 3 investigates **how those signals should be governed**:

- When strategies should be allowed to operate at all
- How regime transitions should be handled
- How allocation behavior changes under uncertainty versus regime dominance

This phase is explicitly **architectural and diagnostic**.  
It is not intended to add new alpha, but to understand and improve **system behavior under stress, transitions, and volatility regimes**.

---

## Phase 3 Goals

Phase 3 answers the following questions:

1. Does smoothing regime transitions (soft allocation) improve risk-adjusted outcomes?
2. Does state persistence (hysteresis) reduce whipsaw and instability?
3. Is suppressing strategies in hostile environments more effective than blending them?
4. Which control mechanisms materially improve equity, and which only improve stability?
5. How should regime control be separated from signal logic going forward?

This phase is about **control correctness**, not optimization.

---

## Core Mechanisms Evaluated

Phase 3 introduces and evaluates three system-level mechanisms:

### Soft Meta Allocator
- Replaces hard regime switches with confidence-weighted blending across states
- Designed to reduce drawdowns and smooth exposure
- Tested with and without power normalization

**Outcome:**  
Improved drawdown behavior and stability, but under-committed in regime-dominant markets, resulting in lower terminal equity versus the hard allocator.

---

### Hysteresis Allocator
- Introduces memory into regime decisions
- Requires stronger evidence to exit a state than to remain in one
- Prevents rapid regime flipping

**Outcome:**  
Improved interpretability and regime stability, but no material improvement in portfolio equity for the current signal set.

---

### Strategy Gates (Pre-Allocator)
- Enforces environmental eligibility constraints on strategies
- Prevents strategies from operating in structurally hostile regimes
- Independent of allocation logic

**Outcome:**  
Material improvement in equity outcomes by eliminating low-expectancy trades before allocation decisions occur.

---

## What Needs to Be Run (and What Does Not)

Phase 3 introduces **no new data ingest or validation scripts**.

All Phase 3 analysis is performed entirely through notebooks using:
- Phase 2 processed data
- Phase 3 allocator and gating logic

No scripts require manual execution.

---

## Notebook Execution and Validation

All Phase 3 functionality is evaluated through notebooks.  
Notebooks act as **controlled validation layers**, not exploratory analysis.

The notebook directory contains targeted tests for each new control mechanism.

---

### Notebook Inventory & Intent

#### Soft Allocator
- `01_soft_allocator_smoke_test.ipynb`  
  Validates basic correctness and behavior of the soft allocator

- `02_soft_allocator_walkforward.ipynb`  
  Evaluates walk-forward performance versus hard allocation

- `03_soft_allocator_concentration_sweep.ipynb`  
  Tests commitment behavior under power normalization

---

#### Hysteresis Allocator
- `04_hysteresis_allocator_smoke_test.ipynb`  
  Verifies state machine logic and transitions

- `05_hysteresis_walkforward_compare.ipynb`  
  Compares hysteresis allocation against baseline allocators

---

#### Strategy Gates
- `06_strategy_gates_smoke_test.ipynb`  
  Validates gate logic and environmental eligibility rules

- `07_gated_allocator_walkforward_compare.ipynb`  
  Evaluates portfolio impact of pre-allocator strategy suppression

---

### Notebook Guarantees

Each notebook:
- Is deterministic and reproducible
- Uses only Phase 2 processed data
- Evaluates one control mechanism in isolation or comparison
- Avoids parameter overfitting and optimization loops

---

## Phase 3 Outputs

Phase 3 produces:

- Comparative allocator performance diagnostics
- Regime transition stability analysis
- Strategy eligibility timelines
- Evidence separating allocation mechanics from strategy validity
- Clear identification of which controls materially improve equity

---

## Phase 3 Conclusions

Phase 3 demonstrates that:

- **Strategy suppression is more impactful than strategy blending** for the current signal set
- Allocation smoothness alone does not improve long-run returns
- Control layers must be evaluated independently from alpha logic
- Structural constraints outperform allocation refinements in hostile regimes

These findings directly inform the design direction for subsequent phases.

---

## Phase 3 Completion Criteria

Phase 3 is considered complete when:

- The role and limits of regime smoothing are understood
- Strategy eligibility is enforced independently of allocation
- Control logic is cleanly separated from signal logic
- Further improvements require changes to signals or higher-order controls, not allocation mechanics
