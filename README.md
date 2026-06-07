# crowd-conviction-gate-calibration

Calibration suite for the crowd-conviction entry gate on Polymarket: FP analysis, policy selection, threshold sweeps.

## Why it exists

The live engine runs a **crowd-conviction one-sided gate**: when one side of a 5-minute crypto up/down market is cheap, the crowd backs the expensive side, and conviction is high, the bot *blocks* fires on the cheap side. That gate trades recall (savings on losing cheap fires) against precision (missed wins when the cheap side actually triumphs). These scripts are the offline lab that picks the gate's thresholds, audits its false positives, and validates that any change clears statistical significance before it ships to the bot — all from recorded market history, never on the WS hot path.

## What's inside

Four standalone, read-only analyzers. Each loads recorded JSONL, computes Wilson CIs / Mann-Whitney / binomial / paired t-tests, prints a console report, and writes a JSON summary.

| Script | Purpose |
| --- | --- |
| `crowd_gate_analysis_vps.py` | Defines and validates the base gate (`loser_bid < X` ∧ crowd backs winner ∧ `conv ≥ Y`). Sweeps the `(loser_bid, conv)` grid for precision/recall/F1, runs Mann-Whitney signal-power tests, and reports the UP/DN-symmetric breakdown. Needs `--history` only. |
| `crowd_gate_fp_analysis_vps.py` | Isolates **false positives** (gate triggers but the cheap side *wins*). Tests every real-time signal to separate TP from FP, sweeps compound `lb × conv × abs_bn` conditions, and simulates the PnL recoverable from perfectly exempting FP markets. Needs `--history` + `--recap`. |
| `crowd_gate_policy_vps.py` | Compares nine candidate gate policies (no-gate baseline → cheap-bid + conv + |BN| compounds) using **zero-future-bias** features only. Per-market paired t-tests vs the current policy, block-precision (win-rate-lose), and a sample-size estimate for Bonferroni confirmation. Needs `--history` + `--recap`. |
| `crowd_gate_threshold_vps.py` | Full per-fire threshold sweep ($0.10–$1.00) across **all** live + shadow strategies. Compares dn_live-only, crowd-only, both, and no-gate PnL; paired t-test on `both vs current`; per-strategy breakdown at the optimal $0.45 cut. Needs `--history` + `--recap`. |

The "real-time signal" discipline is enforced throughout: gate conditions use only the cheap side's live bid/ask, crowd conviction, and `|BN|` delta — never the winner/loser label — so backtest results transfer to production without look-ahead bias.

## Requirements

- Python 3.8+
- `scipy` (optional but recommended — without it all significance tests degrade to `p=1.0`)
- No other third-party deps; everything else is stdlib (`json`, `math`, `argparse`).
- **No wallet, key, or network access.** These are pure offline analyzers and never touch funds, the CLOB, or web3.

## Usage

```bash
# Base gate definition + signal power (history only)
python3 crowd_gate_analysis_vps.py \
    --history /path/to/polymarket-data/market_history.jsonl \
    --out     crowd_gate_report.json

# False-positive audit (needs recap for per-fire PnL)
python3 crowd_gate_fp_analysis_vps.py \
    --history /path/to/polymarket-data/market_history.jsonl \
    --recap   /path/to/polymarket-data/market_recap_history.jsonl \
    --out     crowd_gate_fp_report.json

# Policy comparison (zero future bias)
python3 crowd_gate_policy_vps.py \
    --history /path/to/polymarket-data/market_history.jsonl \
    --recap   /path/to/polymarket-data/market_recap_history.jsonl \
    --out     crowd_gate_policy_report.json

# Threshold sweep across all strategies
python3 crowd_gate_threshold_vps.py \
    --history /path/to/polymarket-data/market_history.jsonl \
    --recap   /path/to/polymarket-data/market_recap_history.jsonl \
    --out     crowd_gate_threshold_report.json
```

## Data

All scripts are read-only consumers of two JSONL datasets from the private **`polymarket-data`** repo:

- `market_history.jsonl` — per-market features, including the `skip_counterfactuals` T-200…T-30 snapshots (crowd side/conviction, winner ask, loser bid, BN delta) and `market_wide` stats.
- `market_recap_history.jsonl` — per-fire records (`side`, `entry_price`, `hypo_pnl`, gate-hold flags) used for PnL simulation.

Point `--history` / `--recap` at your local checkout of that repo. No data files are committed here (see `.gitignore`).

> Private research software. No warranty; trades/handles real funds at your own risk.
