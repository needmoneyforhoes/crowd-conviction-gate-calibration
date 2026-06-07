# crowd-conviction-gate-calibration

Offline calibration scripts for the crowd-conviction one-sided entry gate used by the Polymarket 5-minute crypto bot.

The gate blocks fires on the cheap side when the crowd backs the expensive side and conviction is high. These scripts pick the gate thresholds, audit its false positives, and check significance against recorded market history. Nothing here runs on the live WebSocket path.

## Scripts

| File | What it does |
| --- | --- |
| `crowd_gate_analysis_vps.py` | Defines the base gate (`loser_bid < X` and crowd backs winner and `conv >= Y`). Sweeps the `(loser_bid, conv)` grid for precision/recall/F1, runs Mann-Whitney signal tests, prints the UP/DN breakdown. Needs `--history`. |
| `crowd_gate_fp_analysis_vps.py` | Isolates false positives (gate triggers but the cheap side wins). Tests each real-time signal to split TP from FP, sweeps `lb x conv x abs_bn` compounds, and simulates PnL recoverable from exempting FP markets. Needs `--history` and `--recap`. |
| `crowd_gate_policy_vps.py` | Compares nine gate policies (no-gate baseline through cheap-bid + conv + abs_bn compounds) using zero-future-bias features. Per-market paired t-tests vs the current policy, block precision, and a Bonferroni sample-size estimate. Needs `--history` and `--recap`. |
| `crowd_gate_threshold_vps.py` | Per-fire threshold sweep ($0.10 to $0.60) across all live and shadow strategies. Compares dn_live-only, crowd-only, both, and no-gate PnL, with a paired t-test on `both vs current` and a per-strategy breakdown. Needs `--history` and `--recap`. |

Gate conditions use only the cheap side's live bid/ask, crowd conviction, and `abs(BN)` delta. They never reference the winner/loser label, so backtest results carry over without look-ahead bias.

## Requirements

Python 3.8+. Everything else is stdlib (`json`, `math`, `argparse`). `scipy` is optional; without it the significance tests return `p=1.0`.

## Usage

```bash
python3 crowd_gate_analysis_vps.py \
    --history $DATA_DIR/market_history.jsonl \
    --out     crowd_gate_report.json

python3 crowd_gate_fp_analysis_vps.py \
    --history $DATA_DIR/market_history.jsonl \
    --recap   $DATA_DIR/market_recap_history.jsonl \
    --out     crowd_gate_fp_report.json

python3 crowd_gate_policy_vps.py \
    --history $DATA_DIR/market_history.jsonl \
    --recap   $DATA_DIR/market_recap_history.jsonl \
    --out     crowd_gate_policy_report.json

python3 crowd_gate_threshold_vps.py \
    --history $DATA_DIR/market_history.jsonl \
    --recap   $DATA_DIR/market_recap_history.jsonl \
    --out     crowd_gate_threshold_report.json
```

## Data

Read-only; no credentials required. Both JSONL inputs come from the private `polymarket-data` repo; point `--history` and `--recap` at a local checkout. No data files are committed (see `.gitignore`).

- `market_history.jsonl`: per-market features, including the `skip_counterfactuals` T-200..T-30 snapshots (crowd side/conviction, winner ask, loser bid, BN delta) and `market_wide` stats.
- `market_recap_history.jsonl`: per-fire records (`side`, `entry_price`, `hypo_pnl`, gate-hold flags) for PnL simulation.
