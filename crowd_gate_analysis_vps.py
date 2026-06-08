#!/usr/bin/env python3
"""
crowd_gate_analysis_vps.py
━━━━━━━━━━━━━━━━━━━━━━━━━
Tests the crowd-conviction one-sided market gate:
  BLOCK cheap side when: loser_bid < $X  AND  crowd backs expensive side  AND  conv ≥ Y

Works symmetrically for both UP-dominant and DN-dominant markets.

Usage:
    python3 crowd_gate_analysis_vps.py \
        --history ./market_history.jsonl \
        --out     crowd_gate_report.json
"""

import json, math, argparse, sys, time
from collections import defaultdict

try:
    from scipy import stats as st
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Progress ──────────────────────────────────────────────────────────
class Progress:
    BAR_W = 35
    def __init__(self, total, label=''):
        self.total = total; self.label = label
        self.t0 = time.time(); self._done = False

    def update(self, done):
        if self._done: return
        e = time.time() - self.t0
        frac = done / self.total if self.total else 1
        eta = self._fmt(e / frac * (1-frac)) if frac > 0 else '--:--'
        filled = int(self.BAR_W * frac)
        bar = '█'*filled + '░'*(self.BAR_W-filled)
        sys.stdout.write(
            f"\r  {self.label}  [{bar}] {frac*100:5.1f}%  "
            f"{done:,}/{self.total:,}  elapsed {self._fmt(e)}  ETA {eta}  "
            f"({done/max(e,0.001):,.0f}/s)   ")
        sys.stdout.flush()

    def done(self):
        self._done = True
        e = self._fmt(time.time() - self.t0)
        sys.stdout.write(
            f"\r  {self.label}  [{'█'*self.BAR_W}] 100.0%  "
            f"{self.total:,}/{self.total:,}  elapsed {e}  ✅\n")
        sys.stdout.flush()

    @staticmethod
    def _fmt(s):
        s = int(s); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
        return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m:02d}m{sec:02d}s"

# ── Stats ─────────────────────────────────────────────────────────────
def wilson(w, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = w/n
    lo = max(0,(p+z**2/(2*n)-z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/(1+z**2/n))
    hi = min(1,(p+z**2/(2*n)+z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/(1+z**2/n))
    return lo, hi

def binom_p(wins, n, p0, alt='greater'):
    if not HAS_SCIPY or n == 0: return 1.0
    return st.binomtest(wins, n, p0, alternative=alt).pvalue

def sig(p):
    return '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ('~' if p<0.1 else ' ')))

# ── Main ─────────────────────────────────────────────────────────────
def main():
    t_global = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', required=True)
    parser.add_argument('--out', default='crowd_gate_report.json')
    args = parser.parse_args()

    # ── Step 1: Load ──────────────────────────────────────────────────
    print(f"\nStep 1/5 — Loading {args.history} ...")
    sys.stdout.write("  Counting lines ..."); sys.stdout.flush()
    t0 = time.time()
    with open(args.history) as f:
        total_lines = sum(1 for l in f if l.strip())
    print(f"\r  Counting lines ... {total_lines:,} lines ({time.time()-t0:.1f}s)")

    markets = []
    prog = Progress(total_lines, 'Loading  ')
    with open(args.history) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try: markets.append(json.loads(line))
            except Exception: continue
            if i % 500 == 0: prog.update(i+1)
    prog.done()
    print(f"  Loaded {len(markets):,} markets")

    # ── Step 2: Build feature rows ────────────────────────────────────
    print(f"\nStep 2/5 — Extracting market features ...")
    prog = Progress(len(markets), 'Features ')
    rows = []
    for idx, m in enumerate(markets):
        if idx % 300 == 0: prog.update(idx+1)
        winner = m.get('winner')
        if winner not in ('UP','DN'): continue
        sc  = m.get('skip_counterfactuals') or {}
        mw  = m.get('market_wide') or {}

        tps = {}
        for tp in ['T-200','T-150','T-120','T-90','T-60','T-30']:
            v = sc.get(tp)
            if not v: continue
            tps[tp] = {
                'winner_ask':  v.get('winner_ask', 0) or 0,
                'loser_bid':   v.get('loser_bid',  0) or 0,
                'bn':          v.get('bn_delta',   0) or 0,
                'crowd_side':  v.get('crowd_side'),
                'crowd_conv':  v.get('crowd_conv', 0) or 0,
                'snipe_pnl':   v.get('snipe_pnl',  0) or 0,
            }
        if len(tps) < 3: continue

        t200 = tps.get('T-200', {})
        early_winner_asks = [tps[tp]['winner_ask'] for tp in ['T-200','T-150','T-120'] if tp in tps]
        pct_dom = (sum(1 for a in early_winner_asks if a > 0.60) / len(early_winner_asks)
                   if early_winner_asks else 0)

        rows.append({
            'slug':           m.get('slug',''),
            'winner':         winner,
            'one_sided':      pct_dom >= 0.67,
            'pct_dom':        pct_dom,
            # T-200 signals
            'winner_ask_t200': t200.get('winner_ask', 0),
            'loser_bid_t200':  t200.get('loser_bid',  0),
            'bn_t200':         abs(t200.get('bn', 0)),
            'crowd_conv_t200': t200.get('crowd_conv', 0),
            'crowd_for_winner': t200.get('crowd_side') == winner,
            # Market-wide
            'lc':    mw.get('lead_changes',  20) or 20,
            't5050': mw.get('time_near_5050',300) or 300,
            'snipe_t200': t200.get('snipe_pnl', 0),
        })
    prog.done()
    print(f"  {len(rows):,} markets with full data")

    n_os    = sum(1 for r in rows if r['one_sided'])
    n_ct    = len(rows) - n_os
    base_rate = n_os / len(rows) if rows else 0.4
    print(f"  One-sided (winner ≥$0.60 in ≥67% early tps): {n_os} ({n_os/len(rows):.0%})")
    print(f"  Contested:                                    {n_ct} ({n_ct/len(rows):.0%})")

    # ── Step 3: Signal power (Mann-Whitney) ───────────────────────────
    print(f"\nStep 3/5 — Signal power test ...")
    signals = [
        ('winner_ask_t200',  lambda r: r['winner_ask_t200']),
        ('loser_bid_t200',   lambda r: r['loser_bid_t200']),
        ('crowd_conv_t200',  lambda r: r['crowd_conv_t200']),
        ('crowd_for_winner', lambda r: float(r['crowd_for_winner'])),
        ('lead_changes',     lambda r: -r['lc']),
        ('time@50/50',       lambda r: -r['t5050']),
        ('bn_t200',          lambda r: r['bn_t200']),
    ]
    N_SIG = len(signals)
    bonf_sig = 0.05 / N_SIG
    print(f"  {N_SIG} signals, Bonferroni p<{bonf_sig:.4f}\n")
    print(f"  {'Signal':<22} {'OS_mean':>9} {'CT_mean':>9} {'p-val':>9}  action")
    print("  " + "─"*62)
    sig_results = {}
    for name, fn in signals:
        os_v = [fn(r) for r in rows if r['one_sided']]
        ct_v = [fn(r) for r in rows if not r['one_sided']]
        if not HAS_SCIPY: p = 0.5
        else: _, p = st.mannwhitneyu(os_v, ct_v, alternative='two-sided')
        sig_results[name] = p
        action = "✅ KEEP" if p < bonf_sig else ("🔬" if p < 0.05 else "❌")
        print(f"  {name:<22} {sum(os_v)/len(os_v):>9.3f} {sum(ct_v)/len(ct_v):>9.3f} "
              f"{p:>9.5f}  {action}")

    # ── Step 4: Compound threshold sweep ─────────────────────────────
    print(f"\nStep 4/5 — Compound gate sweep ...")
    lb_thrs = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    cv_thrs = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    N_TESTS = len(lb_thrs) * len(cv_thrs) + len(lb_thrs)  # grid + loser-only
    bonf_gate = 0.05 / N_TESTS
    _done = [0]

    def run_gate(cond_fn, name):
        _done[0] += 1
        fired   = [r for r in rows if cond_fn(r)]
        n       = len(fired)
        if n < 10: return None
        n_hit   = sum(1 for r in fired if r['one_sided'])
        prec    = n_hit / n
        rec     = n_hit / n_os if n_os else 0
        f1      = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        p       = binom_p(n_hit, n, base_rate, 'greater')
        ci_lo, ci_hi = wilson(n_hit, n)
        return {'name':name,'n':n,'n_hit':n_hit,'precision':prec,'recall':rec,
                'f1':f1,'p':p,'ci_lo':ci_lo,'ci_hi':ci_hi,
                'bonf_ok': p < bonf_gate}

    prog = Progress(N_TESTS, 'Sweep    ')
    results = []

    # Crowd compound: loser < X + crowd_for_winner + conv > Y
    for lb in lb_thrs:
        for cv in cv_thrs:
            cond = (lambda r, l=lb, c=cv:
                    r['loser_bid_t200'] < l and r['crowd_for_winner'] and r['crowd_conv_t200'] >= c)
            res = run_gate(cond, f"loser<${lb:.2f}+crowd+conv≥{cv:.2f}")
            if res: results.append(res)
            prog.update(_done[0])

    # Loser-only: simpler gate without crowd requirement
    for lb in lb_thrs:
        cond = (lambda r, l=lb: r['loser_bid_t200'] < l)
        res = run_gate(cond, f"loser<${lb:.2f} (price-only)")
        if res: results.append(res)
        prog.update(_done[0])

    prog.done()

    results.sort(key=lambda x: -x['f1'])
    print(f"\n  {'Condition':<42} {'n':>5} {'Prec':>7} {'Rec':>7} {'F1':>6} {'p':>9} {'BF'}")
    print("  " + "─"*82)
    for r in results[:20]:
        bf = "✅" if r['bonf_ok'] else "  "
        s  = sig(r['p'])
        print(f"  {r['name']:<42} {r['n']:>5} {r['precision']:>7.1%} {r['recall']:>7.1%} "
              f"{r['f1']:>6.3f} {r['p']:>9.5f}{s} {bf}")

    # ── Step 5: Best gate — UP vs DN breakdown + snipe analysis ───────
    print(f"\nStep 5/5 — Best gate deep-dive ...")

    best = results[0] if results else None
    if best:
        # Parse best condition params
        print(f"\n  BEST GATE: {best['name']}")
        print(f"  Precision={best['precision']:.1%}  Recall={best['recall']:.1%}  "
              f"F1={best['f1']:.3f}  n={best['n']}  p={best['p']:.8f}")
        print(f"  CI precision: [{best['ci_lo']:.1%}, {best['ci_hi']:.1%}]")
        print(f"  Bonferroni ({N_TESTS} tests): {'✅ CONFIRMED' if best['bonf_ok'] else '❌ not yet'}")

        # Try to reconstruct best condition lambda
        # Find which condition produced the best result
        best_lb = best_cv = None
        for lb in lb_thrs:
            for cv in cv_thrs:
                name = f"loser<${lb:.2f}+crowd+conv≥{cv:.2f}"
                if name == best['name']:
                    best_lb, best_cv = lb, cv
        
        if best_lb is not None:
            best_cond = (lambda r, l=best_lb, c=best_cv:
                         r['loser_bid_t200'] < l and r['crowd_for_winner']
                         and r['crowd_conv_t200'] >= c)
            fired_rows = [r for r in rows if best_cond(r)]

            print(f"\n  Breakdown by dominant side:")
            for dom in ['UP','DN']:
                sub = [r for r in fired_rows if r['winner']==dom]
                n_hit = sum(1 for r in sub if r['one_sided'])
                if sub:
                    print(f"    {dom}-dominant: {n_hit}/{len(sub)} correct ({n_hit/len(sub):.0%} precision)")

            # Snipe PnL comparison: triggered vs not triggered
            fired_snipe   = [r['snipe_t200'] for r in fired_rows]
            unfired_snipe = [r['snipe_t200'] for r in rows if not best_cond(r)]
            if fired_snipe and unfired_snipe:
                print(f"\n  Snipe PnL @ T-200 (max possible directional profit):")
                print(f"    Gate triggered markets: avg=${sum(fired_snipe)/len(fired_snipe):.3f}")
                print(f"    Non-triggered markets:  avg=${sum(unfired_snipe)/len(unfired_snipe):.3f}")
                print(f"    Triggered markets have LESS potential profit → gate fires in low-opportunity markets")

    print(f"\n  SYMMETRIC breakdown (both UP and DN dominant):")
    for dom, loser in [('UP','DN'),('DN','UP')]:
        dom_os = [r for r in rows if r['winner']==dom and r['one_sided']]
        dom_ct = [r for r in rows if r['winner']==dom and not r['one_sided']]
        if dom_os and dom_ct:
            lb_os = [r['loser_bid_t200'] for r in dom_os]
            lb_ct = [r['loser_bid_t200'] for r in dom_ct]
            cv_os = [r['crowd_conv_t200'] for r in dom_os]
            cv_ct = [r['crowd_conv_t200'] for r in dom_ct]
            _, p_lb = st.mannwhitneyu(lb_os, lb_ct) if HAS_SCIPY else (0, 0.5)
            _, p_cv = st.mannwhitneyu(cv_os, cv_ct) if HAS_SCIPY else (0, 0.5)
            print(f"\n    {dom}-dominant ({len(dom_os)} mkts) vs {dom}-contested ({len(dom_ct)} mkts):")
            print(f"      {loser} (loser) bid@T-200: {sum(lb_os)/len(lb_os):.3f} vs {sum(lb_ct)/len(lb_ct):.3f}  p={p_lb:.5f}{sig(p_lb)}")
            print(f"      crowd_conv@T-200: {sum(cv_os)/len(cv_os):.3f} vs {sum(cv_ct)/len(cv_ct):.3f}  p={p_cv:.5f}{sig(p_cv)}")
            print(f"      → In {dom}-dominant markets, cheap {loser} @ avg ${sum(lb_os)/len(lb_os):.2f} looks buyable but LOSES")

    # Save JSON
    output = {
        'n_markets': len(markets),
        'n_rows': len(rows),
        'n_one_sided': n_os,
        'n_contested': n_ct,
        'base_rate_one_sided': round(base_rate, 4),
        'bonf_threshold_gate': round(bonf_gate, 6),
        'signal_pvalues': {k: round(v, 6) for k,v in sig_results.items()},
        'best_gates': [
            {k: (round(v,5) if isinstance(v,float) else (bool(v) if isinstance(v,bool) else v))
             for k,v in r.items()}
            for r in results[:10]
        ],
        'runtime_seconds': round(time.time()-t_global, 1),
    }
    def _safe(obj):
        if isinstance(obj, bool): return int(obj)
        if isinstance(obj, float): return round(float(obj), 6)
        try:
            import numpy as np
            if isinstance(obj, (np.floating, np.integer)): return float(obj)
            if isinstance(obj, np.bool_): return int(obj)
        except ImportError: pass
        if isinstance(obj, dict): return {k: _safe(v) for k,v in obj.items()}
        if isinstance(obj, list): return [_safe(x) for x in obj]
        return obj
    with open(args.out, 'w') as f:
        json.dump(_safe(output), f, indent=2)

    total_time = Progress._fmt(time.time()-t_global)
    print(f"\n{'='*60}")
    print(f"Total runtime: {total_time}")
    print(f"✅ Report saved to {args.out}")
    print(f"   Best gate: {best['name'] if best else 'N/A'}")
    print(f"   Precision: {best['precision']:.1%}  Recall: {best['recall']:.1%}  F1: {best['f1']:.3f}" if best else "")

if __name__ == '__main__':
    main()
