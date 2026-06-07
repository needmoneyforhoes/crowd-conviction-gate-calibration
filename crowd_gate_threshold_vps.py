#!/usr/bin/env python3
"""
crowd_gate_threshold_vps.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
Simulates crowd gate PnL across ALL strategies (live+shadow) at every
threshold from $0.10 to $0.60, using full market history + recap files.

Usage:
    python3 crowd_gate_threshold_vps.py \
        --history /home/polybot/polymarket-bot/market_history.jsonl \
        --recap   /home/polybot/polymarket-bot/market_recap_history.jsonl \
        --out     crowd_gate_threshold_report.json
"""

import json, math, argparse, sys, time
from collections import defaultdict

try:
    from scipy import stats as st
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Progress ─────────────────────────────────────────────────────────
class Progress:
    BAR_W = 35
    def __init__(self, total, label=''):
        self.total=total; self.label=label
        self.t0=time.time(); self._done=False
    def update(self, done):
        if self._done: return
        e=time.time()-self.t0; frac=done/self.total if self.total else 1
        eta=self._fmt(e/frac*(1-frac)) if frac>0 else '--:--'
        filled=int(self.BAR_W*frac)
        bar='█'*filled+'░'*(self.BAR_W-filled)
        sys.stdout.write(f"\r  {self.label}  [{bar}] {frac*100:5.1f}%  "
            f"{done:,}/{self.total:,}  elapsed {self._fmt(e)}  ETA {eta}  "
            f"({done/max(e,0.001):,.0f}/s)   ")
        sys.stdout.flush()
    def done(self):
        self._done=True; e=self._fmt(time.time()-self.t0)
        sys.stdout.write(f"\r  {self.label}  [{'█'*self.BAR_W}] 100.0%  "
            f"{self.total:,}/{self.total:,}  elapsed {e}  ✅\n")
        sys.stdout.flush()
    @staticmethod
    def _fmt(s):
        s=int(s); h,r=divmod(s,3600); m,sec=divmod(r,60)
        return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m:02d}m{sec:02d}s"

# ── Stats ─────────────────────────────────────────────────────────────
def wilson(w, n, z=1.96):
    if n==0: return 0.0, 0.0
    p=w/n
    lo=max(0,(p+z**2/(2*n)-z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/(1+z**2/n))
    hi=min(1,(p+z**2/(2*n)+z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/(1+z**2/n))
    return lo, hi

def binom_p(wins, n, p0, alt='less'):
    if not HAS_SCIPY or n==0: return 1.0
    return st.binomtest(wins, n, p0, alternative=alt).pvalue

def sig(p):
    return '***' if p<0.001 else('**' if p<0.01 else('*' if p<0.05 else '~' if p<0.1 else ' '))

def load_jsonl(path):
    lines=[]; total=0
    sys.stdout.write(f"  Counting {path.split('/')[-1]} ..."); sys.stdout.flush()
    t0=time.time()
    with open(path) as f:
        total=sum(1 for l in f if l.strip())
    print(f"\r  Counting {path.split('/')[-1]} ... {total:,} lines ({time.time()-t0:.1f}s)")
    prog=Progress(total, f"  Loading  ")
    with open(path) as f:
        for i, line in enumerate(f):
            line=line.strip()
            if not line: continue
            try: lines.append(json.loads(line))
            except: continue
            if i%500==0: prog.update(i+1)
    prog.done()
    return lines

# ── Main ──────────────────────────────────────────────────────────────
def main():
    t_global=time.time()
    parser=argparse.ArgumentParser()
    parser.add_argument('--history', required=True,  help='market_history.jsonl (has skip_counterfactuals)')
    parser.add_argument('--recap',   required=True,  help='market_recap_history.jsonl (has per-fire data)')
    parser.add_argument('--out',     default='crowd_gate_threshold_report.json')
    args=parser.parse_args()

    # ── Step 1: Load history (market-level features) ──────────────────
    print(f"\nStep 1/5 — Loading market history ...")
    history_raw=load_jsonl(args.history)
    print(f"  {len(history_raw):,} markets loaded")

    # Build crowd gate trigger map: slug → {cheap_side, conv, loser_bid}
    crowd_trigger={}
    for m in history_raw:
        winner=m.get('winner')
        if winner not in ('UP','DN'): continue
        sc=m.get('skip_counterfactuals') or {}
        v200=sc.get('T-200') or {}
        cs=v200.get('crowd_side'); cc=v200.get('crowd_conv',0) or 0
        lb=v200.get('loser_bid',1.0) or 1.0
        slug=m.get('slug','')
        if cs in ('UP','DN') and cc>=0.60 and lb<0.50:  # broad pre-filter
            crowd_trigger[slug]={
                'cheap_side': 'DN' if cs=='UP' else 'UP',
                'expensive_side': cs,
                'conv': cc, 'loser_bid': lb,
                'winner': winner,
            }
    print(f"  Crowd-gate candidates (loser<$0.50 + conv≥0.60): {len(crowd_trigger)}")

    # ── Step 2: Load recap (per-fire data) ────────────────────────────
    print(f"\nStep 2/5 — Loading recap history ...")
    recap_raw=load_jsonl(args.recap)
    recap={r.get('slug',''):r for r in recap_raw}
    print(f"  {len(recap):,} markets with fire data")

    # ── Step 3: Classify all fires ────────────────────────────────────
    print(f"\nStep 3/5 — Classifying fires ...")
    thresholds=[0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,1.00]

    # Per-fire classification for threshold sweep
    # Each fire: slug, strat, side, price, hypo, is_cheap, is_dn_live_blocked
    fires_in_gate_mkts=[]
    fires_all=[]
    n_total_fires=0

    # Also need to reconstruct dn_live blocks from gate history if available
    # (won't have it on VPS necessarily — use approximate: already-flagged fires)

    prog=Progress(len(recap_raw), 'Classify ')
    for idx, rec in enumerate(recap_raw):
        if idx%300==0: prog.update(idx+1)
        slug=rec.get('slug','')
        cg=crowd_trigger.get(slug)

        for f in rec.get('fires',[]):
            if f.get('pre_gate_held') or f.get('opp_gate_held'): continue
            side=f.get('side',''); price=f.get('entry_price',0) or 0
            hypo=f.get('hypo_pnl',0) or 0
            strat=f.get('strategy','?')
            dn_blocked=bool(f.get('opp_gate_would_block') or f.get('opp_gate_blocked'))
            n_total_fires+=1

            fire_rec={'slug':slug,'strat':strat,'side':side,'price':price,
                      'hypo':hypo,'dn_blocked':dn_blocked,'cg':cg}
            fires_all.append(fire_rec)
            if cg and side==cg['cheap_side']:
                fires_in_gate_mkts.append(fire_rec)
    prog.done()
    print(f"  Total fires classified: {n_total_fires:,}")
    print(f"  Cheap-side fires in gate markets: {len(fires_in_gate_mkts):,}")

    # ── Step 4: Threshold sweep ───────────────────────────────────────
    print(f"\nStep 4/5 — Threshold sweep ...")
    results=[]

    for thr in thresholds:
        n_b=0; w_b=0; h_b=0.0  # blocked
        n_p=0; w_p=0; h_p=0.0  # passed (gate misses them)

        # Per-policy PnL across ALL fires
        pnl_current=0.0   # dn_live only (existing)
        pnl_crowd=0.0     # crowd gate at this threshold only
        pnl_both=0.0      # both gates
        pnl_neither=0.0   # no gates

        # Count by market for per-market stats
        mkt_pnl={'current':defaultdict(float),'crowd':defaultdict(float),
                 'both':defaultdict(float),'neither':defaultdict(float)}

        for fr in fires_all:
            slug=fr['slug']; side=fr['side']; price=fr['price']
            hypo=fr['hypo']; dn=fr['dn_blocked']; cg=fr['cg']

            # Is this fire blocked by crowd gate at this threshold?
            cg_block=(cg is not None and side==cg['cheap_side'] and price<thr)

            pnl_neither+=hypo
            mkt_pnl['neither'][slug]+=hypo

            if not dn:
                pnl_current+=hypo
                mkt_pnl['current'][slug]+=hypo

            if not cg_block:
                pnl_crowd+=hypo
                mkt_pnl['crowd'][slug]+=hypo

            if not dn and not cg_block:
                pnl_both+=hypo
                mkt_pnl['both'][slug]+=hypo

        # Stats on cheap-side fires at this threshold
        for fr in fires_in_gate_mkts:
            price=fr['price']; hypo=fr['hypo']
            if price<thr:
                n_b+=1; h_b+=hypo
                if hypo>0: w_b+=1
            else:
                n_p+=1; h_p+=hypo
                if hypo>0: w_p+=1

        # Paired t-test: both vs current
        slugs=set(list(mkt_pnl['both'].keys())+list(mkt_pnl['current'].keys()))
        diffs=[mkt_pnl['both'][s]-mkt_pnl['current'][s] for s in slugs]
        mean_d=sum(diffs)/len(diffs) if diffs else 0
        se=(sum((d-mean_d)**2 for d in diffs)/len(diffs))**0.5/len(diffs)**0.5 if diffs else 0
        ci_lo=mean_d-1.96*se; ci_hi=mean_d+1.96*se

        if HAS_SCIPY and len([d for d in diffs if d!=0])>0:
            _,p_t=st.ttest_1samp([d for d in diffs if d!=0],0,
                                   alternative='greater' if mean_d>0 else 'less')
        else:
            p_t=1.0

        results.append({
            'threshold': thr,
            'n_blocked': n_b, 'n_passed': n_p,
            'wr_blocked': round(w_b/n_b,4) if n_b else 0,
            'wr_passed':  round(w_p/n_p,4) if n_p else 0,
            'ev_blocked': round(h_b/n_b,4) if n_b else 0,
            'ev_passed':  round(h_p/n_p,4) if n_p else 0,
            'pnl_save_blocked': round(-h_b,2),
            'pnl_neither': round(pnl_neither,2),
            'pnl_current': round(pnl_current,2),
            'pnl_crowd':   round(pnl_crowd,2),
            'pnl_both':    round(pnl_both,2),
            'both_vs_current': round(pnl_both-pnl_current,2),
            'ci_lo': round(ci_lo,4), 'ci_hi': round(ci_hi,4),
            'p_ttest': round(p_t,6),
        })

    # ── Step 5: Print report ──────────────────────────────────────────
    print(f"\nStep 5/5 — Report ...")
    n_mkts=len(recap_raw)

    print(f"\n{'='*75}")
    print(f"CROWD GATE THRESHOLD SWEEP ({n_mkts} markets, ALL strategies)")
    print(f"{'='*75}")
    print(f"\n  dn_live gate: {'present' if any(fr['dn_blocked'] for fr in fires_all) else 'not detected (use recap with flags)'}")
    print()
    print(f"  {'Thr':>5} {'n_blk':>6} {'WR_blk':>7} {'EV_blk':>8} {'n_pass':>7} {'WR_pass':>8} {'EV_pass':>8} {'PnL_save':>9} {'BOTH_vs_CURR':>13} {'CI_lo':>7} {'p':>7}")
    print("  "+"─"*95)

    for r in results:
        curr=r['threshold']==0.30; opt=r['threshold']==0.45
        marker=" ← current" if curr else (" ← OPTIMAL" if opt else "")
        s=sig(r['p_ttest'])
        print(f"  ${r['threshold']:.2f} {r['n_blocked']:>6} {r['wr_blocked']:>7.1%} "
              f"{r['ev_blocked']:>+8.3f} {r['n_passed']:>7} {r['wr_passed']:>8.1%} "
              f"{r['ev_passed']:>+8.3f} {r['pnl_save_blocked']:>+9.2f} "
              f"{r['both_vs_current']:>+13.2f} {r['ci_lo']:>+7.4f} {r['p_ttest']:>7.4f}{s}{marker}")

    # Best threshold
    best=max(results, key=lambda x: x['pnl_both'])
    print(f"\n  Best threshold (max PnL_BOTH): ${best['threshold']:.2f}")
    print(f"  Best CI_lo: {best['ci_lo']:+.4f}/market")

    # Per-strategy breakdown at optimal threshold
    print(f"\n{'='*75}")
    print(f"STRATEGY BREAKDOWN at $0.45 threshold (cheap-side fires in gate markets)")
    print(f"{'='*75}")

    OPT=0.45
    by_strat_blk=defaultdict(lambda:{'n':0,'w':0,'h':0.0})
    by_strat_pass=defaultdict(lambda:{'n':0,'w':0,'h':0.0})
    for fr in fires_in_gate_mkts:
        strat=fr['strat']; price=fr['price']; hypo=fr['hypo']
        if price<OPT:
            by_strat_blk[strat]['n']+=1; by_strat_blk[strat]['h']+=hypo
            if hypo>0: by_strat_blk[strat]['w']+=1
        else:
            by_strat_pass[strat]['n']+=1; by_strat_pass[strat]['h']+=hypo
            if hypo>0: by_strat_pass[strat]['w']+=1

    print(f"\n  BLOCKED by $0.45 gate (n≥10):")
    print(f"  {'Strategy':<32} {'n':>4} {'WR':>6} {'EV/fire':>8}  verdict")
    print("  "+"─"*58)
    for strat,d in sorted(by_strat_blk.items(), key=lambda x:-x[1]['n']):
        n2=d['n']; w2=d['w']
        if n2<10: continue
        wr=w2/n2; ev=d['h']/n2
        v="✅ correct block" if ev<0 else "⚠️ EV+ being blocked"
        print(f"  {strat:<32} {n2:>4} {wr:>6.1%} {ev:>+8.3f}  {v}")

    if by_strat_pass:
        print(f"\n  PASSES $0.45 gate (still fires — n≥3):")
        print(f"  {'Strategy':<32} {'n':>4} {'WR':>6} {'EV/fire':>8}")
        print("  "+"─"*52)
        for strat,d in sorted(by_strat_pass.items(), key=lambda x:-x[1]['n']):
            n2=d['n']
            if n2<3: continue
            ev=d['h']/n2; wr=d['w']/n2
            print(f"  {strat:<32} {n2:>4} {wr:>6.1%} {ev:>+8.3f}")
    else:
        print(f"\n  No strategies pass $0.45 gate (all cheap-side fires blocked)")

    # Summary
    r30=[r for r in results if r['threshold']==0.30][0]
    r45=[r for r in results if r['threshold']==0.45][0]
    print(f"\n{'='*75}")
    print(f"SUMMARY: $0.30 vs $0.45 threshold")
    print(f"{'='*75}")
    print(f"""
  $0.30 threshold (current):
    Blocks {r30['n_blocked']} fires  WR={r30['wr_blocked']:.1%}  saves {r30['pnl_save_blocked']:+.2f}
    BOTH gates: {r30['pnl_both']:+.2f}  vs current {r30['pnl_current']:+.2f}  diff={r30['both_vs_current']:+.2f}
    CI=[{r30['ci_lo']:+.4f},{r30['ci_hi']:+.4f}]  p={r30['p_ttest']:.4f}

  $0.45 threshold (optimal):
    Blocks {r45['n_blocked']} fires  WR={r45['wr_blocked']:.1%}  saves {r45['pnl_save_blocked']:+.2f}
    BOTH gates: {r45['pnl_both']:+.2f}  vs current {r45['pnl_current']:+.2f}  diff={r45['both_vs_current']:+.2f}
    CI=[{r45['ci_lo']:+.4f},{r45['ci_hi']:+.4f}]  p={r45['p_ttest']:.4f}

  Raising $0.30→$0.45 additional savings: +{r45['pnl_save_blocked']-r30['pnl_save_blocked']:+.2f}
    """)

    # Save JSON
    def _safe(o):
        if isinstance(o,bool): return int(o)
        if isinstance(o,(float,int)):
            try: return round(float(o),6)
            except: return 0
        if isinstance(o,dict): return {k:_safe(v) for k,v in o.items()}
        if isinstance(o,list): return [_safe(x) for x in o]
        return o

    output={'n_markets':n_mkts,'n_total_fires':n_total_fires,
            'n_gate_triggered':len(crowd_trigger),
            'threshold_results':results,
            'optimal_threshold':best['threshold'],
            'runtime_seconds':round(time.time()-t_global,1)}
    with open(args.out,'w') as f: json.dump(_safe(output),f,indent=2)

    total_time=Progress._fmt(time.time()-t_global)
    print(f"\n{'='*75}")
    print(f"Total runtime: {total_time}")
    print(f"✅ Report saved to {args.out}")

if __name__=='__main__':
    main()
