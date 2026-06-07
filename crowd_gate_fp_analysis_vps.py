#!/usr/bin/env python3
"""
crowd_gate_fp_analysis_vps.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Finds all FALSE POSITIVE markets (crowd gate triggers but cheap side WINS)
2. Tests every real-time signal to separate TP from FP
3. Finds optimal compound condition to capture FP profit AND TP savings

Usage:
    python3 crowd_gate_fp_analysis_vps.py \
        --history /home/polybot/polymarket-bot/market_history.jsonl \
        --recap   /home/polybot/polymarket-bot/market_recap_history.jsonl \
        --out     crowd_gate_fp_report.json
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
        bar='█'*int(self.BAR_W*frac)+'░'*(self.BAR_W-int(self.BAR_W*frac))
        sys.stdout.write(f"\r  {self.label}  [{bar}] {frac*100:5.1f}%  "
            f"{done:,}/{self.total:,}  {self._fmt(e)}  ETA {eta}  "
            f"({done/max(e,0.001):,.0f}/s)   ")
        sys.stdout.flush()
    def done(self):
        self._done=True; e=self._fmt(time.time()-self.t0)
        sys.stdout.write(f"\r  {self.label}  [{'█'*self.BAR_W}] 100.0%  "
            f"{self.total:,}/{self.total:,}  {e}  ✅\n")
        sys.stdout.flush()
    @staticmethod
    def _fmt(s):
        s=int(s); h,r=divmod(s,3600); m,sec=divmod(r,60)
        return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m:02d}m{sec:02d}s"

def wilson(w, n, z=1.96):
    if n==0: return 0.0,0.0
    p=w/n
    lo=max(0,(p+z**2/(2*n)-z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/(1+z**2/n))
    hi=min(1,(p+z**2/(2*n)+z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/(1+z**2/n))
    return lo,hi

def binom_p(w,n,p0,alt='two-sided'):
    if not HAS_SCIPY or n==0: return 1.0
    return st.binomtest(w,n,p0,alternative=alt).pvalue

def sig(p):
    return '***' if p<0.001 else('**' if p<0.01 else('*' if p<0.05 else('~' if p<0.1 else ' ')))

def load_jsonl(path, label=''):
    sys.stdout.write(f"  Counting {label or path.split('/')[-1]} ..."); sys.stdout.flush()
    t0=time.time()
    with open(path) as f: total=sum(1 for l in f if l.strip())
    print(f"\r  Counting {label or path.split('/')[-1]} ... {total:,} lines ({time.time()-t0:.1f}s)")
    rows=[]; prog=Progress(total, '  Loading ')
    with open(path) as f:
        for i,line in enumerate(f):
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except: continue
            if i%500==0: prog.update(i+1)
    prog.done()
    return rows

def main():
    t0=time.time()
    p=argparse.ArgumentParser()
    p.add_argument('--history', required=True)
    p.add_argument('--recap',   required=True)
    p.add_argument('--out',     default='crowd_gate_fp_report.json')
    args=p.parse_args()

    # ── Step 1: Load ─────────────────────────────────────────────────
    print(f"\nStep 1/5 — Loading files ...")
    hist=load_jsonl(args.history,'market_history')
    recap_raw=load_jsonl(args.recap,'market_recap')
    recap={r.get('slug',''):r for r in recap_raw}
    print(f"  {len(hist):,} history  |  {len(recap):,} recap markets")

    # ── Step 2: Classify gate markets as TP/FP ───────────────────────
    print(f"\nStep 2/5 — Classifying gate-triggered markets ...")
    prog=Progress(len(hist),'Classify ')
    tp_rows=[]; fp_rows=[]
    all_gate=[]

    for idx,m in enumerate(hist):
        if idx%300==0: prog.update(idx+1)
        winner=m.get('winner')
        if winner not in ('UP','DN'): continue
        sc=m.get('skip_counterfactuals') or {}
        mw=m.get('market_wide') or {}
        v200=sc.get('T-200') or {}

        cs=v200.get('crowd_side'); cc=v200.get('crowd_conv',0) or 0
        lb=v200.get('loser_bid',1.0) or 1.0
        bn=v200.get('bn_delta',0) or 0
        wa=v200.get('winner_ask',0) or 0
        lc=mw.get('lead_changes',0) or 0
        t5=mw.get('time_near_5050',0) or 0
        slug=m.get('slug','')

        # CORRECT cheap_bid: bid of the side crowd does NOT back
        # TP: crowd=winner=expensive → cheap_side=loser → cheap_bid=loser_bid
        # FP: crowd=loser=expensive  → cheap_side=winner → cheap_bid=winner_ask
        crowd_right = (cs == winner)
        cheap_bid = lb if crowd_right else wa

        # Gate trigger: cheap side below threshold
        if not (cs in ('UP','DN') and cc>=0.60 and cheap_bid<0.50): continue

        cheap='DN' if cs=='UP' else 'UP'

        # Fire data
        rec=recap.get(slug,{})
        fires_cheap=[]   # cheap-side fires (would be blocked)
        fires_correct=[] # expensive-side fires (should fire)
        for f in rec.get('fires',[]):
            if f.get('pre_gate_held') or f.get('opp_gate_held'): continue
            side=f.get('side',''); price=f.get('entry_price',0) or 0
            hypo=f.get('hypo_pnl',0) or 0
            if side==cheap and price<0.45: fires_cheap.append(hypo)
            elif side!=cheap: fires_correct.append(hypo)

        row={
            'slug':slug,'winner':winner,'crowd_side':cs,'cheap_side':cheap,
            'crowd_right':crowd_right,
            # Real-time signals (available at gate trigger ~cd=230-250)
            'conv':round(cc,3),
            'cheap_bid':round(cheap_bid,3),   # CORRECTED: bid of side crowd does NOT back
            'loser_bid':round(lb,3),
            'winner_ask':round(wa,3),
            'bn':round(bn,5),
            'abs_bn':round(abs(bn),5),
            'bn_aligned':int((bn<0 and cs=='DN') or (bn>0 and cs=='UP')),
            # Future signals (for analysis only)
            'lc':lc,'t5050':t5,
            # PnL impact
            'n_cheap':len(fires_cheap),
            'hypo_cheap':round(sum(fires_cheap),3),
            'hypo_correct':round(sum(fires_correct),3),
            # Gate condition flag
            'in_gate':int(lb<0.45),
        }
        all_gate.append(row)
        if crowd_right: tp_rows.append(row)
        else:           fp_rows.append(row)

    prog.done()
    n_tp=len(tp_rows); n_fp=len(fp_rows); n_g=n_tp+n_fp
    print(f"\n  Gate candidates (lb<0.50, conv≥0.60): {n_g}")
    print(f"  TRUE POSITIVE  (crowd right → cheap side lost): {n_tp} ({n_tp/n_g:.1%})")
    print(f"  FALSE POSITIVE (crowd wrong → cheap side won):  {n_fp} ({n_fp/n_g:.1%})")

    # Subset: strict gate (lb<0.45)
    tp_strict=[r for r in tp_rows if r['in_gate']]
    fp_strict=[r for r in fp_rows if r['in_gate']]
    n_s=len(tp_strict)+len(fp_strict)
    print(f"\n  STRICT gate (lb<0.45): {n_s} total")
    print(f"    TP={len(tp_strict)} ({len(tp_strict)/n_s:.1%})  FP={len(fp_strict)} ({len(fp_strict)/n_s:.1%})")

    if n_fp==0:
        print(f"\n  ⚠️  No false positives found — gate precision = 100% on this dataset")
        print(f"  Run more markets to accumulate FP data before this analysis is useful.")
        # Save minimal output
        out={'n_markets_total':len(hist),'n_gate_triggered':n_g,
             'n_tp':n_tp,'n_fp':n_fp,'precision':round(n_tp/n_g,4) if n_g else 0}
        with open(args.out,'w') as f: json.dump(out,f,indent=2)
        return

    # ── Step 3: Signal power analysis ────────────────────────────────
    print(f"\nStep 3/5 — Signal power: TP vs FP ...")

    signals=[
        ('conv',       'Crowd conviction (real-time ✅)'),
        ('loser_bid',  'Cheap side bid (real-time ✅)'),
        ('winner_ask', 'Expensive side ask (real-time ✅)'),
        ('abs_bn',     '|BN momentum| (real-time ✅)'),
        ('bn_aligned', 'BN agrees with crowd (real-time ✅)'),
        ('lc',         'Lead changes [FUTURE ❌ analysis only]'),
        ('t5050',      'Time near 50/50 [FUTURE ❌ analysis only]'),
    ]

    N_SIG=len(signals); bonf=0.05/N_SIG
    print(f"\n  {N_SIG} signals  |  Bonferroni p<{bonf:.4f}")
    print(f"\n  {'Signal':<14} {'TP_mean':>9} {'FP_mean':>9} {'diff':>9} {'p-val':>9}  verdict")
    print("  "+"─"*65)

    sig_results=[]
    for name,desc in signals:
        tp_v=[r[name] for r in tp_rows]; fp_v=[r[name] for r in fp_rows]
        m_tp=sum(tp_v)/len(tp_v); m_fp=sum(fp_v)/len(fp_v)
        if HAS_SCIPY:
            _,pv=st.mannwhitneyu(tp_v,fp_v,alternative='two-sided')
        else: pv=1.0
        diff=m_tp-m_fp
        verdict="✅ KEEP" if pv<bonf else("🔬" if pv<0.05 else "❌")
        future=" [FUTURE]" if "FUTURE" in desc else ""
        print(f"  {name:<14} {m_tp:>9.3f} {m_fp:>9.3f} {diff:>+9.3f} {pv:>9.5f}  {verdict}{future}")
        sig_results.append({'name':name,'p':pv,'tp_mean':m_tp,'fp_mean':m_fp,'diff':diff})

    # ── Step 4: Compound threshold sweep (real-time signals only) ────
    print(f"\nStep 4/5 — Compound threshold sweep ...")

    # Test conditions that could BOTH improve precision AND identify FP markets
    # FP markets = cheap side will WIN = should FIRE cheap side not block it
    # Goal: find condition where FP fires are EV-positive

    conv_thrs  = [0.60,0.65,0.70,0.75,0.80]
    bn_thrs    = [0.0,0.02,0.03,0.05,0.08]
    lb_thrs    = [0.30,0.35,0.40,0.45]

    N_TESTS=(len(conv_thrs)*len(bn_thrs)*len(lb_thrs))+len(conv_thrs)+len(bn_thrs)+len(lb_thrs)
    bonf_tests=0.05/N_TESTS
    print(f"  {N_TESTS} tests  |  Bonferroni p<{bonf_tests:.5f}")

    test_results=[]

    def run_test(label, cond):
        tp_sub=[r for r in tp_rows if cond(r)]
        fp_sub=[r for r in fp_rows if cond(r)]
        n=len(tp_sub)+len(fp_sub)
        if n<5: return None
        prec=len(tp_sub)/n
        # EV: for TP (gate correct), blocking cheap fires saves money
        # For FP (gate wrong), blocking cheap fires costs money
        tp_hypo=sum(r['hypo_cheap'] for r in tp_sub)  # negative = saves if blocked
        fp_hypo=sum(r['hypo_cheap'] for r in fp_sub)  # positive = costs if blocked
        pv=binom_p(len(tp_sub),n,0.9,'greater') if HAS_SCIPY else 1.0
        return {'label':label,'n':n,'n_tp':len(tp_sub),'n_fp':len(fp_sub),
                'prec':round(prec,4),
                'tp_hypo':round(tp_hypo,2),'fp_hypo':round(fp_hypo,2),
                'net_hypo':round(tp_hypo+fp_hypo,2),
                'p':round(pv,6),'bonf_ok':int(pv<bonf_tests)}

    # Single condition tests
    for cv in conv_thrs:
        r=run_test(f"conv≥{cv:.2f}", lambda x,c=cv: x['conv']>=c)
        if r: test_results.append(r)
    for bn in bn_thrs:
        r=run_test(f"abs_bn≥{bn:.3f}", lambda x,b=bn: x['abs_bn']>=b)
        if r: test_results.append(r)
    for lb in lb_thrs:
        r=run_test(f"lb<{lb:.2f}", lambda x,l=lb: x['loser_bid']<l)
        if r: test_results.append(r)

    # Compound tests
    for cv in conv_thrs:
        for bn in bn_thrs:
            for lb in lb_thrs:
                lbl=f"lb<{lb:.2f}+conv≥{cv:.2f}+bn≥{bn:.3f}"
                r=run_test(lbl, lambda x,c=cv,b=bn,l=lb:
                           x['loser_bid']<l and x['conv']>=c and x['abs_bn']>=b)
                if r: test_results.append(r)

    test_results.sort(key=lambda x: -x['prec'])
    print(f"\n  Top conditions by precision (TP/(TP+FP)):")
    print(f"  {'Condition':<40} {'n':>4} {'prec':>7} {'n_fp':>5} {'net_hypo':>10}  BF")
    print("  "+"─"*72)
    for r in test_results[:20]:
        bf="✅" if r['bonf_ok'] else "  "
        print(f"  {r['label']:<40} {r['n']:>4} {r['prec']:>7.1%} {r['n_fp']:>5} "
              f"{r['net_hypo']:>+10.2f}  {bf}")

    # ── Step 5: FP profile + fire analysis ───────────────────────────
    print(f"\nStep 5/5 — False positive deep-dive ...")
    print(f"\n  FALSE POSITIVE markets (cheap side won — gate would be WRONG):")
    print(f"  {'Slug':<15} {'winner':>7} {'crowd':>7} {'conv':>6} {'lb':>6} {'abs_bn':>7} {'lc':>4} {'t5050':>6} {'hypo_cheap':>11}")
    print("  "+"─"*78)
    for r in sorted(fp_rows, key=lambda x: x['conv'], reverse=True):
        print(f"  {r['slug'][-13:]:<15} {r['winner']:>7} {r['crowd_side']:>7} "
              f"{r['conv']:>6.2f} {r['loser_bid']:>6.3f} {r['abs_bn']:>7.4f} "
              f"{r['lc']:>4} {r['t5050']:>6.0f} {r['hypo_cheap']:>+11.2f}")

    # PnL simulation: what if we exempt FP markets from blocking?
    print(f"\n  PnL simulation: what if we could perfectly identify FP markets?")
    tp_cheap_hypo=sum(r['hypo_cheap'] for r in tp_rows)  # negative = saved
    fp_cheap_hypo=sum(r['hypo_cheap'] for r in fp_rows)  # positive = missed
    total_cheap_hypo=tp_cheap_hypo+fp_cheap_hypo

    print(f"    Block ALL cheap fires (current): saves {-tp_cheap_hypo:+.2f} from TP, "
          f"costs {-fp_cheap_hypo:+.2f} from FP, net={-total_cheap_hypo:+.2f}")
    print(f"    Block ONLY TP, exempt FP:        saves {-tp_cheap_hypo:+.2f} from TP, "
          f"FP fires WIN {fp_cheap_hypo:+.2f}, net={-tp_cheap_hypo+fp_cheap_hypo:+.2f}")
    print(f"    Max possible gain from perfect FP identification: {fp_cheap_hypo*2:+.2f}")

    # Save JSON
    def _safe(o):
        if isinstance(o,bool): return int(o)
        if isinstance(o,(float,int)):
            try: return round(float(o),6)
            except: return 0
        if isinstance(o,dict): return {k:_safe(v) for k,v in o.items()}
        if isinstance(o,list): return [_safe(x) for x in o]
        return o

    output={
        'n_markets':len(hist),'n_gate_triggered':n_g,
        'n_tp':n_tp,'n_fp':n_fp,
        'precision':round(n_tp/n_g,4) if n_g else 0,
        'n_tp_strict':len(tp_strict),'n_fp_strict':len(fp_strict),
        'signal_analysis':sig_results,
        'top_conditions':test_results[:10],
        'fp_markets':[{k:v for k,v in r.items() if k!='slug'} for r in fp_rows],
        'runtime_seconds':round(time.time()-t0,1),
    }
    with open(args.out,'w') as f: json.dump(_safe(output),f,indent=2)
    print(f"\n{'='*65}")
    print(f"Total runtime: {Progress._fmt(time.time()-t0)}")
    print(f"✅ Saved to {args.out}")
    print(f"   TP={n_tp}  FP={n_fp}  precision={n_tp/n_g:.1%}" if n_g else "")

if __name__=='__main__':
    main()
