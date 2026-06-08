#!/usr/bin/env python3
"""
crowd_gate_policy_vps.py
━━━━━━━━━━━━━━━━━━━━━━━━
Tests all gate policies using ZERO FUTURE BIAS signals only:
  cheap_bid = real-time price of side crowd does NOT back
  conv      = crowd conviction (real-time)
  abs_bn    = |BN delta| (real-time)

No winner/loser labels used in any gate condition.

Usage:
    python3 crowd_gate_policy_vps.py \
        --history ./market_history.jsonl \
        --recap   ./market_recap_history.jsonl \
        --out     crowd_gate_policy_report.json
"""
import json, math, argparse, sys, time
from collections import defaultdict

try:
    from scipy import stats as st
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class Progress:
    BAR_W=35
    def __init__(self,total,label=''):
        self.total=total;self.label=label;self.t0=time.time();self._done=False
    def update(self,done):
        if self._done: return
        e=time.time()-self.t0;frac=done/self.total if self.total else 1
        eta=self._fmt(e/frac*(1-frac)) if frac>0 else '--:--'
        bar='█'*int(self.BAR_W*frac)+'░'*(self.BAR_W-int(self.BAR_W*frac))
        sys.stdout.write(f"\r  {self.label}  [{bar}] {frac*100:5.1f}%  "
            f"{done:,}/{self.total:,}  {self._fmt(e)}  ETA {eta}  "
            f"({done/max(e,0.001):,.0f}/s)   ")
        sys.stdout.flush()
    def done(self):
        self._done=True;e=self._fmt(time.time()-self.t0)
        sys.stdout.write(f"\r  {self.label}  [{'█'*self.BAR_W}] 100.0%  "
            f"{self.total:,}/{self.total:,}  {e}  ✅\n")
        sys.stdout.flush()
    @staticmethod
    def _fmt(s):
        s=int(s);h,r=divmod(s,3600);m,sec=divmod(r,60)
        return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m:02d}m{sec:02d}s"

def wilson(w,n,z=1.96):
    if n==0: return 0.0,0.0
    p=w/n
    lo=max(0,(p+z**2/(2*n)-z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/(1+z**2/n))
    hi=min(1,(p+z**2/(2*n)+z*math.sqrt(p*(1-p)/n+z**2/(4*n**2)))/(1+z**2/n))
    return lo,hi

def load_jsonl(path,label=''):
    lbl=label or path.split('/')[-1]
    sys.stdout.write(f"  Counting {lbl} ..."); sys.stdout.flush()
    t0=time.time()
    with open(path) as f: total=sum(1 for l in f if l.strip())
    print(f"\r  Counting {lbl} ... {total:,} lines ({time.time()-t0:.1f}s)")
    rows=[]; prog=Progress(total,'  Loading ')
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
    p.add_argument('--history',required=True)
    p.add_argument('--recap',required=True)
    p.add_argument('--out',default='crowd_gate_policy_report.json')
    args=p.parse_args()

    print(f"\nStep 1/5 — Loading ...")
    hist=load_jsonl(args.history,'market_history')
    recap_raw=load_jsonl(args.recap,'market_recap')
    recap={r.get('slug',''):r for r in recap_raw}
    print(f"  {len(hist):,} history  |  {len(recap):,} recap")

    print(f"\nStep 2/5 — Building feature rows ...")
    prog=Progress(len(hist),'Features ')
    rows=[]
    for idx,m in enumerate(hist):
        if idx%300==0: prog.update(idx+1)
        winner=m.get('winner')
        if winner not in ('UP','DN'): continue
        sc=m.get('skip_counterfactuals') or {}
        v200=sc.get('T-200') or {}
        cs=v200.get('crowd_side'); cc=v200.get('crowd_conv',0) or 0
        lb=v200.get('loser_bid',1.0) or 1.0
        wa=v200.get('winner_ask',1.0) or 1.0
        bn=v200.get('bn_delta',0) or 0
        if cs not in ('UP','DN') or cc<0.50: continue

        # ── REAL-TIME cheap_bid: price of side crowd does NOT back ──
        # TP: crowd=winner → cheap_side=loser  → cheap_bid=loser_bid
        # FP: crowd=loser  → cheap_side=winner → cheap_bid=winner_ask
        # At runtime the bot computes this from live up_ask/dn_ask directly
        cheap_bid = lb if cs==winner else wa

        cheap_side='DN' if cs=='UP' else 'UP'
        rec=recap.get(m.get('slug',''),{})
        fires=[]
        for f in rec.get('fires',[]):
            if f.get('pre_gate_held') or f.get('opp_gate_held'): continue
            if f.get('side','')!=cheap_side: continue
            fires.append((f.get('entry_price',0) or 0, f.get('hypo_pnl',0) or 0))

        rows.append({
            'slug': m.get('slug',''),
            'conv': cc,
            'cheap_bid': cheap_bid,
            'abs_bn': abs(bn),
            'fires': fires,
            # label for analysis only (NOT used in any gate condition)
            'crowd_right': (cs==winner),
        })
    prog.done()
    n=len(rows)
    print(f"  {n:,} markets with crowd signal")
    print(f"  Total cheap-side fires: {sum(len(r['fires']) for r in rows):,}")

    # ── Step 3: Policy definitions (ZERO future bias) ────────────────
    # Each policy: (label, lambda row,cb -> bool: fire is ALLOWED)
    policies = [
        ('A_NO_GATE',      'No gate (baseline)',
         lambda r,cb: True),
        ('B_CB45',         'Block cb<$0.45 (current)',
         lambda r,cb: cb>=0.45),
        ('C_CB30',         'Block cb<$0.30',
         lambda r,cb: cb>=0.30),
        ('D_CB20',         'Block cb<$0.20',
         lambda r,cb: cb>=0.20),
        ('E_CB45_CV70',    'Block cb<$0.45 + conv≥0.70',
         lambda r,cb: not(cb<0.45 and r['conv']>=0.70)),
        ('F_CB30_CV65',    'Block cb<$0.30 + conv≥0.65',
         lambda r,cb: not(cb<0.30 and r['conv']>=0.65)),
        ('G_CB20_CV65',    'Block cb<$0.20 + conv≥0.65',
         lambda r,cb: not(cb<0.20 and r['conv']>=0.65)),
        ('H_CB45_BN003',   'Block cb<$0.45 + abs_bn≥0.03',
         lambda r,cb: not(cb<0.45 and r['abs_bn']>=0.03)),
        ('I_CB45_CV70_BN', 'Block cb<$0.45 + cv≥0.70 + bn≥0.03',
         lambda r,cb: not(cb<0.45 and r['conv']>=0.70 and r['abs_bn']>=0.03)),
    ]
    N_POL=len(policies); BONF=0.05/N_POL

    print(f"\nStep 3/5 — Policy PnL simulation ({N_POL} policies, Bonferroni p<{BONF:.4f}) ...")
    mkt_pnls={name:[] for name,_,_ in policies}
    for r in rows:
        for name,_,allow in policies:
            pnl=sum(h for cb,h in r['fires'] if allow(r,cb))
            mkt_pnls[name].append(pnl)

    totals={name:sum(v) for name,v in mkt_pnls.items()}
    BASE='B_CB45'

    print(f"\nStep 4/5 — Statistical tests vs {BASE} ...")
    results=[]
    base_v=mkt_pnls[BASE]
    for name,label,_ in policies:
        t=totals[name]; m_val=t/n
        diffs=[mkt_pnls[name][i]-base_v[i] for i in range(n)]
        mean_d=sum(diffs)/n
        se=(sum((d-mean_d)**2 for d in diffs)/n)**0.5/n**0.5
        ci_lo=mean_d-1.96*se; ci_hi=mean_d+1.96*se
        nd=[d for d in diffs if abs(d)>0.001]
        if nd and mean_d!=0 and HAS_SCIPY:
            _,pv=st.ttest_1samp(nd,0,alternative='greater' if mean_d>0 else 'less')
        else: pv=1.0
        results.append({'name':name,'label':label,'total':t,'per_mkt':m_val,
                        'vs_base':mean_d,'ci_lo':ci_lo,'ci_hi':ci_hi,'p':pv,
                        'bonf_ok':int(pv<BONF)})

    print(f"\nStep 5/5 — Precision analysis + report ...")

    # Print full report
    print(f"\n{'='*75}")
    print(f"POLICY PnL — {n} markets — ZERO FUTURE BIAS")
    print(f"Gate conditions: cheap_bid + conv + abs_bn only (all real-time)")
    print(f"{'='*75}")
    print(f"\n  {'Policy':<32} {'Total':>10} {'Per_mkt':>9} {'vs_B':>9} {'CI_lo':>8} {'p':>8}  BF")
    print("  "+"─"*80)
    best_t=max(r['total'] for r in results)
    for r in results:
        bf="✅" if r['bonf_ok'] else("🔬" if r['p']<0.05 else "  ")
        cur=" ←" if r['name']==BASE else ""
        best=" BEST✅" if r['total']==best_t and r['name']!=BASE else ""
        print(f"  {r['label']:<32} {r['total']:>+10.2f} {r['per_mkt']:>+9.4f} "
              f"{r['vs_base']:>+9.4f} {r['ci_lo']:>+8.4f} {r['p']:>8.4f}  {bf}{cur}{best}")
    print(f"\n  Bonferroni threshold: p<{BONF:.4f}")

    # Precision analysis
    print(f"\n{'='*75}")
    print(f"PRECISION (WR = cheap-side fires that LOST = gate was correct)")
    print(f"No future bias: hypo<0 means the fire lost — no winner label needed")
    print(f"{'='*75}")
    print(f"\n  {'Policy':<32} {'n_blk':>8} {'WR_lose':>9} {'CI_lo':>8} {'EV/f':>8}  p")
    print("  "+"─"*72)
    prec_results=[]
    for name,label,allow in policies:
        if name=='A_NO_GATE': continue
        blk=[(cb,h) for r in rows for cb,h in r['fires'] if not allow(r,cb)]
        nb=len(blk)
        if nb==0: continue
        nl=sum(1 for _,h in blk if h<0); wr=nl/nb; ev=sum(h for _,h in blk)/nb
        cl,ch=wilson(nl,nb)
        pv=st.binomtest(nl,nb,0.5,alternative='greater').pvalue if HAS_SCIPY else 1.0
        bf="✅" if pv<BONF else "  "
        print(f"  {label:<32} {nb:>8} {wr:>9.1%} {cl:>8.1%} {ev:>+8.3f}  {pv:.6f} {bf}")
        prec_results.append({'name':name,'n_blocked':nb,'wr':wr,'ci_lo':cl,'ev':ev,'p':pv})

    # Sample size analysis for best policy
    print(f"\n{'='*75}")
    print(f"SAMPLE SIZE — how many more markets to confirm best policy?")
    print(f"{'='*75}")
    best_pol=max(results,key=lambda x:x['total'])
    if best_pol['name']!=BASE:
        diffs=[mkt_pnls[best_pol['name']][i]-base_v[i] for i in range(n)]
        mean_d=best_pol['vs_base']
        std_d=(sum((d-mean_d)**2 for d in diffs)/n)**0.5
        if mean_d>0 and std_d>0:
            n_ci=math.ceil((1.96*std_d/mean_d)**2)+1
            # Bonferroni z-score
            from scipy.stats import norm
            z_bonf=norm.ppf(1-BONF)
            n_bonf=math.ceil((z_bonf*std_d/mean_d)**2)+1
        else:
            n_ci=n_bonf=9999
        print(f"""
  Best policy: {best_pol['label']}
  Current n={n}  improvement={mean_d:+.4f}/mkt  std={std_d:.4f}
  
  Need n≥{n_ci} for CI_lo > 0      ({max(0,n_ci-n):+} markets = {max(0,n_ci-n)/288:.1f} days)
  Need n≥{n_bonf} for Bonferroni p<{BONF:.4f} ({max(0,n_bonf-n):+} markets = {max(0,n_bonf-n)/288:.1f} days)
  
  {'✅ ALREADY CONFIRMED' if best_pol['bonf_ok'] else '⚠️  NOT YET CONFIRMED — collect more data'}
        """)

    # Save
    def _s(o):
        if isinstance(o,bool): return int(o)
        if isinstance(o,(float,int)):
            try: return round(float(o),6)
            except: return 0
        if isinstance(o,dict): return {k:_s(v) for k,v in o.items()}
        if isinstance(o,list): return [_s(x) for x in o]
        return o
    output={'n_markets':n,'n_policies':N_POL,'bonf_threshold':BONF,
            'policy_results':results,'precision_results':prec_results,
            'runtime_seconds':round(time.time()-t0,1)}
    with open(args.out,'w') as f: json.dump(_s(output),f,indent=2)
    print(f"\n{'='*75}")
    print(f"Total runtime: {Progress._fmt(time.time()-t0)}")
    print(f"✅ Report saved to {args.out}")

if __name__=='__main__': main()
