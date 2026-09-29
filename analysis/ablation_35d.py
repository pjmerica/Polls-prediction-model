"""Ablation at 35 days out (2026-09-29 model review): full model vs no-macro / no-fundraising /
polls-only feature sets vs the poll-average softmax baseline, expanding-window 2018-2024.

    py -X utf8 analysis/ablation_35d.py
"""
import sys, io, contextlib, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src'); sys.path.insert(0,'.')
import pandas as pd, numpy as np, xgboost as xgb
import features as F
from cycles import CYCLES, EVAL_CYCLES, natl_env
from macro_features import build_macro
from sklearn.metrics import brier_score_loss, log_loss
d=pd.read_csv('polls_long_with_results.csv',low_memory=False)
g=d[d['stage']=='general']; q=g.groupby(['race_id','question_id'])['has_result'].agg(['sum','size'])
dead=set(q[(q['sum']>0)&(q['sum']<q['size'])].index)
d=d[~(pd.Series([k in dead for k in zip(d.race_id,d.question_id)],index=d.index)&(d.stage=='general'))]
d=F.prepare_polls(d[d.has_result==1].copy()); d=d[d.year.isin(CYCLES)].copy()
d['race_id']=d.year.astype(str)+'_'+d.state+'_'+d.office+d.district.radd('-').where(d.district!='','')
meta=json.load(open('data/model_features.json')); FEAT=meta['features']; P=meta['xgb_params']; P.pop('eval_metric',None)
MAC=set(meta['macro_feats'])
with contextlib.redirect_stdout(io.StringIO()):
    macro=build_macro(); ne=natl_env(); funds=F.load_fundamentals(); fec=F.load_fec(extended=True)
    bias=F.compute_bias_priors(d); bios=F.load_candidate_bios(); pr=F.load_primary_results()
    FULL=F.build_candidate_table(d,macro,ne,funds,house_train_years=CYCLES,fec=fec,bias_priors=bias,candidate_bios=bios,primary_results=pr)
    T35=F.build_candidate_table(d[d.days_to_elec>=35],macro,ne,funds,house_train_years=CYCLES,fec=fec,bias_priors=bias,candidate_bios=bios,primary_results=pr)
sets={'full model':FEAT,'no macro (144 econ/approval)':[f for f in FEAT if f not in MAC],'no fundraising':[f for f in FEAT if not f.startswith('fund_')],'polls only':[f for f in FEAT if f not in MAC and not f.startswith('fund_') and f not in ('bio_office_level','prior_margin_cand','is_incumbent','is_inc_party_race','natl_env_cand','bias_prior_cand','is_president_party','primary_margin','opp_primary_margin','primary_margin_diff')]}
def softmax(g,t=4.0):
    v=g.poll_avg.fillna(g.poll_avg.mean()).values; e=np.exp((v-np.nanmax(v))/t); return pd.Series(e/e.sum(),index=g.index)
rows=[]
for H,TEST in [(0,FULL),(35,T35)]:
    for ty in EVAL_CYCLES:
        te=TEST[(TEST.year==ty)].copy(); tr=FULL[FULL.year<ty]; y=te.won.astype(int)
        for name,fs in sets.items():
            m=xgb.XGBClassifier(**P); m.fit(tr[fs],tr.won.astype(int))
            p=m.predict_proba(te[fs])[:,1]; te['pn']=p/pd.Series(p,index=te.index).groupby(te.race_id).transform('sum')
            pick=te.loc[te.groupby('race_id').pn.idxmax()]
            rows.append(dict(H=H,model=name,cycle=ty,brier=brier_score_loss(y,te.pn),logloss=log_loss(y,te.pn.clip(1e-6,1-1e-6)),race_acc=pick.won.mean()))
        te['ps']=te.groupby('race_id',group_keys=False).apply(softmax)
        pick=te.loc[te.groupby('race_id').ps.idxmax()]
        rows.append(dict(H=H,model='poll-average softmax baseline',cycle=ty,brier=brier_score_loss(y,te.ps),logloss=log_loss(y,te.ps.clip(1e-6,1-1e-6)),race_acc=pick.won.mean()))
r=pd.DataFrame(rows)
print(r.groupby(['H','model'])[['brier','logloss','race_acc']].mean().round(4).to_string())
print(r.pivot_table(index=['H','cycle'],columns='model',values='race_acc').round(3).to_string())
