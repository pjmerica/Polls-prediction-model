"""Horizon test (2026-09-29 model review): honest expanding-window eval of the WIN model with each
test cycle's polls truncated at 35 / 60 days before election - what the live model sees ~5 weeks
out - plus a calibration table. Also compares training on truncated history.

    py -X utf8 analysis/horizon_eval.py <out_dir>
"""
import sys, io, contextlib, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'src'); sys.path.insert(0,'.')
import pandas as pd, numpy as np, xgboost as xgb
import features as F
from cycles import CYCLES, EVAL_CYCLES, natl_env
from macro_features import build_macro
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
S=sys.argv[1]
d=pd.read_csv('polls_long_with_results.csv',low_memory=False)
g=d[d['stage']=='general']; q=g.groupby(['race_id','question_id'])['has_result'].agg(['sum','size'])
dead=set(q[(q['sum']>0)&(q['sum']<q['size'])].index)
d=d[~(pd.Series([k in dead for k in zip(d.race_id,d.question_id)],index=d.index)&(d.stage=='general'))]
d=F.prepare_polls(d[d.has_result==1].copy()); d=d[d.year.isin(CYCLES)].copy()
d['race_id']=d.year.astype(str)+'_'+d.state+'_'+d.office+d.district.radd('-').where(d.district!='','')
meta=json.load(open('data/model_features.json')); FEAT=meta['features']; P=meta['xgb_params']; P.pop('eval_metric',None)
q=io.StringIO()
with contextlib.redirect_stdout(q):
    macro=build_macro(); ne=natl_env(); funds=F.load_fundamentals(); fec=F.load_fec(extended=True)
    bias=F.compute_bias_priors(d); bios=F.load_candidate_bios(); pr=F.load_primary_results()
def table(dd, house_years):
    with contextlib.redirect_stdout(io.StringIO()):
        return F.build_candidate_table(dd, macro, ne, funds, house_train_years=house_years, fec=fec, bias_priors=bias, candidate_bios=bios, primary_results=pr)
FULL=table(d, CYCLES)
def norm(te,col):
    return te[col]/te.groupby('race_id')[col].transform('sum')
rows=[]; calib=[]
for H in [0,35,60]:
    dt = d if H==0 else d[d.days_to_elec>=H]
    TR_trunc = FULL if H==0 else table(dt, CYCLES)
    for ty in EVAL_CYCLES:
        te=TR_trunc[TR_trunc.year==ty].copy()
        full_te=FULL[FULL.year==ty]
        te=te[te.race_id.isin(full_te.race_id)]
        for train_mode in ['full-history','same-horizon']:
            trsrc = FULL if train_mode=='full-history' else TR_trunc
            tr=trsrc[trsrc.year<ty]
            m=xgb.XGBClassifier(**P); m.fit(tr[FEAT],tr.won.astype(int))
            te['p']=m.predict_proba(te[FEAT])[:,1]; te['pn']=norm(te,'p')
            y=te.won.astype(int)
            pick=te.loc[te.groupby('race_id').pn.idxmax()]
            rows.append(dict(H=H,train=train_mode,cycle=ty,n_races=te.race_id.nunique(),
                brier=brier_score_loss(y,te.pn.clip(0,1)),logloss=log_loss(y,te.pn.clip(1e-6,1-1e-6)),
                race_acc=pick.won.mean(),
                fav_prob=pick.pn.mean(), fav_won=pick.won.mean()))
            if train_mode=='full-history':
                t=te[['pn','won']].assign(H=H,cycle=ty); calib.append(t)
r=pd.DataFrame(rows)
print(r.groupby(['H','train'])[['n_races','brier','logloss','race_acc','fav_prob','fav_won']].mean().round(3).to_string())
c=pd.concat(calib); c['bin']=pd.cut(c.pn,[0,.1,.3,.5,.7,.9,.97,1.0])
print(c.groupby(['H','bin'],observed=True).agg(n=('won','size'),pred=('pn','mean'),actual=('won','mean')).round(3).to_string())
r.to_csv(S+'/horizon_rows.csv',index=False); c.to_csv(S+'/horizon_calib.csv',index=False)
