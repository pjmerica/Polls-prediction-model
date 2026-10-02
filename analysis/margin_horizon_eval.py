"""Margin-model horizon test (2026-10-02): honest expanding-window MAE 2018-2024 with each test
cycle's polls truncated at 35 / 60 days before the election (what the live model sees), vs the
raw poll lead and a calibrated-poll linear baseline; plus signed bias by party.

    py -X utf8 analysis/margin_horizon_eval.py
"""
import sys, io, contextlib, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.linear_model import LinearRegression
import features as F
from cycles import CYCLES, EVAL_CYCLES, natl_env
from macro_features import build_macro

d = pd.read_csv('polls_long_with_results.csv', low_memory=False)
g = d[d['stage'] == 'general']; q = g.groupby(['race_id', 'question_id'])['has_result'].agg(['sum', 'size'])
dead = set(q[(q['sum'] > 0) & (q['sum'] < q['size'])].index)
d = d[~(pd.Series([k in dead for k in zip(d.race_id, d.question_id)], index=d.index) & (d.stage == 'general'))]
d = F.prepare_polls(d[d.has_result == 1].copy()); d = d[d.year.isin(CYCLES)].copy()
d['race_id'] = d.year.astype(str) + '_' + d.state + '_' + d.office + d.district.radd('-').where(d.district != '', '')
meta = json.load(open('data/margin_model_features.json')); FEAT = meta['features']
P = {k: v for k, v in meta.get('xgb_params', meta.get('params', {})).items() if k not in ('eval_metric',)}
with contextlib.redirect_stdout(io.StringIO()):
    macro = build_macro(); ne = natl_env(); funds = F.load_fundamentals(); fec = F.load_fec(extended=True)
    bias = F.compute_bias_priors(d); bios = F.load_candidate_bios(); pr = F.load_primary_results()
    def table(dd):
        c = F.build_candidate_table(dd, macro, ne, funds, house_train_years=CYCLES, fec=fec, bias_priors=bias,
                                    candidate_bios=bios, primary_results=pr)
        c['margin_actual'] = c['vote_pct'] - c['best_other_pct']
        return c[c['margin_actual'].notna()]
    FULL = table(d); T = {0: FULL, 35: table(d[d.days_to_elec >= 35]), 60: table(d[d.days_to_elec >= 60])}
rows, biasrows = [], []
for H, TEST in T.items():
    for ty in EVAL_CYCLES:
        tr = FULL[FULL.year < ty]; te = TEST[TEST.year == ty].copy()
        m = xgb.XGBRegressor(**P); m.fit(tr[FEAT], tr.margin_actual)
        te['pred'] = m.predict(te[FEAT])
        lin = LinearRegression().fit(tr[['avg_margin_over_time']].fillna(0), tr.margin_actual)
        te['lin'] = lin.predict(te[['avg_margin_over_time']].fillna(0))
        y = te.margin_actual
        rows.append(dict(H=H, cycle=ty, n=len(te), MAE_model=(te.pred - y).abs().mean(),
                         MAE_calibrated_poll=(te.lin - y).abs().mean(), MAE_raw_poll_lead=(te.poll_lead - y).abs().mean(),
                         winner_acc=((te.pred > 0) == (y > 0)).mean()))
        for p in ('DEM', 'REP'):
            t = te[te.party == p]; biasrows.append(dict(H=H, cycle=ty, party=p, mean_error=(t.pred - t.margin_actual).mean()))
r = pd.DataFrame(rows)
print(r.groupby('H')[['n', 'MAE_model', 'MAE_calibrated_poll', 'MAE_raw_poll_lead', 'winner_acc']].mean().round(3).to_string())
print(r.pivot_table(index='cycle', columns='H', values='MAE_model').round(2).to_string())
b = pd.DataFrame(biasrows)
print('\nmean signed error (pred - actual, pts; + = overstated that party):')
print(b.pivot_table(index=['H', 'party'], columns='cycle', values='mean_error').round(2).to_string())
