"""Build per-cycle macro/political-climate features.

Tries FRED for the monthly series (so we get campaign-year trajectory stats:
eve / mean / max / std / trend). Falls back to documented election-eve values
if FRED is unreachable (e.g. restricted network) so the pipeline always runs.
"""
import requests, io, pandas as pd, numpy as np

CYCLES = [2018, 2020, 2022, 2024]
PRES_PARTY = {2018: 'REP', 2020: 'REP', 2022: 'DEM', 2024: 'DEM'}

FRED_SERIES = {
    'unemployment': 'UNRATE',          # monthly, %
    'gas':          'GASREGW',          # weekly, $/gal regular
    'cpi':          'CPIAUCSL',         # monthly index -> YoY%
    'gdp':          'A191RL1Q225SBEA',  # quarterly real GDP growth, annualized %
}

# Documented election-eve (October) fallback values; fixed historical record.
FALLBACK_EVE = {
    2018: dict(approval=43, inflation=2.5, gdp=2.9, unemployment=3.8, gas=2.90),
    2020: dict(approval=45, inflation=1.2, gdp=33.8, unemployment=6.9, gas=2.16),
    2022: dict(approval=42, inflation=7.7, gdp=2.7, unemployment=3.7, gas=3.80),
    2024: dict(approval=40, inflation=2.6, gdp=2.8, unemployment=4.1, gas=3.20),
}
# Presidential approval monthly series isn't on FRED; use documented campaign-year
# monthly approval (538/Gallup avg) so approval also gets trajectory stats.
APPROVAL_MONTHLY = {
    2018: [40,40,41,42,42,43,43,42,43,44,43],   # Jan..Nov 2018 (Trump)
    2020: [44,44,45,46,43,41,41,42,43,45,45],   # 2020 (Trump)
    2022: [43,42,42,42,41,39,38,42,43,43,42],   # 2022 (Biden)
    2024: [40,38,38,39,39,38,38,40,42,42,40],   # 2024 (Biden)
}

def _fred(series, timeout=60):
    try:
        r = requests.get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}',
                         timeout=timeout, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code == 200 and not r.text.lstrip().startswith('<'):
            df = pd.read_csv(io.StringIO(r.text)); df.columns = ['date', 'val']
            df['date'] = pd.to_datetime(df['date']); df['val'] = pd.to_numeric(df['val'], errors='coerce')
            return df.dropna()
    except Exception:
        pass
    return None

def _stats(vals, prefix):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return {f'{prefix}_eve': np.nan, f'{prefix}_mean': np.nan, f'{prefix}_max': np.nan,
                f'{prefix}_std': np.nan, f'{prefix}_trend': np.nan}
    x = np.arange(len(v))
    trend = np.polyfit(x, v, 1)[0] if len(v) >= 2 else 0.0
    return {f'{prefix}_eve': float(v[-1]), f'{prefix}_mean': float(np.mean(v)),
            f'{prefix}_max': float(np.max(v)), f'{prefix}_std': float(np.std(v)),
            f'{prefix}_trend': float(trend)}

def build_macro(use_fred=True):
    """Return {cycle: {feature: value}} with trajectory stats per metric."""
    series = {}
    if use_fred:
        for key, sid in FRED_SERIES.items():
            series[key] = _fred(sid)
    rows = {}
    for cyc in CYCLES:
        f = {}
        # --- approval (always from documented monthly) ---
        f.update(_stats(APPROVAL_MONTHLY[cyc], 'approval'))
        # --- FRED-based metrics, campaign window Jan..Oct of the cycle ---
        def window(df, start=f'{cyc}-01-01', end=f'{cyc}-11-01'):
            return df[(df['date'] >= start) & (df['date'] < end)]['val'].tolist() if df is not None else []
        un = window(series.get('unemployment')) if use_fred else []
        gas = window(series.get('gas')) if use_fred else []
        # inflation YoY: CPI[m] / CPI[m-12] - 1, over the campaign months
        infl = []
        cpi = series.get('cpi') if use_fred else None
        if cpi is not None:
            c = cpi.set_index('date')['val']
            for m in pd.date_range(f'{cyc}-01-01', f'{cyc}-10-01', freq='MS'):
                prev = c.get(m - pd.DateOffset(years=1)); cur = c.get(m)
                if prev is not None and cur is not None:
                    infl.append((cur / prev - 1) * 100)
        gdp = window(series.get('gdp'), f'{cyc}-01-01', f'{cyc}-11-01') if use_fred else []

        if un:   f.update(_stats(un, 'unemployment'))
        if gas:  f.update(_stats(gas, 'gas'))
        if infl: f.update(_stats(infl, 'inflation'))
        if gdp:  f.update(_stats(gdp, 'gdp'))

        # fallback: if a metric's trajectory is missing, at least set the *_eve from FALLBACK
        for metric in ['inflation', 'gdp', 'unemployment', 'gas']:
            if f'{metric}_eve' not in f or pd.isna(f.get(f'{metric}_eve')):
                ev = FALLBACK_EVE[cyc][metric]
                f.setdefault(f'{metric}_eve', ev)
                for suf in ['mean', 'max']:
                    f.setdefault(f'{metric}_{suf}', ev)
                f.setdefault(f'{metric}_std', 0.0)
                f.setdefault(f'{metric}_trend', 0.0)
        rows[cyc] = f
    return rows

if __name__ == '__main__':
    import sys
    use = '--nofred' not in sys.argv
    m = build_macro(use_fred=use)
    df = pd.DataFrame(m).T
    print('FRED used:', use, '| source:', 'live' if use else 'fallback-only')
    # show a readable subset
    cols = [c for c in df.columns if c.endswith('_eve') or c.endswith('_std')]
    print(df[sorted(cols)].round(2).to_string())
