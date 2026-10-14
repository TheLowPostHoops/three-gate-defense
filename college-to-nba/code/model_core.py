"""Shared pieces of model.py (identical code): data, ridge fit, penalty choice, final model and project()."""
import numpy as np, pandas as pd, json, os, re
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'data'); R = os.path.join(H, '..', 'results')
COL = ['age', 'hgt', 'clsn', 'minpct', 'usg', 'ts', 'ortg', 'orb', 'drb', 'ast', 'to', 'stl', 'blk', 'tpar', 'ftr', 'p2', 'p3', 'pft']
FULL = ['lpick'] + COL
ALPHAS = [1, 3, 10, 30, 100, 300, 1000, 3000]
p = pd.read_csv(os.path.join(D, 'prospects_all.csv')); o = pd.read_csv(os.path.join(D, 'nba_outcomes.csv'))
df = p.merge(o[['year', 'pick', 'player', 'outcome', 'nba_minutes']], on=['year', 'pick', 'player'])
df = df.dropna(subset=FULL).reset_index(drop=True)

def fit(X, y, a):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; ym = y.mean()
    b = np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - ym)); return mu, sd, b, ym
def pred(m, X): mu, sd, b, ym = m; return ((X - mu) / sd) @ b + ym

def cv_alpha(tr, cols):
    """pick the ridge penalty by leaving out one draft class at a time, inside the training years only"""
    best, bs, oof = None, 9e9, None
    for a in ALPHAS:
        e = np.zeros(len(tr)); 
        for yr in tr.year.unique():
            te = (tr.year == yr).values
            m = fit(tr.loc[~te, cols].values, tr.loc[~te, 'outcome'].values, a); e[te] = pred(m, tr.loc[te, cols].values)
        s = np.abs(e - tr.outcome.values).mean()
        if s < bs: best, bs, oof = a, s, e
    return best, oof

def run(data, cols, tests=range(2016, 2023), pickcol_only=False):
    out = []
    for D_ in tests:
        tr = data[data.year <= D_ - 4]; te = data[data.year == D_]
        a, oof = cv_alpha(tr, cols); m = fit(tr[cols].values, tr.outcome.values, a)
        sdres = (oof - tr.outcome.values).std()
        t = te.copy(); t['pred'] = pred(m, te[cols].values); t['sd'] = sdres; t['alpha'] = a; out.append(t)
    return pd.concat(out)


# ---- final model on every finished class (2008-2022), then projections
tr = df; a, oof = cv_alpha(tr, FULL); m = fit(tr[FULL].values, tr.outcome.values, a); sd = (oof - tr.outcome.values).std()
mb_a, mb_oof = cv_alpha(tr, ['lpick']); mb = fit(tr[['lpick']].values, tr.outcome.values, mb_a)
coef = pd.Series(m[2], index=FULL).sort_values(); coef.round(3).to_csv(os.path.join(R, 'coefficients.csv'), header=['std_coef'])
print('final alpha', a, 'sd', round(sd, 3)); print(coef.round(2).to_string())
def project(t, label=None):
    t = t.copy(); has = t[FULL].notna().all(axis=1)
    t['proj'] = np.nan; t.loc[has, 'proj'] = pred(m, t.loc[has, FULL].values)
    t['slot_only'] = pred(mb, t[['lpick']].values)
    t['college_data'] = has; t['proj_final'] = t.proj.fillna(t.slot_only)
    t['sdv'] = np.where(has, sd, (mb_oof - tr.outcome.values).std())
    t['lo80'] = t.proj_final - 1.2816 * t.sdv; t['hi80'] = t.proj_final + 1.2816 * t.sdv
    return t
