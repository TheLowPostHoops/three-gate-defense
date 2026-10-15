"""Aging curve model. Rules are in ../PREREG.txt (written before the first run)."""
import numpy as np, pandas as pd, os, json
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'data'); R = os.path.join(H, '..', 'results')
ALPHAS = [1, 3, 10, 30, 100, 300, 1000, 3000]
L = pd.read_csv(os.path.join(D, 'lebron_master.csv'))
L = L.rename(columns={'O-LEBRON': 'olb', 'D-LEBRON': 'dlb', 'NBA ID': 'id', 'LEBRON': 'lb', 'Minutes': 'mins', 'Age': 'age'})
L['grp'] = L.Pos.map(lambda p: 'G' if p in ('PG', 'SG') else ('F' if p in ('SF', 'PF') else 'C'))
nx = L[['id', 'year', 'lb', 'mins']].rename(columns={'lb': 'y1', 'mins': 'mins1'}); nx['year'] -= 1
pv = L[['id', 'year', 'lb']].rename(columns={'lb': 'prev'}); pv['year'] += 1
A = L.merge(nx, on=['id', 'year']).merge(pv, on=['id', 'year'], how='left')
A = A[(A.mins >= 500) & (A.mins1 >= 300)].copy(); A['prev'] = A.prev.fillna(A.lb); A = A.rename(columns={'year': 't'})
A['lmin'] = np.log(A.mins); A['G'] = (A.grp == 'G') * 1.0; A['Fw'] = (A.grp == 'F') * 1.0
def add_age(d):
    d['a1'] = d.age - 28; d['a2'] = d.a1 ** 2; d['a3'] = d.a1 ** 3
    for g in ('G', 'Fw'):
        d[g + '_a1'] = d[g] * d.a1; d[g + '_a2'] = d[g] * d.a2
    return d
A = add_age(A)
P = A[A.age >= 26].reset_index(drop=True)
RAT = ['lb', 'olb', 'dlb', 'prev', 'lmin']; AGE = ['a1', 'a2', 'a3']; POS = ['G', 'Fw', 'G_a1', 'G_a2', 'Fw_a1', 'Fw_a2']
FULL = RAT + AGE + POS; AGEONLY = RAT + AGE; BASE = ['lb']; BLOCK = AGE + POS
def fit(X, y, a):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; ym = y.mean()
    return mu, sd, np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - ym)), ym
def pr(m, X): return ((X - m[0]) / m[1]) @ m[2] + m[3]
def cv_alpha(tr, cols):
    best, bs, oof = None, 9e9, None; ys = tr.y1.values; X = tr[cols].values; ts = tr.t.values
    for a in ALPHAS:
        e = np.zeros(len(tr))
        for t in np.unique(ts):
            te = ts == t; e[te] = pr(fit(X[~te], ys[~te], a), X[te])
        s = np.abs(e - ys).mean()
        if s < bs: best, bs, oof = a, s, e
    return best, oof
def run(data, tests=range(2016, 2026), shuffle=None):
    out = []
    for t in tests:
        tr = data[data.t + 1 <= t]; te = data[data.t == t].copy()
        a, oof = cv_alpha(tr, FULL); m = fit(tr[FULL].values, tr.y1.values, a)
        ab, _ = cv_alpha(tr, BASE); mb = fit(tr[BASE].values, tr.y1.values, ab)
        aa, _ = cv_alpha(tr, AGEONLY); ma = fit(tr[AGEONLY].values, tr.y1.values, aa)
        X = te[FULL].values.copy()
        if shuffle is not None:
            idx = [FULL.index(c) for c in BLOCK]; X[:, idx] = X[shuffle.permutation(len(X))][:, idx]
        te['pm'] = pr(m, X); te['pb'] = pr(mb, te[BASE].values); te['pa'] = pr(ma, te[AGEONLY].values); te['pp'] = te.lb
        te['sd'] = (oof - tr.y1.values).std(); out.append(te)
    return pd.concat(out)
def score(o):
    mae = lambda c: (o[c] - o.y1).abs().mean()
    r = dict(n=len(o), mae_model=mae('pm'), mae_base=mae('pb'), mae_persist=mae('pp'), mae_age=mae('pa'))
    g = o.groupby('t'); r['wins'] = int(g.apply(lambda d: (d.pm - d.y1).abs().mean() < (d.pb - d.y1).abs().mean()).sum())
    r['wins_pos'] = int(g.apply(lambda d: (d.pm - d.y1).abs().mean() < (d.pa - d.y1).abs().mean()).sum()); r['seasons'] = g.ngroups
    return r
if __name__ == '__main__':
    print('pairs', len(P), 'seasons', P.t.min(), P.t.max()); o = run(P); s = score(o); B = {}
    B['A1'] = 1 - s['mae_model'] / s['mae_base']; B['A2'] = s['wins']; B['A3'] = 1 - s['mae_model'] / s['mae_persist']
    def low(c):
        g = []
        for t, d in o[o.age >= 31].groupby('t'):
            d = d.assign(ch=d[c] - d.lb); k = max(1, int(round(len(d) * .2))); g.append((d.nsmallest(k, 'ch').eval('y1 - lb')))
        return pd.concat(g).mean()
    B['A4'] = low('pb') - low('pm'); B['low_model'] = low('pm'); B['low_base'] = low('pb')
    B['A5'] = (np.abs(o.pm - o.y1) <= 1.2816 * o.sd).mean(); B['A7'] = s['wins_pos']
    real = s['mae_base'] - s['mae_model']; rng = np.random.default_rng(7); cnt = 0
    for k in range(200):
        so = score(run(P, shuffle=rng)); cnt += (so['mae_base'] - so['mae_model']) < real
    B['A6'] = cnt / 200
    pas = dict(A1=B['A1'] >= .03, A2=s['wins'] >= 7, A3=B['A3'] >= .08, A4=B['A4'] >= .15, A5=.75 <= B['A5'] <= .85, A6=B['A6'] > .95, A7=s['wins_pos'] >= 6)
    res = dict(score=s, bars=B, passed={k: bool(v) for k, v in pas.items()}, npass=int(sum(pas.values())))
    print(json.dumps(res, indent=1, default=float)); json.dump(res, open(os.path.join(R, 'backtest_summary.json'), 'w'), indent=1, default=float)
    o[['player' if 'player' in o else 'Player', 't', 'age', 'grp', 'lb', 'y1', 'pm', 'pb', 'pa', 'sd']].round(3).to_csv(os.path.join(R, 'backtest_predictions.csv'), index=False)
