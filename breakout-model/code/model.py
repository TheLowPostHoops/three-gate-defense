"""Breakout model. Rules are in ../PREREG.txt (written before the first run)."""
import numpy as np, pandas as pd, os, re, json, unicodedata
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'data'); R = os.path.join(H, '..', 'results')
ALPHAS = [1, 3, 10, 30, 100, 300, 1000, 3000]
def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'[^a-z ]', '', s.replace('-', ' ')); s = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', s)
    return re.sub(r'\s+', ' ', s).strip()
L = pd.read_csv(os.path.join(D, 'lebron_master.csv'))
L = L.rename(columns={'O-LEBRON': 'olb', 'D-LEBRON': 'dlb', 'NBA ID': 'id', 'LEBRON': 'lb', 'Minutes': 'mins', 'Age': 'age'})
# players traded mid-year can appear once per team: keep the full-season row if present (no duplicates found in the id/year key)
L['key'] = L.Player.map(norm)
dr = pd.read_csv(os.path.join(D, 'drafts_2008_2026.csv')); dr['key'] = dr.player.map(norm)
dr = dr.sort_values('year').drop_duplicates('key', keep='last')   # one draft record per name
pk = dict(zip(dr.key, dr.pick)); dy = dict(zip(dr.key, dr.year))
L['pick'] = L.key.map(pk).fillna(61); L['dyear'] = L.key.map(dy)
# older drafts (before 2008) are not in the draft file: treat as undrafted-looking only if age says rookie-ish; use age instead (see features)
L['grp'] = L.Pos.map(lambda p: 'G' if p in ('PG', 'SG') else ('F' if p in ('SF', 'PF') else 'C'))
Lk = L.set_index(['id', 'year'])
rows = []
for (i, y), r in Lk.iterrows():
    if r.mins < 500 or r.age > 25: continue
    nx = Lk.index.isin([(i, y + 1)]).any() and Lk.loc[(i, y + 1)] if (i, y + 1) in Lk.index else None
    if nx is None or nx.mins < 300: continue
    pv = Lk.loc[(i, y - 1)].lb if (i, y - 1) in Lk.index else r.lb
    rows.append(dict(id=i, player=r.Player, t=y, age=r.age, lb=r.lb, olb=r.olb, dlb=r.dlb, lmin=np.log(r.mins), prev=pv, pick=r.pick, dyear=r.dyear,
                     yrs=(y - r.dyear) if pd.notna(r.dyear) else np.nan, g=r.grp, y1=nx.lb))
P = pd.DataFrame(rows)
P['lpick'] = np.log(P.pick); P['yrs'] = P.yrs.fillna(P.age - 20).clip(lower=0)   # drafted before 2008: estimate from age
P['G'] = (P.g == 'G') * 1.0; P['Fw'] = (P.g == 'F') * 1.0
FULL = ['lb', 'olb', 'dlb', 'age', 'lmin', 'prev', 'yrs', 'lpick', 'G', 'Fw']; BASE = ['lb']; EXTRA = [c for c in FULL if c not in BASE]
def fit(X, y, a):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; ym = y.mean()
    return mu, sd, np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - ym)), ym
def pr(m, X): return ((X - m[0]) / m[1]) @ m[2] + m[3]
def cv_alpha(tr, cols):
    best, bs, oof = None, 9e9, None
    for a in ALPHAS:
        e = np.zeros(len(tr))
        for t in tr.t.unique():
            te = (tr.t == t).values
            e[te] = pr(fit(tr.loc[~te, cols].values, tr.y1.values[~te], a), tr.loc[te, cols].values)
        s = np.abs(e - tr.y1.values).mean()
        if s < bs: best, bs, oof = a, s, e
    return best, oof
def run(data, tests=range(2016, 2026), shuffle=None):
    out = []
    for t in tests:
        tr = data[data.t < t]; te = data[data.t == t].copy()      # train targets: t_train+1 <= t
        tr = tr[tr.t + 1 <= t]
        d = data
        a, oof = cv_alpha(tr, FULL); m = fit(tr[FULL].values, tr.y1.values, a)
        ab, oofb = cv_alpha(tr, BASE); mb = fit(tr[BASE].values, tr.y1.values, ab)
        X = te[FULL].values.copy()
        if shuffle is not None:
            for c in EXTRA:
                j = FULL.index(c); X[:, j] = shuffle.permutation(X[:, j])
        te['pm'] = pr(m, X); te['pb'] = pr(mb, te[BASE].values); te['pp'] = te.lb
        te['sd'] = (oof - tr.y1.values).std(); te['alpha'] = a; out.append(te)
    return pd.concat(out)
def score(o):
    mae = lambda c: (o[c] - o.y1).abs().mean()
    r = dict(n=len(o), mae_model=mae('pm'), mae_base=mae('pb'), mae_persist=mae('pp'))
    per = o.groupby('t').apply(lambda d: (d.pm - d.y1).abs().mean() < (d.pb - d.y1).abs().mean()); r['wins'] = int(per.sum()); r['seasons'] = len(per)
    return r
if __name__ == '__main__':
    print('pairs', len(P), 'seasons', P.t.min(), P.t.max()); o = run(P); s = score(o)
    B = {}
    B['B1'] = 1 - s['mae_model'] / s['mae_base']; B['B2'] = s['wins']; B['B3'] = 1 - s['mae_model'] / s['mae_persist']
    def top(c):
        g = []
        for t, d in o.groupby('t'):
            d = d.assign(gain=d[c] - d.lb); k = max(1, int(round(len(d) * .1))); g.append((d.nlargest(k, 'gain').eval('y1 - lb')))
        return pd.concat(g).mean()
    B['B4'] = top('pm') - top('pb'); B['top_model'] = top('pm'); B['top_base'] = top('pb')
    B['B5'] = (np.abs(o.pm - o.y1) <= 1.2816 * o.sd).mean()
    real = s['mae_base'] - s['mae_model']; rng = np.random.default_rng(7); cnt = 0
    for k in range(200):
        so = score(run(P, shuffle=rng)); cnt += (so['mae_base'] - so['mae_model']) < real
    B['B6'] = cnt / 200
    hi = o[o.pm - o.lb >= 1.0]; B['B7_n'] = len(hi); B['B7_gain'] = (hi.y1 - hi.lb).mean() if len(hi) else np.nan; B['B7_share'] = ((hi.y1 - hi.lb) > 0).mean() if len(hi) else np.nan
    pas = dict(B1=B['B1'] >= .03, B2=s['wins'] >= 7, B3=B['B3'] >= .10, B4=B['B4'] >= .20, B5=.75 <= B['B5'] <= .85, B6=B['B6'] > .95,
               B7=bool(len(hi) and B['B7_gain'] >= .5 and B['B7_share'] >= .6))
    res = dict(score=s, bars=B, passed=pas, npass=int(sum(pas.values())))
    print(json.dumps(res, indent=1, default=float)); json.dump(res, open(os.path.join(R, 'backtest_summary.json'), 'w'), indent=1, default=float)
    o[['player', 't', 'age', 'lb', 'y1', 'pm', 'pb', 'sd']].round(3).to_csv(os.path.join(R, 'backtest_predictions.csv'), index=False)
