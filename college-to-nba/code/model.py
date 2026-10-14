"""Step 2: walk-forward test against the bars in PREREG.txt, then project 2026 rookies and the 2027 board.
python3 -I model.py     (needs numpy, pandas, scipy)"""
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

def summarize(full, base):
    r = {}
    mae = lambda t: np.abs(t.pred - t.outcome).mean()
    r['mae_full'], r['mae_base'] = mae(full), mae(base); r['mae_gain_pct'] = 100 * (1 - r['mae_full'] / r['mae_base'])
    r['class_wins'] = int(sum(mae(full[full.year == y]) < mae(base[base.year == y]) for y in full.year.unique()))
    r['spearman_full'], r['spearman_base'] = spearmanr(full.pred, full.outcome)[0], spearmanr(base.pred, base.outcome)[0]
    late = lambda t: t[t.pick >= 21]; q = lambda t: late(t).nlargest(int(np.ceil(len(late(t)) / 4)), 'pred').outcome.mean()
    r['late_top_quarter_full'], r['late_top_quarter_base'] = q(full), q(base)
    z = 1.2816; r['coverage80'] = float(((full.outcome >= full.pred - z * full.sd) & (full.outcome <= full.pred + z * full.sd)).mean())
    return r

full = run(df, FULL); base = run(df, ['lpick']); S = summarize(full, base)
# C6: shuffle the college block among players of the same class, keep each player's draft slot
rng = np.random.default_rng(2026); gains = []
for _ in range(200):
    s = df.copy()
    for y in s.year.unique():
        ix = s.index[s.year == y]; s.loc[ix, COL] = s.loc[rng.permutation(ix), COL].values
    f = run(s, FULL); gains.append(np.abs(base.pred.values - base.outcome.values).mean() - np.abs(f.pred - f.outcome).mean())
real_gain = S['mae_base'] - S['mae_full']; S['shuffle_pctile'] = float((np.array(gains) < real_gain).mean())
# C7: college only (no draft slot) against predicting the training average
co = run(df, COL); avg = full.copy(); 
avg['pred'] = [df[df.year <= y - 4].outcome.mean() for y in avg.year]
S['mae_college_only'] = np.abs(co.pred - co.outcome).mean(); S['mae_average'] = np.abs(avg.pred - avg.outcome).mean()
S['college_only_gain_pct'] = 100 * (1 - S['mae_college_only'] / S['mae_average'])
S['n_test'] = int(len(full)); S['n_all'] = int(len(df)); S['classes_per_test'] = full.groupby('year').size().to_dict()
bars = [
 ('C1 Accuracy: error at least 5% below draft-slot-only', S['mae_gain_pct'] >= 5, f"{S['mae_gain_pct']:.1f}% (full {S['mae_full']:.3f} vs slot-only {S['mae_base']:.3f})"),
 ('C2 Consistency: beats baseline in 4+ of 7 classes', S['class_wins'] >= 4, f"{S['class_wins']} of 7"),
 ('C3 Ranking: Spearman beats baseline by 0.03', S['spearman_full'] - S['spearman_base'] >= .03, f"{S['spearman_full']:.3f} vs {S['spearman_base']:.3f}"),
 ('C4 Late-pick finds: picks 21-60 top quarter beats baseline', S['late_top_quarter_full'] > S['late_top_quarter_base'], f"{S['late_top_quarter_full']:.2f} vs {S['late_top_quarter_base']:.2f}"),
 ('C5 Range honesty: 80% ranges cover 75-85%', .75 <= S['coverage80'] <= .85, f"{100*S['coverage80']:.1f}%"),
 ('C6 Not luck: real gain beats 95% of 200 shuffles', S['shuffle_pctile'] >= .95, f"{100*S['shuffle_pctile']:.0f}th percentile"),
 ('C7 College stats alone: error 10% below the average guess', S['college_only_gain_pct'] >= 10, f"{S['college_only_gain_pct']:.1f}%"),
]
S['bars'] = [dict(bar=b, passed=bool(ok), result=res) for b, ok, res in bars]
json.dump(S, open(os.path.join(R, 'backtest_summary.json'), 'w'), indent=1, default=float)
full.assign(pred_base=base.pred.values)[['year', 'pick', 'player', 'outcome', 'pred', 'pred_base', 'sd']].round(3).to_csv(os.path.join(R, 'backtest_predictions.csv'), index=False)
for b in bars: print('PASS' if b[1] else 'FAIL', b[0], '->', b[2])

# ---- final model on every finished class (2008-2022), then projections
tr = df; a, oof = cv_alpha(tr, FULL); m = fit(tr[FULL].values, tr.outcome.values, a); sd = (oof - tr.outcome.values).std()
mb_a, mb_oof = cv_alpha(tr, ['lpick']); mb = fit(tr[['lpick']].values, tr.outcome.values, mb_a)
coef = pd.Series(m[2], index=FULL).sort_values(); coef.round(3).to_csv(os.path.join(R, 'coefficients.csv'), header=['std_coef'])
print('final alpha', a, 'sd', round(sd, 3)); print(coef.round(2).to_string())
def project(t, label):
    t = t.copy(); has = t[FULL].notna().all(axis=1)
    t['proj'] = np.nan; t.loc[has, 'proj'] = pred(m, t.loc[has, FULL].values)
    t['slot_only'] = pred(mb, t[['lpick']].values)
    t['college_data'] = has; t['proj_final'] = t.proj.fillna(t.slot_only)
    t['sdv'] = np.where(has, sd, (mb_oof - tr.outcome.values).std())
    t['lo80'] = t.proj_final - 1.2816 * t.sdv; t['hi80'] = t.proj_final + 1.2816 * t.sdv
    return t
r26 = project(p[p.year == 2026], 2026)
r26[['pick', 'player', 'team', 'college_data', 'slot_only', 'proj_final', 'lo80', 'hi80']].round(2).sort_values('pick').to_csv(os.path.join(R, 'rookies_2026.csv'), index=False)
print(r26.sort_values('pick')[['pick', 'player', 'proj_final', 'slot_only']].head(12).round(2).to_string())
