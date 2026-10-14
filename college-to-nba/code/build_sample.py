"""Step 1: build the prospect table and the NBA outcome. No modeling here.
python3 build_sample.py <lebron_master.csv>   (LEBRON ratings from BBall Index; outcomes are saved in data/nba_outcomes.csv)"""
import pandas as pd, numpy as np, re, sys, os
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'data')
SUF = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
def key(s):
    t = re.sub(r"[^a-z ]", '', str(s).lower().replace('.', '').replace("'", '')).split()
    while len(t) > 1 and t[-1] in SUF: t.pop()
    return ''.join(t)
def inches(s):
    a, b = str(s).split('-'); return int(a) * 12 + int(b)
dr = pd.read_csv(os.path.join(D, 'drafts_2008_2026.csv')); dr['k'] = dr.player.map(key)
bt = pd.concat([pd.read_csv(os.path.join(D, f), sep='|') for f in ('barttorvik_prospects.csv', 'barttorvik_prospects_suffix.csv')]); bt['k'] = bt.name.map(key)   # second file: players whose college name carries Jr./III and was missed by the first pull
rows = []
for r in dr.itertuples():
    c = bt[(bt.k == r.k) & (bt.year.isin([r.year, r.year - 1]))]
    if c.empty: continue
    last = c.year.max(); c = c[c.year == last]
    c = c.sort_values('minpct', ascending=False).iloc[0]          # same-name collisions: the one with the bigger role
    rows.append(dict(year=r.year, pick=r.pick, player=r.player, k=r.k, **c.drop(['name', 'k', 'year']).to_dict(), cyear=last))
p = pd.DataFrame(rows)
shots = p['2pa'] + p['3pa']
lg = lambda m, a, pr, n: (m + pr * n) / (a + n)                   # shrink small samples toward league average (20 or 50 attempts)
p['age'] = (pd.to_datetime(p.year.astype(str) + '-06-25') - pd.to_datetime(p.bday)).dt.days / 365.25
p['hgt'] = p.ht.map(inches); p['clsn'] = p.cls.map({'Fr': 1, 'So': 2, 'Jr': 3, 'Sr': 4})
p['tpar'] = p['3pa'] / shots.clip(lower=1); p['ftr'] = p.fta / shots.clip(lower=1)
p['p2'] = lg(p['2pm'], p['2pa'], .50, 50); p['p3'] = lg(p['3pm'], p['3pa'], .34, 20); p['pft'] = lg(p.ftm, p.fta, .70, 20)
p['lpick'] = np.log(p.pick)
p.to_csv(os.path.join(D, 'prospects_all.csv'), index=False)
print('drafted 2008-2026:', len(dr), ' with D1 college stats:', len(p))
print(p.groupby('year').size().to_dict())
if len(sys.argv) > 1:
    l = pd.read_csv(sys.argv[1]); l['k'] = l.Player.map(key)
    out = []
    for r in p[p.year <= 2022].itertuples():
        s = l[(l.k == r.k) & (l.year.between(r.year + 2, r.year + 4))]
        mins = s.Minutes.sum(); tot = (s.LEBRON * s.Minutes).sum()
        pad = max(0, 1500 - mins); y = (tot + (-2.0) * pad) / (mins + pad)
        out.append(dict(year=r.year, pick=r.pick, player=r.player, nba_minutes=mins, outcome=y))
    o = pd.DataFrame(out); o.to_csv(os.path.join(D, 'nba_outcomes.csv'), index=False)
    print(o.groupby('year').agg(n=('outcome', 'size'), mean=('outcome', 'mean'), reached=('nba_minutes', lambda x: (x >= 500).mean())).round(2))
