"""Final model on every finished season pair, then 2026-27 projections for players 25 or younger."""
import json, os, numpy as np, pandas as pd
from model import *
a, oof = cv_alpha(P, FULL); m = fit(P[FULL].values, P.y1.values, a); sd = (oof - P.y1.values).std()
coef = pd.Series(m[2], index=FULL).sort_values(); coef.round(3).to_csv(os.path.join(R, 'coefficients.csv'), header=['std_coef'])
print('alpha', a, 'sd', round(sd, 3)); print(coef.round(2).to_string())
c = L[(L.year == 2026) & (L.mins >= 500) & (L.age <= 25)].copy()
prev = L[L.year == 2025].set_index('id').lb
c['prev'] = c.id.map(prev).fillna(c.lb); c['lmin'] = np.log(c.mins); c['lpick'] = np.log(c.pick)
c['yrs'] = (2026 - c.dyear).where(c.dyear.notna(), c.age - 20).clip(lower=0); c['G'] = (c.grp == 'G') * 1.0; c['Fw'] = (c.grp == 'F') * 1.0
c['lb'] = c.lb; c['proj'] = pr(m, c[FULL].values); c['lo'] = c.proj - 1.2816 * sd; c['hi'] = c.proj + 1.2816 * sd
c['gain'] = c.proj - c.lb
out = c[['Player', 'team', 'age', 'Pos', 'mins', 'lb', 'proj', 'gain', 'lo', 'hi']].rename(columns={'Player': 'player', 'Pos': 'pos', 'mins': 'min'}).sort_values('gain', ascending=False)
out.round(2).to_csv(os.path.join(R, 'projections_2026_27.csv'), index=False)
print(len(out)); print(out.head(15).round(2).to_string())
