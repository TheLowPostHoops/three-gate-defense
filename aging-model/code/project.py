"""Final model on every finished season pair, 2026-27 projections for players 26+, and the average age curve."""
import json, os, numpy as np, pandas as pd
from model import *
a, oof = cv_alpha(P, FULL); m = fit(P[FULL].values, P.y1.values, a); sd = (oof - P.y1.values).std()
print('alpha', a, 'sd', round(sd, 3))
c = L[(L.year == 2026) & (L.mins >= 500) & (L.age >= 26)].copy()
prev = L[L.year == 2025].set_index('id').lb
c['prev'] = c.id.map(prev).fillna(c.lb); c['lmin'] = np.log(c.mins); c['G'] = (c.grp == 'G') * 1.0; c['Fw'] = (c.grp == 'F') * 1.0; c = add_age(c)
c['proj'] = pr(m, c[FULL].values); c['lo'] = c.proj - 1.2816 * sd; c['hi'] = c.proj + 1.2816 * sd; c['gain'] = c.proj - c.lb
out = c[['Player', 'team', 'age', 'Pos', 'mins', 'lb', 'proj', 'gain', 'lo', 'hi']].rename(columns={'Player': 'player', 'Pos': 'pos', 'mins': 'min'}).sort_values('gain')
out.round(2).to_csv(os.path.join(R, 'projections_2026_27.csv'), index=False); print(len(out)); print(out.head(12).round(2).to_string())
# description: average actual change by age and position group, all ages in the sample (not a bar)
Z = A.copy(); Z['ch'] = Z.y1 - Z.lb; Z['ageb'] = Z.age.clip(19, 37).astype(int)
cur = Z.groupby(['grp', 'ageb']).ch.agg(['mean', 'count']).reset_index(); cur = cur[cur['count'] >= 15]; cur.round(3).to_csv(os.path.join(R, 'age_curve.csv'), index=False)
al = Z.groupby('ageb').ch.agg(['mean', 'count']).round(3); print(al.to_string())
