"""Step 4: package results for the website page (college/data.json)."""
import json, os, pandas as pd, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'data'); R = os.path.join(H, '..', 'results')
S = json.load(open(os.path.join(R, 'backtest_summary.json')))
r26 = pd.read_csv(os.path.join(R, 'rookies_2026.csv')); d26 = pd.read_csv(os.path.join(D, 'draft2026.csv'))
full = d26.merge(r26.drop(columns=['team']), on='pick', how='left', suffixes=('', '_m')).drop(columns=['player_m'])
import sys; sys.path.insert(0, H)
from importlib import util
sd_slot = None
full['college_data'] = full.college_data.fillna(False).astype(bool)
# picks with no U.S. college season get a draft-slot-only projection with the slot-only range
core = open(os.path.join(H, 'model_core.py')).read(); ns = {'__file__': os.path.join(H, 'model_core.py')}; exec(core, ns)
miss = ~full.college_data
full.loc[miss, 'slot_only'] = ns['pred'](ns['mb'], np.log(full.loc[miss, 'pick'].values)[:, None])
sdb = (ns['mb_oof'] - ns['tr'].outcome.values).std()
full.loc[miss, 'proj_final'] = full.loc[miss, 'slot_only']; full.loc[miss, 'lo80'] = full.loc[miss, 'proj_final'] - 1.2816 * sdb; full.loc[miss, 'hi80'] = full.loc[miss, 'proj_final'] + 1.2816 * sdb
rk = lambda df, cols: [{c: (None if pd.isna(v) else (round(float(v), 2) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else (int(v) if isinstance(v, (int, np.integer)) else v)))) for c, v in zip(cols, row)} for row in df[cols].itertuples(index=False)]
b27 = pd.read_csv(os.path.join(R, 'board_2027.csv'))
out = dict(bars=S['bars'], n_test=S['n_test'], n_all=S['n_all'], mae_full=round(S['mae_full'], 3), mae_base=round(S['mae_base'], 3), gain=round(S['mae_gain_pct'], 1),
           spearman_full=round(S['spearman_full'], 3), spearman_base=round(S['spearman_base'], 3), coverage=round(100 * S['coverage80'], 1), sd=round(float(ns['sd']), 2),
           rookies=rk(full, ['pick', 'player', 'school', 'college_data', 'slot_only', 'proj_final', 'lo80', 'hi80']),
           board=rk(b27, ['rank', 'player', 'school', 'college_data', 'slot_only', 'proj_final', 'lo80', 'hi80']))
out['verdict'] = ("Against the draft slot alone, the model's average miss was %.1f%% smaller (%.2f vs %.2f rating points) over %d players in the 2016-2022 classes, it was better in %d of 7 classes, and it ordered players better (rank correlation %.2f vs %.2f). Shuffling the college numbers 200 times never matched it. It failed two bars: it did not find late-pick players better than slot alone (picks 21 to 60), and its 80%% ranges held %.1f%% of players, a little too wide. The two close passes (accuracy 5.3%% against a 5%% bar, college-only 10.1%% against 10%%) are thin. A first run that missed 8 players whose college names carry Jr. or III failed both; the matching was fixed and the run repeated with the bars unchanged. Both runs are in the repo." % (S['mae_gain_pct'], S['mae_full'], S['mae_base'], S['n_test'], S['class_wins'], S['spearman_full'], S['spearman_base'], 100 * S['coverage80']))
out['learned'] = "<b>What moved projections.</b> After draft slot, the college numbers that pushed a player up were assist rate, block rate, steal rate and rebounding. Being older at the draft and very high usage pushed slightly down. The size of each effect is small next to draft slot, which is why the model nudges the slot instead of replacing it. College three-point percentage pointed slightly down once the other shooting numbers were counted, which is more likely overlap between those numbers than a real rule."
os.makedirs(os.path.join(H, '..', 'site'), exist_ok=True)
json.dump(out, open(os.path.join(H, '..', 'site', 'data.json'), 'w'), separators=(',', ':'), ensure_ascii=False)
print(len(out['rookies']), len(out['board']), sum(not r['college_data'] for r in out['rookies']))
