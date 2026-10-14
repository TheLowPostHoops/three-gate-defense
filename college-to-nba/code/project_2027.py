"""Step 3: score the 2027 draft board. Uses the final model from model.py (same features, same penalty rule).
Prospects with a 2025-26 Division I season are scored from last season's numbers; freshmen and players from outside college
basketball are scored from the draft slot alone and marked as such. Slot = rank on the public big board (Tankathon)."""
import numpy as np, pandas as pd, os, re
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_core.py')).read())
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, '..', 'data'); R = os.path.join(H, '..', 'results')
def key(s):
    SUF = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
    t = re.sub(r"[^a-z ]", '', str(s).lower().replace('.', '').replace("'", '')).split()
    while len(t) > 1 and t[-1] in SUF: t.pop()
    return ''.join(t)
bt = pd.concat([pd.read_csv(os.path.join(D, f), sep='|') for f in ('barttorvik_prospects.csv', 'barttorvik_prospects_suffix.csv')]); bt['k'] = bt.name.map(key)
bd = pd.read_csv(os.path.join(D, 'board2027.csv')); bd['k'] = bd.player.map(key)
WRONG_MATCH = {'Cameron Williams'}                  # the Duke freshman is not the 2025-26 Portland player
rows = []
for r in bd.itertuples():
    c = bt[(bt.k == r.k) & (bt.year == 2026)]
    if r.player in WRONG_MATCH: c = c.iloc[0:0]      # same name, different player (checked by school and height)
    rec = dict(rank=r.rank, player=r.player, school=r.school, college_data=False)
    if len(c):
        c = c.sort_values('minpct', ascending=False).iloc[0]
        shots = c['2pa'] + c['3pa']; ht = c.ht.split('-')
        rec.update(college_data=True, team=c.team, age=(pd.Timestamp('2027-06-25') - pd.Timestamp(c.bday)).days / 365.25,
                   hgt=int(ht[0]) * 12 + int(ht[1]), clsn=min(4, {'Fr': 1, 'So': 2, 'Jr': 3, 'Sr': 4}[c.cls] + 1), minpct=c.minpct, usg=c.usg, ts=c.ts,
                   ortg=c.ortg, orb=c.orb, drb=c.drb, ast=c.ast, to=c.to, stl=c.stl, blk=c.blk, tpar=c['3pa'] / max(shots, 1), ftr=c.fta / max(shots, 1),
                   p2=(c['2pm'] + 25) / (c['2pa'] + 50), p3=(c['3pm'] + 6.8) / (c['3pa'] + 20), pft=(c.ftm + 14) / (c.fta + 20), games=c.gp)
    rows.append(rec)
b = pd.DataFrame(rows); b['lpick'] = np.log(b['rank'])
out = project(b)
cols = ['rank', 'player', 'school', 'college_data', 'slot_only', 'proj_final', 'lo80', 'hi80']
out[cols].round(2).to_csv(os.path.join(R, 'board_2027.csv'), index=False)
print(out[cols].round(2).to_string())
