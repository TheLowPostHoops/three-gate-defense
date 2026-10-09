"""Convert cdn.nba.com-style play-by-play (shufinskiy cdnnba_YYYY.csv) to the stats.nba.com column layout build_stints.py expects."""
import sys, re, numpy as np, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, low_memory=False)
d = d.sort_values(["gameId", "orderNumber"]).reset_index(drop=True)
def clk(s):
    m = re.match(r"PT(\d+)M(\d+)", str(s)); return "%d:%02d" % (int(m.group(1)), int(m.group(2))) if m else "0:00"
rows = []; bad = 0
for gid, g in d.groupby("gameId", sort=True):
    g = g.reset_index(drop=True)
    sh = g[(g.actionType.isin(["2pt", "3pt"])) & (g.shotResult == "Made")]
    home = None
    prev = (0, 0)
    for r in g.itertuples():
        if r.actionType in ("2pt", "3pt", "freethrow") and r.shotResult == "Made" and pd.notna(r.teamId):
            pass
    # home team = team whose made shots raise scoreHome
    sc = g[["scoreHome", "scoreAway"]].astype(float).values; tid = g.teamId.values
    cnt = {}
    for i in range(1, len(g)):
        if g.actionType[i] in ("2pt", "3pt", "freethrow") and g.shotResult[i] == "Made" and pd.notna(tid[i]):
            if sc[i, 0] > sc[i - 1, 0]: cnt[int(tid[i])] = cnt.get(int(tid[i]), 0) + 1
    if not cnt: bad += 1; continue
    home = max(cnt, key=cnt.get)
    teams = [int(t) for t in g.teamId.dropna().unique() if int(t) != 0]
    away = [t for t in teams if t != home]
    if len(away) != 1: bad += 1; continue
    away = away[0]
    side = lambda t: 4 if int(t) == home else 5
    opp = lambda t: 5 if int(t) == home else 4
    pteam = {}
    for p, t in zip(g.personId, g.teamId):
        if pd.notna(t) and p and p != 0: pteam[int(p)] = int(t)
    out = []
    def emit(r, et, at, p1=0, t1=0, p2=0, t2=0, p3=0, t3=0, desc=None):
        t = int(r.teamId) if pd.notna(r.teamId) else home
        ds = desc if desc is not None else (r.description if isinstance(r.description, str) else "")
        out.append(dict(GAME_ID=int(gid), EVENTNUM=int(r.actionNumber), EVENTMSGTYPE=et, EVENTMSGACTIONTYPE=at, PERIOD=int(r.period), PCTIMESTRING=clk(r.clock),
                        HOMEDESCRIPTION=ds if t == home else "", VISITORDESCRIPTION=ds if t != home else "",
                        SCORE="%d - %d" % (int(r.scoreAway), int(r.scoreHome)), PERSON1TYPE=t1, PLAYER1_ID=p1, PERSON2TYPE=t2, PLAYER2_ID=p2, PERSON3TYPE=t3, PLAYER3_ID=p3,
                        PLAYER1_TEAM_ID=t))
    subs_done = set()
    lastfoul = None
    for i, r in enumerate(g.itertuples()):
        a = r.actionType; t = int(r.teamId) if pd.notna(r.teamId) else None
        pid = int(r.personId) if pd.notna(r.personId) else 0
        if a in ("2pt", "3pt", "heave"):
            et = 1 if r.shotResult == "Made" else 2
            ds = (r.description if isinstance(r.description, str) else "")
            if a != "2pt" and "3PT" not in ds: ds += " 3PT"
            p2 = int(r.assistPersonId) if pd.notna(r.assistPersonId) else 0
            p3 = int(r.blockPersonId) if pd.notna(r.blockPersonId) else 0
            emit(r, et, 1, pid, side(t), p2, side(t) if p2 else 0, p3, opp(t) if p3 else 0, ds)
        elif a == "freethrow":
            ds = r.description if isinstance(r.description, str) else ""
            if r.shotResult != "Made" and "MISS" not in ds: ds = "MISS " + ds
            at = 16 if (lastfoul is not None and lastfoul[0] == "technical" and lastfoul[1] == r.clock and str(r.subType) == "1 of 1") else 11
            emit(r, 3, at, pid, side(t), desc=ds)
        elif a == "rebound":
            if pid == 0 or "TEAM" in str(r.description): emit(r, 4, 0, t, 2 if t == home else 3)
            else: emit(r, 4, 0, pid, side(t))
        elif a == "turnover":
            ds = r.description if isinstance(r.description, str) else ""
            p2 = int(r.stealPersonId) if pd.notna(r.stealPersonId) else 0
            if pid == 0: emit(r, 5, 0, t, 2 if t == home else 3, p2, opp(t) if p2 else 0, desc=ds)
            else: emit(r, 5, 0, pid, side(t), p2, opp(t) if p2 else 0, desc=ds)
        elif a == "foul":
            st = str(r.subType); ds_ = str(r.descriptor); lastfoul = (st, r.clock)
            # stats.nba.com action codes: 2 shooting, 3 loose ball, 4 offensive, 26 offensive charge, 11 technical, 1 other personal
            if st == "technical": at = 11
            elif st == "offensive": at = 26 if ds_ == "charge" else 4
            else: at = 2 if ds_ == "shooting" else (3 if ds_ == "loose ball" else 1)
            dr = int(r.foulDrawnPersonId) if pd.notna(r.foulDrawnPersonId) else 0
            if pid: emit(r, 6, at, pid, side(t), dr, opp(t) if dr else 0)
        elif a == "violation":
            if pid: emit(r, 7, 1, pid, side(t))
        elif a == "substitution":
            if i in subs_done: continue
            if r.subType == "out":
                # find matching "in" row later with same team and clock
                for j in range(i + 1, min(i + 40, len(g))):
                    if j not in subs_done and g.actionType[j] == "substitution" and g.subType[j] == "in" and g.teamId[j] == r.teamId and g.clock[j] == r.clock:
                        subs_done.add(j); pin = int(g.personId[j]); emit(r, 8, 0, pid, side(t), pin, side(t)); break
            else:
                for j in range(i + 1, min(i + 40, len(g))):
                    if j not in subs_done and g.actionType[j] == "substitution" and g.subType[j] == "out" and g.teamId[j] == r.teamId and g.clock[j] == r.clock:
                        subs_done.add(j); pout = int(g.personId[j]); emit(r, 8, 0, pout, side(t), pid, side(t)); break
        elif a == "jumpball":
            w = int(r.jumpBallWonPersonId) if pd.notna(r.jumpBallWonPersonId) else 0
            l = int(r.jumpBallLostPersonId) if pd.notna(r.jumpBallLostPersonId) else 0
            tw = pteam.get(w); tl = pteam.get(l)
            emit(r, 10, 0, w, side(tw) if tw else 0, l, side(tl) if tl else 0)
        elif a == "period":
            emit(r, 12 if r.subType == "start" else 13, 0)
        elif a == "timeout":
            emit(r, 9, 0)
    rows.extend(out)
o = pd.DataFrame(rows)
o.to_csv(dst, index=False); print(dst, o.shape, "bad games", bad, "games", o.GAME_ID.nunique())
print(o.EVENTMSGTYPE.value_counts().to_dict())
