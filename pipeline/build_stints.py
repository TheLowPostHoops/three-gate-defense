"""Rebuild lineups, stints and player tallies from NBA play-by-play (stats.nba.com format).
Usage: python3 build_stints.py <season_start_year>
Writes out/stints_<year>.pkl and out/tally_<year>.pkl and prints validation counts."""
import sys, os, pickle, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)

TECH_FOUL = {9, 11, 12, 13, 14, 15, 18, 19, 25}
TECH_FT = {16} | set(range(20, 40))
SIDE = {4: "H", 2: "H", 5: "V", 3: "V"}

def tsec(s):
    m, x = str(s).split(":"); return int(m) * 60 + int(x)

def parse_score(s):
    if not isinstance(s, str) or " - " not in s: return None
    a, b = s.split(" - "); return int(a), int(b)       # visitor - home

def run(year):
    d = pd.read_csv(os.path.join(DATA, f"nbastats_{year}.csv"), low_memory=False)
    d = d[d.EVENTMSGTYPE.isin([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13])]
    cols = ["GAME_ID", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD", "PCTIMESTRING", "HOMEDESCRIPTION", "VISITORDESCRIPTION", "SCORE",
            "PERSON1TYPE", "PLAYER1_ID", "PERSON2TYPE", "PLAYER2_ID", "PERSON3TYPE", "PLAYER3_ID"]
    d = d[cols].copy()
    for c in ["PERSON1TYPE", "PERSON2TYPE", "PERSON3TYPE"]: d[c] = d[c].fillna(0).astype(int)
    for c in ["PLAYER1_ID", "PLAYER2_ID", "PLAYER3_ID"]: d[c] = d[c].fillna(0).astype(np.int64)
    d["HOMEDESCRIPTION"] = d.HOMEDESCRIPTION.fillna(""); d["VISITORDESCRIPTION"] = d.VISITORDESCRIPTION.fillna("")
    # team ids per game (home = person type 4 players' team id)
    stints = []; tally = {}; stats = dict(games=0, bad_games=0, periods=0, periods_fixed=0, periods_bad=0, acts=0, off_court=0, subs=0, sub_bad=0)
    ftab = pd.read_csv(os.path.join(DATA, f"nbastats_{year}.csv"), usecols=["GAME_ID", "PLAYER1_TEAM_ID", "PERSON1TYPE"], low_memory=False)
    ftab = ftab[ftab.PERSON1TYPE.isin([4, 5])].drop_duplicates(["GAME_ID", "PERSON1TYPE"])
    teams = {(g, int(p)): int(t) for g, p, t in zip(ftab.GAME_ID, ftab.PERSON1TYPE, ftab.PLAYER1_TEAM_ID)}

    def T(pid, season_tally):
        return season_tally.setdefault(int(pid), dict(sec=0, fga=0, fgm=0, f3a=0, f3m=0, fta=0, ftm=0, ast=0, tov=0, oreb=0, dreb=0, stl=0, blk=0, ufgm=0, pts=0))

    for gid, g in d.groupby("GAME_ID", sort=True):
        stats["games"] += 1
        ev = list(g.itertuples(index=False))
        home_id, vis_id = teams.get((gid, 4)), teams.get((gid, 5))
        if home_id is None or vis_id is None: stats["bad_games"] += 1; continue
        game_ok = True; game_stints = []
        prev_end = {"H": set(), "V": set()}
        last_score = (0, 0)
        by_period = {}
        for e in ev: by_period.setdefault(e.PERIOD, []).append(e)
        for per in sorted(by_period):
            pe = by_period[per]; stats["periods"] += 1
            plen = 720 if per <= 4 else 300
            # ---- infer starters
            start = {"H": set(), "V": set()}; subbed_in = {"H": set(), "V": set()}
            for e in pe:
                et = e.EVENTMSGTYPE
                if et == 8:
                    s = SIDE.get(e.PERSON1TYPE)
                    if s is None: continue
                    o, i = e.PLAYER1_ID, e.PLAYER2_ID
                    if o not in subbed_in[s]: start[s].add(o)
                    subbed_in[s].add(i)
                elif et in (1, 2, 3, 4, 5, 6, 7, 10):
                    if et == 6 and e.EVENTMSGACTIONTYPE in TECH_FOUL: continue
                    if et == 3 and e.EVENTMSGACTIONTYPE in TECH_FT: continue
                    for ptype, pid in ((e.PERSON1TYPE, e.PLAYER1_ID), (e.PERSON2TYPE, e.PLAYER2_ID), (e.PERSON3TYPE, e.PLAYER3_ID)):
                        if ptype in (4, 5) and pid:
                            s = SIDE[ptype]
                            if pid not in subbed_in[s]: start[s].add(pid)
            lineup = {}
            fixed = False; bad = False
            for s in "HV":
                st = set(start[s])
                if len(st) < 5 and per > 1:
                    fill = [p for p in prev_end[s] if p not in st and p not in subbed_in[s]]
                    st |= set(fill[: 5 - len(st)]); fixed = True
                if len(st) != 5: bad = True
                lineup[s] = st
            if fixed: stats["periods_fixed"] += 1
            if bad:
                stats["periods_bad"] += 1; game_ok = False
                prev_end = {"H": set(), "V": set()}
                continue
            # ---- walk the period
            acc = dict(pts_h=0, pts_v=0, fga_h=0, fga_v=0, fta_h=0, fta_v=0, orb_h=0, orb_v=0, tov_h=0, tov_v=0, trp_h=0, trp_v=0, trt_h=0, trt_v=0)
            cur = [None]
            t_flush = plen; pending = []; last_miss_side = None
            def flush(t_now):
                nonlocal acc, t_flush
                secs = t_flush - t_now
                if secs > 0 or any(acc.values()):
                    ph = acc["fga_h"] + 0.44 * acc["fta_h"] - acc["orb_h"] + acc["tov_h"]
                    pv = acc["fga_v"] + 0.44 * acc["fta_v"] - acc["orb_v"] + acc["tov_v"]
                    game_stints.append((gid, per, tuple(sorted(lineup["H"])), tuple(sorted(lineup["V"])), secs, acc["pts_h"], acc["pts_v"], ph, pv, acc["orb_h"], acc["orb_v"], acc["tov_h"], acc["tov_v"], acc["fta_h"], acc["fta_v"], acc["fga_h"], acc["fga_v"], acc["trp_h"], acc["trp_v"], acc["trt_h"], acc["trt_v"]))
                    for s in "HV":
                        for p in lineup[s]: T(p, tally)["sec"] += secs
                acc = dict(pts_h=0, pts_v=0, fga_h=0, fga_v=0, fta_h=0, fta_v=0, orb_h=0, orb_v=0, tov_h=0, tov_v=0, trp_h=0, trp_v=0, trt_h=0, trt_v=0)
                t_flush = t_now
            def close_poss():
                c = cur[0]
                if c is not None and c["live"] and c["shot"] is not None and c["shot"] <= 8:
                    k = c["side"].lower(); acc["trp_" + k] += 1; acc["trt_" + k] += c["pts"]
                cur[0] = None
            def open_poss(side, t, live):
                close_poss(); cur[0] = dict(side=side, start=t, live=live, shot=None, pts=0)
            def see_action(side, t):
                c = cur[0]
                if c is None or c["side"] != side:
                    open_poss(side, t, False); c = cur[0]
                if c["shot"] is None: c["shot"] = c["start"] - t
            for e in pe:
                et = e.EVENTMSGTYPE; tnow = tsec(e.PCTIMESTRING)
                if et == 8:
                    pending.append(e); continue
                if pending:
                    flush(tnow if False else tsec(pending[0].PCTIMESTRING))
                    for se in pending:
                        s = SIDE.get(se.PERSON1TYPE)
                        stats["subs"] += 1
                        if s is None: continue
                        o, i = se.PLAYER1_ID, se.PLAYER2_ID
                        if o in lineup[s]: lineup[s].discard(o)
                        else: stats["sub_bad"] += 1
                        lineup[s].add(i)
                    pending = []
                    if len(lineup["H"]) != 5 or len(lineup["V"]) != 5: game_ok = False
                if et in (12,): continue
                if et == 13: flush(0); continue
                sc = parse_score(e.SCORE)
                if sc is not None:
                    dv, dh = sc[0] - last_score[0], sc[1] - last_score[1]
                    if 0 <= dv <= 4 and 0 <= dh <= 4:
                        acc["pts_v"] += dv; acc["pts_h"] += dh
                        if cur[0] is not None: cur[0]["pts"] += (dh if cur[0]["side"] == "H" else dv)
                    last_score = sc
                s1 = SIDE.get(e.PERSON1TYPE)
                desc = e.HOMEDESCRIPTION + " " + e.VISITORDESCRIPTION
                # acting-player consistency
                if et in (1, 2, 4, 5) or (et == 3 and e.EVENTMSGACTIONTYPE not in TECH_FT) or (et == 6 and e.EVENTMSGACTIONTYPE not in TECH_FOUL):
                    if e.PERSON1TYPE in (4, 5) and e.PLAYER1_ID:
                        stats["acts"] += 1
                        if e.PLAYER1_ID not in lineup[s1]: stats["off_court"] += 1
                if et in (1, 2) and s1:
                    see_action(s1, tnow)
                    acc["fga_" + s1.lower()] += 1
                    p = T(e.PLAYER1_ID, tally); p["fga"] += 1
                    is3 = "3PT" in desc
                    if is3: p["f3a"] += 1
                    if et == 1:
                        p["fgm"] += 1; p["pts"] += 3 if is3 else 2
                        if is3: p["f3m"] += 1
                        if e.PERSON2TYPE in (4, 5) and e.PLAYER2_ID: T(e.PLAYER2_ID, tally)["ast"] += 1
                        else: p["ufgm"] += 1
                        last_miss_side = None
                        open_poss("V" if s1 == "H" else "H", tnow, False)
                    else:
                        last_miss_side = s1
                        if e.PERSON3TYPE in (4, 5) and e.PLAYER3_ID: T(e.PLAYER3_ID, tally)["blk"] += 1
                elif et == 3 and e.EVENTMSGACTIONTYPE not in TECH_FT and s1:
                    see_action(s1, tnow)
                    acc["fta_" + s1.lower()] += 1
                    p = T(e.PLAYER1_ID, tally); p["fta"] += 1
                    if "MISS" in desc: last_miss_side = s1
                    else: p["ftm"] += 1; p["pts"] += 1; last_miss_side = None
                elif et == 4:
                    s = SIDE.get(e.PERSON1TYPE)
                    if s and last_miss_side:
                        off = (s == last_miss_side)
                        if off: acc["orb_" + s.lower()] += 1
                        else: open_poss(s, tnow, True)
                        if e.PERSON1TYPE in (4, 5) and e.PLAYER1_ID:
                            T(e.PLAYER1_ID, tally)["oreb" if off else "dreb"] += 1
                    last_miss_side = None
                elif et == 5 and s1:
                    acc["tov_" + s1.lower()] += 1
                    open_poss("V" if s1 == "H" else "H", tnow, bool(e.PERSON2TYPE in (4, 5) and e.PLAYER2_ID))
                    if e.PERSON1TYPE in (4, 5) and e.PLAYER1_ID: T(e.PLAYER1_ID, tally)["tov"] += 1
                    if e.PERSON2TYPE in (4, 5) and e.PLAYER2_ID: T(e.PLAYER2_ID, tally)["stl"] += 1
                    last_miss_side = None
            close_poss()
            if pending:
                flush(0)
            prev_end = {"H": set(lineup["H"]), "V": set(lineup["V"])}
        if game_ok:
            stints.extend((year,) + s[:9] + (home_id, vis_id) + s[9:] for s in game_stints)
        else:
            stats["bad_games"] += 1
    return stints, tally, stats

if __name__ == "__main__":
    y = int(sys.argv[1])
    st, ta, stats = run(y)
    pickle.dump(st, open(os.path.join(OUT, f"stints_{y}.pkl"), "wb"))
    pickle.dump(ta, open(os.path.join(OUT, f"tally_{y}.pkl"), "wb"))
    print(y, stats, "stints", len(st), "players", len(ta))
