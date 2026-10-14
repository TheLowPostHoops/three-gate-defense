"""Box scores for the season being played (NBA live data CDN), kept in pipeline/live/box_2026.csv.gz.

    python3 fc_live.py          # downloads box scores for finished games not yet saved

Used by the forecast for games played, starts, points and a box-score rating so far.
"""
import os, json, re, time, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); LIVE = os.path.join(HERE, "live"); OUT = os.path.join(HERE, "out")
SEASON = 2026
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{}.json"
COLS = ["gid", "date", "pid", "name", "team", "starter", "played", "reason", "min", "pts", "oreb", "dreb", "ast", "stl", "blk", "tov", "fga", "fta", "f3a", "f3m", "pm"]
PATH = os.path.join(LIVE, f"box_{SEASON}.csv.gz")


def mins(s):
    m = re.match(r"PT(\d+)M([\d.]+)S", s or "")
    return int(m.group(1)) + float(m.group(2)) / 60 if m else 0.0


def load():
    return pd.read_csv(PATH) if os.path.exists(PATH) else pd.DataFrame(columns=COLS)


def update(schedule):
    """schedule: DataFrame from fc_sim.load_schedule (live source, with gid and status)."""
    os.makedirs(LIVE, exist_ok=True)
    B = load(); have = set(B.gid.astype(str))
    todo = schedule[(schedule.status == 3) & (schedule.gid.astype(str) != "") & ~schedule.gid.astype(str).isin(have)]
    rows = []
    for x in todo.itertuples():
        try:
            j = json.loads(urllib.request.urlopen(urllib.request.Request(BOX_URL.format(x.gid), headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read())["game"]
        except Exception as e:
            print("box score", x.gid, "failed:", str(e)[:80]); continue
        for side in ("homeTeam", "awayTeam"):
            t = j[side]
            for p in t["players"]:
                s = p.get("statistics", {})
                rows.append(dict(gid=str(x.gid), date=str(x.date.date()), pid=int(p["personId"]), name=p.get("name", ""), team=t["teamTricode"],
                                 starter=int(p.get("starter") == "1"), played=int(p.get("played") == "1"), reason=p.get("notPlayingReason", "") or "",
                                 min=mins(s.get("minutes")), pts=s.get("points", 0), oreb=s.get("reboundsOffensive", 0), dreb=s.get("reboundsDefensive", 0),
                                 ast=s.get("assists", 0), stl=s.get("steals", 0), blk=s.get("blocks", 0), tov=s.get("turnovers", 0),
                                 fga=s.get("fieldGoalsAttempted", 0), fta=s.get("freeThrowsAttempted", 0), f3a=s.get("threePointersAttempted", 0),
                                 f3m=s.get("threePointersMade", 0), pm=s.get("plusMinusPoints", 0)))
        time.sleep(0.3)
    if rows:
        B = pd.concat([B, pd.DataFrame(rows)], ignore_index=True)
        B.to_csv(PATH, index=False)
    print("box scores saved:", B.gid.nunique(), "games (+%d)" % len(todo), flush=True)
    return B


def so_far(B, spm):
    """per player: games, starts, minutes, points per game, and a box-score rating (offense, defense) from the box prior."""
    if not len(B): return pd.DataFrame()
    P = B[B.played == 1].groupby("pid").agg(gp=("gid", "nunique"), starts=("starter", "sum"), min=("min", "sum"), pts=("pts", "sum"), oreb=("oreb", "sum"),
                                         dreb=("dreb", "sum"), ast=("ast", "sum"), stl=("stl", "sum"), blk=("blk", "sum"), tov=("tov", "sum"),
                                         fga=("fga", "sum"), fta=("fta", "sum"), f3a=("f3a", "sum"), f3m=("f3m", "sum"), team=("team", "last"), name=("name", "last"))
    poss = P["min"] * 100 / 48                                       # about 100 possessions per 48 minutes
    box = ["pts", "fga", "fta", "f3a", "f3m", "ast", "tov", "oreb", "dreb", "stl", "blk"]
    F = np.column_stack([100 * P[k] / poss.clip(lower=1) for k in box])
    P["o_box"] = spm["o"][0] + F @ np.array(spm["o"][1]); P["d_box"] = -(spm["d"][0] + F @ np.array(spm["d"][1]))
    P["poss"] = poss; P["ppg"] = P.pts / P.gp; P["mpg"] = P["min"] / P.gp
    P["stocks"] = (P.stl + P.blk) / P.gp; P["dreb_pg"] = P.dreb / P.gp
    # games each team has played, and the current injury list (players listed out for their team's last game)
    last = B.sort_values("date").groupby("team").gid.last()
    out_now = B[B.gid.isin(last.values) & (B.played == 0) & B.reason.str.contains("INJURY", na=False)].pid
    P["out_now"] = P.index.isin(out_now)
    return P


if __name__ == "__main__":
    import fc_sim
    S = fc_sim.load_schedule()
    if S.source.iloc[0] == "live": update(S)
    else: print("no live schedule; box scores not updated")
