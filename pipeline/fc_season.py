"""2026-27 inputs for the forecast: projected player ratings, minutes, availability, team strengths.

    python3 fc_season.py        # writes out/fc_players_2026.csv and out/fc_teams_2026.csv
"""
import os, json
import numpy as np, pandas as pd
import fc_project as FP

HERE = FP.HERE; OUT = FP.OUT; INP = FP.INP
SEASON = 2026
LEAGUE_AVAIL = 0.86          # share of team games a rotation player plays (league norm, used to shrink)


def abbr():
    return {int(k): v for k, v in json.load(open(os.path.join(HERE, "coaches.json")))["teams"].items()}


def roster():
    ros = json.load(open(os.path.join(INP, "rosters_2026.json")))
    A = abbr()
    return pd.DataFrame([dict(pid=int(r[1]), name=r[2], team=A.get(int(r[0]), "?"), team_id=int(r[0]), how=r[10], pos=r[4]) for r in ros]).set_index("pid")


def availability(pids):
    """share of team games played, last three seasons weighted 3/2/1, a season missed entirely counts as 0 when he played
    before and after; shrunk toward the league norm with weight 1.5 seasons."""
    rates = {}
    for k, w in ((SEASON - 1, 3), (SEASON - 2, 2), (SEASON - 3, 1)):
        g = FP.games(k)
        tg = pd.read_csv(os.path.join(OUT, f"teamgames_{k}.csv.gz")); tg = tg[tg.kind == "rs"].groupby("team").gid.nunique()
        for p in pids:
            if p in g.index:
                rates.setdefault(p, []).append((w, min(1.0, g.loc[p, "gp"] / tg.get(g.loc[p, "team"], 82))))
    out = {}
    seen = {p: [k for k in range(SEASON - 3, SEASON) if p in FP.games(k).index] for p in pids}
    for p in pids:
        r = rates.get(p, [])
        # whole season missed between seasons he played (or the last season, if he played the one before): 0
        s = seen[p]
        if s and (SEASON - 1) not in s and (SEASON - 2) in s: r = r + [(3, 0.0)]
        num = sum(w * x for w, x in r) + 1.5 * LEAGUE_AVAIL; den = sum(w for w, _ in r) + 1.5
        out[p] = float(np.clip(num / den, 0.3, 0.97))
    return pd.Series(out)


def build():
    R = roster()
    M = FP.fit_player_model(range(2013, SEASON))
    P = FP.project(SEASON, M, list(R.index))
    mpg = FP.expected_mpg(SEASON, list(R.index), P)
    av = availability(list(R.index))
    # training-camp invitees: never played an NBA game and not picked in the last two drafts. Past rosters in the
    # backtest only contain players who actually played, so these players are given 1 minute a night, not 10.
    played = set()
    for k in range(2010, SEASON): played |= set(FP.games(k).index)
    dr = FP.draft()
    camp = [p for p in R.index if p not in played and not (p in dr and dr[p][0] >= SEASON - 1)]
    mpg.loc[[p for p in camp if p in mpg.index]] = 1.0
    D = R.join(P[["o", "d", "net", "rookie"]], how="left")
    D["rookie"] = D.rookie.fillna(False)
    D["mpg"] = mpg; D["avail"] = av
    D["age"] = [FP.age_at(p, SEASON) for p in D.index]
    T, DD = FP.team_strength(P, R.team, mpg, av)
    D["min_share"] = DD.m
    bt = json.load(open(os.path.join(OUT, "fc_backtest.json")))
    c = bt["scale"]
    T["net_c"] = T.net - T.net.mean()
    T["mov"] = c["k"] * T.net_c
    # surprise-factor inputs (S3): returning minutes, minutes-weighted age, minutes to players with no NBA rating history
    prev = FP.games(SEASON - 1); tot_prev = prev.groupby("team").secs.sum() / 60; A = abbr()
    prev_team = prev.team.map(A)
    T["cont"] = [sum(prev.loc[p, "secs"] / 60 for p in R[R.team == tm].index if p in prev.index and prev_team.get(p) == tm) / tot_prev.get(next(k for k, v in A.items() if v == tm), np.nan) for tm in T.index]
    DD["age"] = D.age.reindex(DD.index).fillna(26); DD["new"] = ~DD.index.isin(FP.ratings(SEASON - 1).index)
    T["age"] = [float(np.average(DD[DD.team == tm].age, weights=DD[DD.team == tm].m)) for tm in T.index]
    T["rook"] = [float(DD[(DD.team == tm) & DD.new].m.sum() / 240) for tm in T.index]
    D.to_csv(os.path.join(OUT, f"fc_players_{SEASON}.csv")); T.to_csv(os.path.join(OUT, f"fc_teams_{SEASON}.csv"))
    json.dump({k: [float(v) for v in vv] if hasattr(vv, "__len__") else float(vv) for k, vv in M.items()}, open(os.path.join(OUT, "fc_player_model.json"), "w"))
    return D, T


if __name__ == "__main__":
    D, T = build()
    print(T.sort_values("mov", ascending=False)[["mov"]].round(1).T.to_string())
    print(D.sort_values("net", ascending=False)[["name", "team", "o", "d", "net", "mpg", "avail", "age"]].head(25).round(2).to_string())
    print(D[D.rookie].sort_values("net", ascending=False)[["name", "team", "net", "mpg"]].head(8).round(1).to_string())
