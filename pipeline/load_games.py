"""Player-game table for the Load & Durability model.

    python pipeline/load_games.py 2010 2025      # build seasons 2010-11 through 2025-26

For each season writes pipeline/out/games_<year>.csv.gz with one row per player per team game:
season, game id, date, regular season or playoffs, team, player, appeared (0/1), seconds and possessions on the floor.
Possessions on the floor = both teams' possessions while he was on the court, so a faster game counts for more.
Games whose lineups could not be rebuilt keep the appearance but leave seconds and possessions empty.
"""
import os, sys, subprocess, pickle
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from update import get_list, fetch, DATA, OUT, log   # noqa: E402
import build_stints   # noqa: E402

PTYPES = (4, 5)


def pbp_for(key, avail):
    """stats.nba.com-format play-by-play for one dataset key (cdn data converted on the fly)."""
    if f"nbastats_{key}" in avail:
        return pd.read_csv(fetch(f"nbastats_{key}"), low_memory=False)
    if f"cdnnba_{key}" in avail:
        src = fetch(f"cdnnba_{key}"); dst = os.path.join(DATA, f"conv_{key}.csv")
        subprocess.run([sys.executable, os.path.join(HERE, "convert_cdn.py"), src, dst], check=True, stdout=subprocess.DEVNULL)
        return pd.read_csv(dst, low_memory=False)
    return None


def dates_for(key, avail, pbp):
    """game id -> date. Shot detail carries GAME_DATE; converted cdn data may carry it too."""
    for c in ("GAME_DATE", "gameDate", "GAME_DATE_EST"):
        if c in pbp.columns:
            s = pbp[["GAME_ID", c]].dropna().drop_duplicates("GAME_ID")
            return dict(zip(s.GAME_ID.astype(int), pd.to_datetime(s[c].astype(str).str[:10])))
    if f"shotdetail_{key}" in avail:
        s = pd.read_csv(fetch(f"shotdetail_{key}"), usecols=["GAME_ID", "GAME_DATE"], low_memory=False).drop_duplicates("GAME_ID")
        return dict(zip(s.GAME_ID.astype(int), pd.to_datetime(s.GAME_DATE.astype(str), format="%Y%m%d", errors="coerce")))
    return {}


def appearances(d):
    """(game, player) -> team for every player who shows up in an event or a substitution."""
    rows = []
    for k in (1, 2, 3):
        t, p, tm = f"PERSON{k}TYPE", f"PLAYER{k}_ID", f"PLAYER{k}_TEAM_ID"
        if tm not in d.columns: continue
        x = d[[ "GAME_ID", t, p, tm]].dropna()
        x = x[x[t].astype(int).isin(PTYPES) & (x[p].astype(np.int64) > 0)]
        rows.append(pd.DataFrame(dict(gid=x.GAME_ID.astype(int).values, pid=x[p].astype(np.int64).values, team=x[tm].astype(np.int64).values)))
    a = pd.concat(rows).drop_duplicates(["gid", "pid"])
    return a


def floor_time(st):
    """seconds and possessions on the floor per (game, player) from the rebuilt stints."""
    acc = {}
    good = set()
    for s in st:
        gid, H, V, secs, ph, pv = int(s[1]), s[3], s[4], s[5], s[8], s[9]
        good.add(gid); poss = ph + pv
        for p in H + V:
            a = acc.setdefault((gid, int(p)), [0.0, 0.0]); a[0] += secs; a[1] += poss
    f = pd.DataFrame([(g, p, v[0], v[1]) for (g, p), v in acc.items()], columns=["gid", "pid", "secs", "poss"])
    return f, good


def build(y, avail):
    out = []; names = {}
    for kind, suffix in (("rs", ""), ("po", "po_")):
        key = f"{suffix}{y}"
        d = pbp_for(key, avail)
        if d is None: continue
        dates = dates_for(key, avail, d)
        tag = f"L{key}"
        d.to_csv(os.path.join(DATA, f"nbastats_{tag}.csv"), index=False)
        st, _, stats = build_stints.run(tag)
        os.remove(os.path.join(DATA, f"nbastats_{tag}.csv"))
        log(y, kind, "lineup rebuild:", stats)
        a = appearances(d)
        nm = d[["PLAYER1_ID", "PLAYER1_NAME"]].dropna().drop_duplicates("PLAYER1_ID") if "PLAYER1_NAME" in d.columns else None
        if nm is not None:
            names.update({int(p): str(n) for p, n in zip(nm.PLAYER1_ID, nm.PLAYER1_NAME) if int(p) > 0})
        f, good = floor_time(st)
        g = a.merge(f, on=["gid", "pid"], how="left")
        g.loc[~g.gid.isin(good), ["secs", "poss"]] = np.nan
        g.loc[g.gid.isin(good) & g.secs.isna(), ["secs", "poss"]] = 0.0     # appeared in an event but no stint time (rare)
        # team schedule: every team-game, so absences can be counted
        tg = a.groupby(["gid", "team"]).size().reset_index()[["gid", "team"]]
        tg = tg[tg.team > 0]
        g["date"] = g.gid.map(dates); tg["date"] = tg.gid.map(dates)
        g["kind"] = kind; tg["kind"] = kind
        out.append((g, tg))
        log(y, kind, "games", tg.gid.nunique(), "rebuilt", len(good), "player-games", len(g))
    if not out: return False
    G = pd.concat([o[0] for o in out]); TG = pd.concat([o[1] for o in out])
    G["season"] = y; TG["season"] = y
    G.to_csv(os.path.join(OUT, f"games_{y}.csv.gz"), index=False)
    TG.to_csv(os.path.join(OUT, f"teamgames_{y}.csv.gz"), index=False)
    pd.DataFrame(sorted(names.items()), columns=["pid", "name"]).to_csv(os.path.join(OUT, f"names_{y}.csv"), index=False)
    return True


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    avail = get_list()
    for y in range(a, b + 1):
        if os.path.exists(os.path.join(OUT, f"games_{y}.csv.gz")) and "--force" not in sys.argv:
            log(y, "already built"); continue
        build(y, avail)
