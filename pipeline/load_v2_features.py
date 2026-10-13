"""Version 2 inputs for the Load & Durability model: contact exposure, travel and body size.

    python pipeline/load_v2_features.py

Writes pipeline/out/contact.csv (player-season), pipeline/out/travel.csv (team-game) from the cached raw files
(pipeline/data/raw, filled by load_games.py), and pipeline/out/bio_v2.csv from pipeline/inputs/ (browser pulls).
"""
import os, json, math
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "data", "raw"); OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "data"); INP = os.path.join(HERE, "inputs")

# arena location (lat, lon) and standard UTC offset; Pistons moved downtown in 2017-18
ARENA = {"ATL": (33.757, -84.396, -5), "BOS": (42.366, -71.062, -5), "CLE": (41.496, -81.688, -5), "NOP": (29.949, -90.082, -6),
         "NOH": (29.949, -90.082, -6), "CHI": (41.881, -87.674, -6), "DAL": (32.790, -96.810, -6), "DEN": (39.749, -105.008, -7),
         "GSW": (37.768, -122.388, -8), "HOU": (29.751, -95.362, -6), "LAC": (34.043, -118.267, -8), "LAL": (34.043, -118.267, -8),
         "MIA": (25.781, -80.188, -5), "MIL": (43.045, -87.917, -6), "MIN": (44.979, -93.276, -6), "BKN": (40.683, -73.975, -5),
         "NJN": (40.734, -74.171, -5), "NYK": (40.751, -73.993, -5), "ORL": (28.539, -81.384, -5), "IND": (39.764, -86.155, -5),
         "PHI": (39.901, -75.172, -5), "PHX": (33.446, -112.071, -7), "POR": (45.532, -122.667, -8), "SAC": (38.580, -121.500, -8),
         "SAS": (29.427, -98.437, -6), "OKC": (35.463, -97.515, -6), "TOR": (43.643, -79.379, -5), "UTA": (40.768, -111.901, -7),
         "MEM": (35.138, -90.051, -6), "WAS": (38.898, -77.021, -5), "DET": (42.341, -83.055, -5), "CHA": (35.225, -80.839, -5)}
ALTITUDE = {"DEN", "UTA"}
BUBBLE = (pd.Timestamp("2020-07-01"), pd.Timestamp("2020-10-31"), (28.337, -81.556, -5))       # 2019-20 restart, Orlando
TAMPA = (27.943, -82.452, -5)                                                                    # Raptors' home in 2020-21


def miles(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


_AVAIL = None


def avail():
    global _AVAIL
    if _AVAIL is None:
        from update import get_list
        _AVAIL = get_list()
    return _AVAIL


def raw_shots(key):
    """shot detail csv for a dataset key, downloaded if this run has not fetched it yet (a cached season skips the download)."""
    p = os.path.join(RAW, f"shotdetail_{key}.csv")
    if os.path.exists(p): return p
    if f"shotdetail_{key}" not in avail(): return None
    from update import fetch
    return fetch(f"shotdetail_{key}")


def raw_pbp(y, po=False):
    k = f"nbastats_{'po_' if po else ''}{y}"
    p = os.path.join(RAW, k + ".csv")
    if os.path.exists(p): return pd.read_csv(p, low_memory=False)
    p = os.path.join(DATA, f"conv_{'po_' if po else ''}{y}.csv")
    if os.path.exists(p): return pd.read_csv(p, low_memory=False)
    from load_games import pbp_for
    return pbp_for(f"{'po_' if po else ''}{y}", avail())


def contact(seasons):
    """per player-season, regular season: shots at the rim, free throws, fouls drawn, charges taken (per 36 later)."""
    rows = []
    for y in seasons:
        sd = pd.read_csv(raw_shots(str(y)), usecols=["PLAYER_ID", "SHOT_ZONE_BASIC"], low_memory=False)
        rim = sd[sd.SHOT_ZONE_BASIC == "Restricted Area"].groupby("PLAYER_ID").size().rename("rim_fga")
        d = raw_pbp(y)
        ft = d[(d.EVENTMSGTYPE == 3)].groupby("PLAYER1_ID").size().rename("fta")
        fouls = d[d.EVENTMSGTYPE == 6]
        drawn = fouls[fouls.PERSON2TYPE.isin([4, 5])].groupby("PLAYER2_ID").size().rename("fouls_drawn")
        chg = fouls[fouls.EVENTMSGACTIONTYPE.isin([4, 26])]                      # offensive / charge fouls: player2 took the charge
        took = chg[chg.PERSON2TYPE.isin([4, 5])].groupby("PLAYER2_ID").size().rename("charges_taken")
        c = pd.concat([rim, ft, drawn, took], axis=1).fillna(0)
        c.index.name = "pid"; c = c.reset_index(); c["season"] = y
        rows.append(c[c.pid > 0])
        print("contact", y, len(c), flush=True)
    return pd.concat(rows, ignore_index=True)


def travel(seasons):
    """per team-game: miles flown since the last game, time zones crossed, days of rest, altitude, home/away."""
    out = []
    for y in seasons:
        for po in (False, True):
            p = raw_shots(f"{'po_' if po else ''}{y}")
            if p is None: continue
            s = pd.read_csv(p, usecols=["GAME_ID", "GAME_DATE", "HTM", "VTM"], low_memory=False).drop_duplicates("GAME_ID")
            s["date"] = pd.to_datetime(s.GAME_DATE.astype(str), format="%Y%m%d")
            for side, other in (("HTM", "VTM"), ("VTM", "HTM")):
                t = pd.DataFrame(dict(GAME_ID=s.GAME_ID.values, date=s.date.values, team=s[side].values, HTM=s.HTM.values))
                t["home"] = int(side == "HTM"); t["season"] = y; t["po"] = int(po)
                out.append(t)
    T = pd.concat(out, ignore_index=True).sort_values(["season", "team", "date"])
    def loc(r):
        if BUBBLE[0] <= r.date <= BUBBLE[1]: return BUBBLE[2]
        if r.HTM == "TOR" and r.season == 2020: return TAMPA
        return ARENA.get(r.HTM, ARENA["NYK"])
    T["loc"] = [loc(r) for r in T.itertuples()]
    res = []
    for (s, t), g in T.groupby(["season", "team"], sort=False):
        prev, prevd = None, None
        homeloc = ARENA.get(t, ARENA["NYK"])
        for r in g.itertuples():
            here = r.loc
            if prev is None:
                mi, tz, rest = miles(homeloc, here), abs(homeloc[2] - here[2]), 7
            else:
                mi, tz, rest = miles(prev, here), abs(prev[2] - here[2]), (r.date - prevd).days - 1
            res.append((r.GAME_ID, t, s, r.po, r.date, r.home, round(mi), tz, rest, int(r.HTM in ALTITUDE)))
            prev, prevd = here, r.date
    return pd.DataFrame(res, columns=["gid", "abbr", "season", "po", "date", "home", "fly_miles", "tz", "rest_days", "altitude"])


def bio():
    rows = {}
    for r in json.load(open(os.path.join(INP, "rosters_2026.json"))):
        team, pid, name, _, pos, ht, wt, bd = r[:8]
        rows[int(pid)] = dict(ht=ht, wt=wt, bday=pd.to_datetime(bd, format="%b %d, %Y", errors="coerce"), team26=int(team), how26=r[10], exp26=r[8])
    p = os.path.join(INP, "bio_players.json")
    if os.path.exists(p):
        for pid, v in json.load(open(p)).items():
            if int(pid) in rows: continue
            rows[int(pid)] = dict(ht=v[0], wt=v[1], bday=pd.to_datetime(v[2][:10], errors="coerce"), team26=None, how26=None, exp26=None)
    B = pd.DataFrame.from_dict(rows, orient="index"); B.index.name = "pid"; B = B.reset_index()
    def inch(h):
        try: a, b = str(h).split("-"); return int(a) * 12 + int(b)
        except Exception: return np.nan
    B["height_in"] = B.ht.map(inch); B["weight_lb"] = pd.to_numeric(B.wt, errors="coerce")
    B["bmi"] = 703 * B.weight_lb / B.height_in ** 2
    return B


if __name__ == "__main__":
    seasons = list(range(2010, 2026))
    travel(seasons).to_csv(os.path.join(OUT, "travel.csv"), index=False)
    print("travel done", flush=True)
    contact(seasons).to_csv(os.path.join(OUT, "contact.csv"), index=False)
    bio().to_csv(os.path.join(OUT, "bio_v2.csv"), index=False)
    print("done")
