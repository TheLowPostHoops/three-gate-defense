"""College (C1) and miles-run (L6) tests for the Load & Durability model.

    python pipeline/load_extra.py

Inputs (pulled through a browser from sites that block scripts, committed under pipeline/inputs/):
  college_seasons.csv  Bart Torvik player seasons 2008-2026 for names that match an NBA player
  miles_seasons.csv    NBA.com speed/distance season totals 2013-14 on, regular season and playoffs
  miles_games.csv      (optional) NBA.com speed/distance by game day
"""
import os, re, json, unicodedata
import numpy as np, pandas as pd
import load_model as LM

OUT = LM.OUT


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s).strip(); s = re.sub(r" (jr|sr|ii|iii|iv)$", "", s)
    return s.replace(" ", "")


def nba_birthdates():
    out = {}
    for pid, v in json.load(open(os.path.join(LM.HERE, "inputs", "bio_players.json"))).items():
        t = pd.to_datetime(str(v[2])[:10], errors="coerce")
        if pd.notna(t): out[int(pid)] = t
    for r in json.load(open(os.path.join(LM.HERE, "inputs", "rosters_2026.json"))):
        t = pd.to_datetime(r[7], format="%b %d, %Y", errors="coerce")
        if pd.notna(t): out[int(r[1])] = t
    return out


NBA_BD = nba_birthdates()


def inches(h):
    m = re.match(r"(\d+)-(\d+)", str(h))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def college(names, first_season):
    C = pd.read_csv(os.path.join(LM.HERE, "inputs", "college_seasons.csv"))
    C["bday"] = C.bday.fillna("?")
    rows = {}
    for pid, nm in names.items():
        f = first_season.get(pid)
        if f is None: continue
        c = C[C.name == norm(nm)]
        if c.empty: continue
        best = None
        for bd, x in c.groupby("bday"):
            last = x.year.max()                        # Torvik year = season end year
            if 0 <= f - last <= 2:
                if best is None or len(x) > len(best[1]): best = (bd, x)
        if best is None: continue
        bd, x = best
        nb = NBA_BD.get(pid)
        if nb is not None and bd != "?" and pd.to_datetime(bd, errors="coerce") != nb:
            continue                                    # same name, different person
        x = x.sort_values("year")
        rows[pid] = dict(col_years=x.year.nunique(), col_gp=x.gp.sum(), col_tg=x.groupby("year").team_games.max().sum(),
                         col_min=float((x.gp * x.mpg).sum()), col_mpg=float((x.gp * x.mpg).sum() / max(x.gp.sum(), 1)),
                         height=inches(x.ht.iloc[-1]), bday=None if bd == "?" else bd, col_last=int(x.year.max()))
    R = pd.DataFrame.from_dict(rows, orient="index"); R.index.name = "pid"
    R["col_missed"] = 1 - R.col_gp / R.col_tg.clip(lower=1)
    return R.reset_index()


def main():
    G, TG, names = LM.load_games()
    bio = LM.load_bio()
    D = pd.read_csv(os.path.join(OUT, "load_features.csv.gz"))
    D["tload"] = D.next_team_load.fillna(D.team_load)
    # first NBA season: Basketball-Reference if older than the play-by-play, else first play-by-play season
    fb = bio.groupby("nba_id").season.min()
    fp = G.groupby("pid").season.min()
    first = {p: int(min(fb.get(p, 9999), fp[p])) for p in fp.index}
    CO = college(names, first)
    log = {}
    log["college_matched"] = len(CO)
    D = D.merge(CO, on="pid", how="left")
    D["first_nba"] = D.pid.map(first)
    D["no_college"] = (D.col_years.isna() & (D.first_nba >= 2009)).astype(float)
    for c in ("col_min", "col_missed", "col_years", "height"):
        D[c + "_f"] = D[c]
    # better ages from birth dates where we have them (age on Feb 1 of season t+1)
    # ages: NBA.com birthdates by player id (team rosters + player pages); Basketball-Reference carry-forward only
    # when NBA.com has none. College birthdates are not used for ages (matched by name, so they can be the wrong person).
    bd = D.pid.map(nba_birthdates())
    feb1 = pd.to_datetime((D.season + 2).astype(str) + "-02-01")
    exact = (feb1 - bd).dt.days / 365.25
    log["ages_from_nba_birthdate"] = int(exact.notna().sum()); log["ages_without_birthdate"] = int(exact.isna().sum())
    log["age_diff_mean"] = float((exact - D.age_next).abs().mean())
    D.loc[exact.notna(), "age_next"] = np.floor(exact[exact.notna()])

    S = D[(D.min_rs >= LM.MIN_MIN) & (D.season >= LM.FIRST + 2) & (D.season <= LM.LAST - 1)].dropna(subset=["y_missed"]).reset_index(drop=True)
    full = ["H", "A", "P", "M", "W", "T"]

    # C1: players who entered the NBA in 2008-09 or later (college files start 2008)
    LM.FEATS["C"] = ["col_min_f", "col_missed_f", "col_years_f", "height_f", "no_college"]
    SC = S[S.first_nba >= 2009].reset_index(drop=True)
    yb = (SC.y_missed >= LM.BIG_MISS).astype(int).values
    _, p0 = LM.loso(SC, full); _, p1 = LM.loso(SC, full + ["C"])
    b0, b1 = LM.brier(p0, yb), LM.brier(p1, yb)
    C1 = dict(n=len(SC), with_college_record=int(SC.col_years.notna().sum()), no_college=int(SC.no_college.sum()),
              brier_without=b0, brier_with=b1, gain=(b0 - b1) / b0, pass_=(b0 - b1) / b0 >= 0.01)
    print("C1", C1)

    # L6: miles (season level; game-level spike only if the game file exists)
    M = pd.read_csv(os.path.join(LM.HERE, "inputs", "miles_seasons.csv"))
    log["miles_rows_dropped_bad_speed"] = int((M.speed > 5.5).sum())
    M = M[M.speed <= 5.5]                 # NBA.com rows with impossible average speeds (44 of them in 2025-26)
    rs = M[M.kind == "rs"].groupby(["season", "pid"]).agg(mi=("miles", "sum"), mn=("min", "sum"), gp=("gp", "sum"),
                                                           spd=("speed", "mean")).reset_index()
    rs["mi36"] = rs.mi / rs.mn.clip(lower=1) * 36; rs["mipg"] = rs.mi / rs.gp.clip(lower=1)
    po = M[M.kind == "po"].groupby(["season", "pid"]).miles.sum().rename("mi_po").reset_index()
    S6 = S.merge(rs[["season", "pid", "mi36", "mipg", "spd"]], on=["season", "pid"], how="inner").merge(po, on=["season", "pid"], how="left")
    S6["mi_po"] = S6.mi_po.fillna(0)
    feats6 = ["mi36", "mipg", "spd", "mi_po"]
    gpath = os.path.join(LM.HERE, "inputs", "miles_spike.csv")
    if os.path.exists(gpath):
        S6 = S6.merge(pd.read_csv(gpath), on=["season", "pid"], how="left"); feats6.append("mi_spike")
    LM.FEATS["MI"] = feats6
    yb6 = (S6.y_missed >= LM.BIG_MISS).astype(int).values
    _, q0 = LM.loso(S6, full); _, q1 = LM.loso(S6, full + ["MI"])
    c0, c1 = LM.brier(q0, yb6), LM.brier(q1, yb6)
    L6 = dict(n=len(S6), inputs=feats6, brier_without=c0, brier_with=c1, gain=(c0 - c1) / c0, pass_=(c0 - c1) / c0 >= 0.01)
    print("L6", L6)
    # descriptive: league miles per 36 by season
    lg = rs[rs.mn >= 500].groupby("season").apply(lambda x: x.mi.sum() / x.mn.sum() * 36, include_groups=False).round(3).to_dict()
    json.dump(dict(C1=C1, L6=L6, log=log, miles36_by_season=lg), open(os.path.join(OUT, "load_extra_tests.json"), "w"), indent=1, default=float)
    D.to_csv(os.path.join(OUT, "load_features_plus.csv.gz"), index=False)
    print(log, lg)


if __name__ == "__main__":
    main()
