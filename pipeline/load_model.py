"""Load & Durability model: who is likely to miss games next season, and why.

    python pipeline/load_games.py 2010 2025      # first: player-game tables (pipeline/out/games_*.csv.gz)
    python pipeline/load_model.py                # features, pre-set tests (LOAD_PREREG.txt), publish data/load*.json

Everything here is built from play-by-play: who played, for how long, at what pace, on which nights.
"""
import os, sys, json, datetime, urllib.request, io
import numpy as np, pandas as pd
from sklearn.linear_model import PoissonRegressor, LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out"); PUB = os.path.join(REPO, "data")
BIO_URL = "https://raw.githubusercontent.com/gabriel1200/site_Data/master/tplayer_stats.csv"
FIRST, LAST = 2010, 2025                 # seasons of play-by-play (start years)
MIN_MIN = 1000                           # regular-season minutes in season t to be in the sample
RUN = 3                                  # missed games count only in runs of this many or more
BIG_MISS = 20                            # "missed 20+ games"


def log(*a): print(*a, flush=True)


# ------------------------------------------------------------------ inputs
def load_games():
    G, TG, N = [], [], {}
    for y in range(FIRST, LAST + 1):
        p = os.path.join(OUT, f"games_{y}.csv.gz")
        if not os.path.exists(p): continue
        G.append(pd.read_csv(p, parse_dates=["date"])); TG.append(pd.read_csv(os.path.join(OUT, f"teamgames_{y}.csv.gz"), parse_dates=["date"]))
        q = os.path.join(OUT, f"names_{y}.csv")
        if os.path.exists(q): N.update(dict(pd.read_csv(q).values))
    return pd.concat(G, ignore_index=True), pd.concat(TG, ignore_index=True), N


def load_bio():
    cache = os.path.join(OUT, "bio.csv")
    if not os.path.exists(cache):
        blob = urllib.request.urlopen(BIO_URL, timeout=120).read()
        open(cache, "wb").write(blob)
    b = pd.read_csv(cache, low_memory=False, usecols=["player", "Pos", "Age", "MP", "G", "year", "nba_id"])
    b = b.dropna(subset=["nba_id"]); b["nba_id"] = b.nba_id.astype(np.int64)
    b["season"] = b.year - 1                                   # Basketball-Reference year 2024 = season 2023-24
    b["MP"] = pd.to_numeric(b.MP, errors="coerce"); b["Age"] = pd.to_numeric(b.Age, errors="coerce")
    b = b.sort_values("MP").groupby(["nba_id", "season"]).tail(1)   # combined row for traded players has the most minutes
    return b


def pos_group(p):
    p = str(p)
    if p.startswith("C") or p in ("PF-C", "C-PF"): return "big"
    if p.startswith("PF"): return "wing"
    if p.startswith("PG") or p in ("SG-PG",): return "guard"
    if p in ("SG", "SG-SF", "SF-SG"): return "wing"
    if p.startswith("SF"): return "wing"
    return "wing"


# ------------------------------------------------------------------ absences
def absences(G, TG):
    """missed games (runs of RUN+) per player-season, regular season only, scaled to 82 games."""
    g = G[G.kind == "rs"]; tg = TG[TG.kind == "rs"].copy()
    tg = tg.sort_values(["season", "team", "date", "gid"])
    tg["k"] = tg.groupby(["season", "team"]).cumcount()
    nteam = tg.groupby(["season", "team"]).k.max().add(1)
    sched = {key: grp.gid.values for key, grp in tg.groupby(["season", "team"])}
    kidx = dict(zip(zip(tg.season, tg.team, tg.gid), tg.k))
    tdates = {key: grp.date.values for key, grp in tg.groupby(["season", "team"])}
    g = g.assign(k=[kidx.get(x) for x in zip(g.season, g.team, g.gid)]).dropna(subset=["k"])
    g["k"] = g.k.astype(int)
    rows = []
    for (s, p), pg in g.sort_values(["date", "gid"]).groupby(["season", "pid"], sort=False):
        teams = list(dict.fromkeys(pg.team))
        missed = 0; window = 0; runs10 = 0; runs = []
        for i, t in enumerate(teams):
            ks = set(pg.k[pg.team == t]); n = int(nteam[(s, t)])
            lo = 0 if i == 0 else min(ks)
            if i > 0:                                   # games the new team played between his last old-team game and his first new-team game
                last_old = pg.date[pg.team == teams[i - 1]].max()
                kd = tdates[(s, t)]
                gap = int(((kd > np.datetime64(last_old)) & (np.arange(len(kd)) < lo)).sum()) - 2
                if gap >= RUN: missed += gap; window += gap; runs.append(gap); runs10 += gap >= 10
            hi = max(ks) if i < len(teams) - 1 else n - 1
            seq = [0 if k in ks else 1 for k in range(lo, hi + 1)]
            window += len(seq)
            r = 0
            for v in seq + [0]:
                if v: r += 1
                else:
                    if r >= RUN: missed += r; runs.append(r)
                    if r >= 10: runs10 += 1
                    r = 0
        season_len = float(nteam.loc[s].median())
        rows.append((s, p, missed, window, season_len, runs10, len(teams), teams[0], teams[-1]))
    A = pd.DataFrame(rows, columns=["season", "pid", "missed_raw", "window", "season_len", "runs10", "nteams", "first_team", "last_team"])
    A["missed"] = A.missed_raw * 82.0 / A.season_len
    return A


# ------------------------------------------------------------------ workload
def workload(G):
    g = G.copy()
    # games whose lineups could not be rebuilt: keep the game, fill seconds with his season median
    med = g[g.kind == "rs"].groupby(["season", "pid"]).secs.median()
    m = g.secs.isna()
    g.loc[m, "secs"] = [med.get((s, p), np.nan) for s, p in zip(g.season[m], g.pid[m])]
    g.loc[g.poss.isna(), "poss"] = g.secs / 60 * 2.05
    g["mins"] = g.secs / 60
    rs = g[g.kind == "rs"]
    out = rs.groupby(["season", "pid"]).agg(gp=("gid", "size"), min_rs=("mins", "sum"), poss_rs=("poss", "sum"),
                                              g38=("mins", lambda x: (x >= 38).sum()))
    po = g[g.kind == "po"].groupby(["season", "pid"]).mins.sum().rename("min_po")
    out = out.join(po).fillna({"min_po": 0.0})
    out["mpg"] = out.min_rs / out.gp
    out["ppm"] = out.poss_rs / out.min_rs.clip(lower=1)                      # both teams' trips per minute on the floor
    out["g38"] = out.g38 / out.gp
    # back-to-back second nights with 30+ minutes on both nights, per game played; and the biggest 7-day spike
    b2b, spike = {}, {}
    for (s, p), x in rs.sort_values("date").groupby(["season", "pid"], sort=False):
        d = x.date.values.astype("datetime64[D]").astype(np.int64); mn = x.mins.values
        prev = np.r_[-99, d[:-1]]; pm = np.r_[0, mn[:-1]]
        b2b[(s, p)] = float(((d - prev == 1) & (mn >= 30) & (pm >= 30)).sum()) / len(d)
        if len(d) >= 15:
            day = np.zeros(d.max() - d.min() + 1); np.add.at(day, d - d.min(), mn)
            c7 = np.convolve(day, np.ones(7))[:len(day)]; c28 = np.convolve(day, np.ones(28))[:len(day)] / 4
            ok = np.arange(len(day)) >= 27
            r = np.where(ok & (c28 > 40), c7 / np.maximum(c28, 1), np.nan)
            spike[(s, p)] = float(np.nanmax(r)) if np.isfinite(r).any() else np.nan
        else:
            spike[(s, p)] = np.nan
    out["b2b30"] = pd.Series(b2b); out["spike"] = pd.Series(spike)
    return out.reset_index()


def team_load(G):
    """per team-season: mean minutes per game played of the team's five highest-minute players (regular season)."""
    rs = G[(G.kind == "rs")].copy(); rs["mins"] = rs.secs / 60
    t = rs.groupby(["season", "team", "pid"]).agg(m=("mins", "sum"), gp=("gid", "size")).reset_index()
    t["mpg"] = t.m / t.gp
    top = t.sort_values("m").groupby(["season", "team"]).tail(5)
    tl = top.groupby(["season", "team"]).mpg.mean().rename("team_load").reset_index()
    # 36+ minute games by any player, per team game (a second view of the same habit)
    gg = rs[rs.mins >= 36].groupby(["season", "team"]).size()
    ng = rs.groupby(["season", "team"]).gid.nunique()
    tl = tl.merge((gg / ng).rename("g36_per_game").reset_index(), on=["season", "team"], how="left").fillna({"g36_per_game": 0})
    return tl


def league(G, TG, A):
    rs = G[G.kind == "rs"]
    poss = rs.groupby("season").poss.sum() / 5 / 2                        # each possession counted by 5 players x 2 teams
    games = TG[TG.kind == "rs"].groupby("season").gid.nunique()
    pace = poss / games / 2                                              # per team per game
    out = pd.DataFrame(dict(pace=pace.round(2)))
    return out


# ------------------------------------------------------------------ features
def features(G, TG, bio):
    A = absences(G, TG); W = workload(G); TL = team_load(G)
    D = W.merge(A, on=["season", "pid"], how="left")
    # history
    H = A.set_index(["season", "pid"])
    for lag in (1, 2):
        prev = H.missed.rename(f"missed_l{lag}").reset_index(); prev["season"] += lag
        D = D.merge(prev, on=["season", "pid"], how="left")
    # a whole season missing between two seasons he played = a season-long absence (82 games), e.g. a torn ACL year
    firstA = A.groupby("pid").season.min()
    for lag in (1, 2):
        gone = D[f"missed_l{lag}"].isna() & ((D.season - lag) > D.pid.map(firstA))
        D.loc[gone, f"missed_l{lag}"] = 82.0
    for lag in (0, 1, 2):
        r = H.runs10.rename(f"r10_{lag}").reset_index(); r["season"] += lag
        D = D.merge(r, on=["season", "pid"], how="left")
    D["runs10_3y"] = D[["r10_0", "r10_1", "r10_2"]].fillna(0).sum(axis=1)
    # career regular-season minutes through t: Basketball-Reference before 2010-11 + play-by-play from 2010-11
    pre = bio[bio.season < FIRST].groupby("nba_id").MP.sum()
    D = D.sort_values(["pid", "season"])
    D["career_min"] = D.groupby("pid").min_rs.cumsum() + D.pid.map(pre).fillna(0)
    # if a player skipped seasons we still have his cumulative play-by-play minutes; fine
    # age next season: last known Basketball-Reference age + years since
    ages = bio.dropna(subset=["Age"])[["nba_id", "season", "Age"]]
    lastage = ages.sort_values("season").groupby("nba_id").tail(1).set_index("nba_id")
    agemap = dict(zip(zip(ages.nba_id, ages.season), ages.Age))
    def age_next(p, s):
        a = agemap.get((p, s))
        if a is not None: return a + 1, 0
        if p in lastage.index: return lastage.Age[p] + (s - lastage.season[p]) + 1, 0
        return np.nan, 1
    an = [age_next(p, s) for p, s in zip(D.pid, D.season)]
    D["age_next"] = [a for a, _ in an]; D["age_imputed"] = [f for _, f in an]
    first_seen = D.groupby("pid").season.transform("min")
    m = D.age_next.isna()
    D.loc[m, "age_next"] = 21.0 + (D.season[m] - first_seen[m]) + 1          # unknown: assume a 21-year-old rookie
    pos = bio.dropna(subset=["Pos"]).sort_values("season").groupby("nba_id").Pos.last()
    D["pos"] = D.pid.map(pos).map(pos_group).fillna("wing")
    # team load of the team he will play for (his last team of season t; next-season team used in the backtest when known)
    D = D.merge(TL.rename(columns={"team": "last_team"}), on=["season", "last_team"], how="left")
    # outcome: next season
    nxt = A[["season", "pid", "missed", "first_team"]].rename(columns={"missed": "y_missed", "first_team": "next_team"}); nxt["season"] -= 1
    D = D.merge(nxt, on=["season", "pid"], how="left")
    tln = TL.rename(columns={"team": "next_team", "team_load": "next_team_load", "g36_per_game": "next_g36"})
    D = D.merge(tln, on=["season", "next_team"], how="left")
    return D, A, TL


FEATS = dict(
    H=["missed", "missed_l1", "missed_l2", "runs10_3y"],
    A=["age_next", "age_sq"],
    P=["pos_big", "pos_guard"],
    M=["career_min", "min_po"],
    W=["mpg", "ppm", "g38", "b2b30", "spike"],
    T=["tload"],
)


def design(D, blocks, stats=None):
    X = pd.DataFrame(index=D.index)
    X["missed"] = D.missed.fillna(D.missed.mean()); X["missed_l1"] = D.missed_l1; X["missed_l2"] = D.missed_l2
    X["runs10_3y"] = D.runs10_3y
    X["age_next"] = D.age_next; X["age_sq"] = (D.age_next - 27) ** 2
    X["pos_big"] = (D.pos == "big").astype(float); X["pos_guard"] = (D.pos == "guard").astype(float)
    X["career_min"] = D.career_min / 1000; X["min_po"] = D.min_po / 100
    X["mpg"] = D.mpg; X["ppm"] = D.ppm; X["g38"] = D.g38; X["b2b30"] = D.b2b30; X["spike"] = D.spike
    X["tload"] = D.tload if "tload" in D else np.nan
    cols = [c for b in blocks for c in FEATS[b]]
    for c in cols:
        if c not in X: X[c] = D[c]
    X = X[cols].astype(float)
    if stats is None:
        stats = dict(mu=X.mean(), sd=X.std().replace(0, 1))
    X = X.fillna(stats["mu"])
    Z = (X - stats["mu"]) / stats["sd"]
    Z = Z.clip(-4, 4)
    return Z, stats


def fit_predict(train, test, blocks, alpha=1.0):
    Ztr, st = design(train, blocks); Zte, _ = design(test, blocks, st)
    pm = PoissonRegressor(alpha=alpha / len(train) * 50, max_iter=1000).fit(Ztr, train.y_missed)
    yb = (train.y_missed >= BIG_MISS).astype(int)
    lm = LogisticRegression(C=1.0, max_iter=2000).fit(Ztr, yb)
    return pm.predict(Zte), lm.predict_proba(Zte)[:, 1], pm, lm, st


def brier(p, y): return float(np.mean((p - y) ** 2))


def loso(S, blocks):
    mu, pb = np.zeros(len(S)), np.zeros(len(S))
    for s in sorted(S.season.unique()):
        tr, te = S[S.season != s], S[S.season == s]
        m, p, *_ = fit_predict(tr, te, blocks)
        mu[S.season.values == s] = m; pb[S.season.values == s] = p
    return mu, pb


def baseline(S):
    mu, pb = np.zeros(len(S)), np.zeros(len(S))
    for s in sorted(S.season.unique()):
        tr, te = S[S.season != s], S[S.season == s]
        avg = tr.y_missed.mean()
        mu[S.season.values == s] = 0.5 * te.missed.fillna(avg) + 0.5 * avg
        lr = LogisticRegression().fit(tr[["missed"]].fillna(avg), (tr.y_missed >= BIG_MISS).astype(int))
        pb[S.season.values == s] = lr.predict_proba(te[["missed"]].fillna(avg))[:, 1]
    return mu, pb


def ranges(S, mu, q=(0.10, 0.90), nb=10):
    """80% range for games missed: quantiles of actual results among other seasons' players with a similar forecast."""
    lo, hi = np.zeros(len(S)), np.zeros(len(S))
    for s in sorted(S.season.unique()):
        tr = S.season.values != s; te = ~tr
        edges = np.quantile(mu[tr], np.linspace(0, 1, nb + 1)); edges[0] = -1; edges[-1] = 1e9
        bt = np.digitize(mu[tr], edges) - 1; be = np.digitize(mu[te], edges) - 1
        yt = S.y_missed.values[tr]
        ql = np.array([np.quantile(yt[bt == b], q[0]) for b in range(nb)]); qh = np.array([np.quantile(yt[bt == b], q[1]) for b in range(nb)])
        lo[te] = ql[be.clip(0, nb - 1)]; hi[te] = qh[be.clip(0, nb - 1)]
    return lo, hi


def main():
    G, TG, names = load_games(); bio = load_bio()
    log("player-games", len(G), "seasons", sorted(G.season.unique()))
    D, A, TL = features(G, TG, bio)
    D["tload"] = D.next_team_load.fillna(D.team_load)
    D.to_csv(os.path.join(OUT, "load_features.csv.gz"), index=False)
    S = D[(D.min_rs >= MIN_MIN) & (D.season >= FIRST + 2) & (D.season <= LAST - 1)]
    dropped = int(S.y_missed.isna().sum())
    S = S.dropna(subset=["y_missed"]).reset_index(drop=True)
    log("sample", len(S), "player-seasons; left out (did not play next season):", dropped)
    yb = (S.y_missed >= BIG_MISS).astype(int).values
    R = dict(n=len(S), left_out=dropped, seasons=[int(S.season.min()), int(S.season.max())], share_20plus=float(yb.mean()), avg_missed=float(S.y_missed.mean()))

    b_mu, b_pb = baseline(S)
    full = ["H", "A", "P", "M", "W", "T"]
    f_mu, f_pb = loso(S, full)
    noW = ["H", "A", "P", "T"]; base = ["H", "A", "P"]
    n_mu, n_pb = loso(S, base)
    w_mu, w_pb = loso(S, base + ["M", "W"])
    t0_mu, t0_pb = loso(S, ["H", "A", "P", "M", "W"])
    mae = lambda m: float(np.mean(np.abs(m - S.y_missed.values)))
    R["L1"] = dict(mae_base=mae(b_mu), mae_full=mae(f_mu), brier_base=brier(b_pb, yb), brier_full=brier(f_pb, yb))
    R["L1"]["pass_a"] = R["L1"]["mae_base"] - R["L1"]["mae_full"] >= 0.50
    R["L1"]["pass_b"] = (R["L1"]["brier_base"] - R["L1"]["brier_full"]) / R["L1"]["brier_base"] >= 0.02
    R["L2"] = dict(brier_without=brier(n_pb, yb), brier_with=brier(w_pb, yb), mae_without=mae(n_mu), mae_with=mae(w_mu))
    R["L2"]["pass"] = ((R["L2"]["brier_without"] - R["L2"]["brier_with"]) / R["L2"]["brier_without"] >= 0.01) and (R["L2"]["mae_without"] - R["L2"]["mae_with"] >= 0.20)
    R["L3"] = dict(brier_without=brier(t0_pb, yb), brier_with=brier(f_pb, yb), mae_without=mae(t0_mu), mae_with=mae(f_mu))
    R["L3"]["pass"] = (R["L3"]["brier_without"] - R["L3"]["brier_with"]) / R["L3"]["brier_without"] >= 0.005
    for k in ("L1", "L2", "L3"): log(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in R[k].items()})
    # each workload input alone on top of the base, for the page (informational)
    R["single"] = {}
    for c in ["career_min", "min_po", "mpg", "ppm", "g38", "b2b30", "spike", "tload"]:
        FEATS["X"] = [c]
        m1, p1 = loso(S, base + ["X"])
        R["single"][c] = dict(brier=brier(p1, yb), mae=mae(m1))
    R["single_base"] = dict(brier=brier(n_pb, yb), mae=mae(n_mu))
    log("single inputs", {k: (round(v["brier"], 5), round(v["mae"], 3)) for k, v in R["single"].items()}, "base", R["single_base"])
    json.dump(R, open(os.path.join(OUT, "load_tests_partial.json"), "w"), indent=1, default=float)
    return D, S, R


if __name__ == "__main__":
    main()
