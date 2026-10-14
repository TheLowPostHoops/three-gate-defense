"""Player projections, team strengths and the backtest (F1-F4 in FORECAST_PREREG.txt).

    python3 fc_project.py backtest     # tests on 2014-15 .. 2025-26, writes out/fc_backtest.json
"""
import os, sys, json, math
import numpy as np, pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); INP = os.path.join(HERE, "inputs")
FIRST_T, LAST_T = 2014, 2025          # seasons predicted in the backtest (start year)
_c = {}


def draft():
    if "draft" not in _c:
        d = {}
        for line in open(os.path.join(INP, "draft_picks.txt")):
            y, rest = line.strip().split(":")
            for x in rest.split(","):
                n, p = x.split("."); d[int(p)] = (int(y), int(n))
        _c["draft"] = d
    return _c["draft"]


def bdays():
    if "bd" not in _c:
        b = pd.read_csv(os.path.join(OUT, "bio_v2.csv"))
        d = {int(p): pd.Timestamp(x) for p, x in zip(b.pid, b.bday) if isinstance(x, str)}
        for k, v in json.load(open(os.path.join(INP, "bio_players.json"))).items():
            if v[2] and int(k) not in d: d[int(k)] = pd.Timestamp(v[2][:10])
        _c["bd"] = d
    return _c["bd"]


def age_at(pid, season):
    b = bdays().get(int(pid))
    return np.nan if b is None else (pd.Timestamp(f"{season + 1}-02-01") - b).days / 365.25


def games(y):
    """player-season table: gp, minutes, mpg, opening team, main team."""
    k = ("g", y)
    if k not in _c:
        G = pd.read_csv(os.path.join(OUT, f"games_{y}.csv.gz"), parse_dates=["date"]); G = G[G.kind == "rs"]
        G = G.sort_values("date")
        a = G.groupby("pid").agg(gp=("gid", "nunique"), secs=("secs", "sum"), team0=("team", "first"))
        a["mpg"] = a.secs / 60 / a.gp
        main = G.groupby(["pid", "team"]).secs.sum().reset_index().sort_values("secs").groupby("pid").tail(1).set_index("pid").team
        a["team"] = main
        _c[k] = a
    return _c[k]


def results(y):
    """game results from the stints: gid, home, away, home points, away points, date."""
    k = ("r", y)
    if k not in _c:
        import pickle
        s = pickle.load(open(os.path.join(OUT, f"stints_{y}.pkl"), "rb"))
        rows = {}
        for x in s:
            if not str(x[1]).startswith("2"): continue
            r = rows.setdefault(x[1], [x[10], x[11], 0, 0]); r[2] += x[6]; r[3] += x[7]
        R = pd.DataFrame([(g, *v) for g, v in rows.items()], columns=["gid", "home", "away", "hp", "ap"])
        TG = pd.read_csv(os.path.join(OUT, f"teamgames_{y}.csv.gz"), parse_dates=["date"]).drop_duplicates("gid").set_index("gid").date
        R["date"] = R.gid.map(TG); R = R.sort_values(["date", "gid"]).reset_index(drop=True)
        _c[k] = R
    return _c[k]


def team_mov(y):
    R = results(y)
    h = R.assign(t=R.home, m=R.hp - R.ap)[["t", "m"]]; a = R.assign(t=R.away, m=R.ap - R.hp)[["t", "m"]]
    return pd.concat([h, a]).groupby("t").m.mean()


def ratings(y):
    return pd.read_csv(os.path.join(OUT, f"fc_ratings_{y}.csv")).set_index("pid")


def single(y):
    return pd.read_csv(os.path.join(OUT, f"fc_single_{y}.csv")).set_index("pid")


# ------------------------------------------------------------------ player projection model
def pred_rows(t):
    """features for every player who has a rating window ending t-1, to predict season t."""
    R = ratings(t - 1)
    X = pd.DataFrame(index=R.index)
    X["o"] = R.o; X["d"] = R.d; X["lposs"] = np.log(R.poss.clip(lower=50))
    X["age"] = [age_at(p, t) for p in R.index]
    X["age"] = X.age.fillna(27.0)
    X["a1"] = X.age - 27; X["a2"] = (X.age - 27) ** 2
    return X


def fit_player_model(train_ts):
    """o_t, d_t (single-season ratings) on last window's rating, age and sample size. possession weighted."""
    fr = []
    for t in train_ts:
        X = pred_rows(t); S = single(t)
        j = X.join(S[["o", "d", "poss"]].rename(columns={"o": "yo", "d": "yd", "poss": "w"}), how="inner")
        fr.append(j[j.w > 300])
    D = pd.concat(fr)
    out = {}
    for side in ("o", "d"):
        F = np.column_stack([np.ones(len(D)), D[side], D[side] * D.lposs, D.a1, D.a2, D.lposs])
        w = D.w.values; A = (F * w[:, None]).T @ F
        out[side] = np.linalg.solve(A, (F * w[:, None]).T @ D["y" + side].values)
    # rookies: first-season rating by draft slot
    rk = []
    dr = draft()
    for t in train_ts:
        S = single(t); prev = set(ratings(t - 1).index)
        for p in S.index:
            if p in prev or S.loc[p, "poss"] < 300: continue
            y, n = dr.get(int(p), (None, None))
            if y is not None and y < t - 1: continue                     # drafted earlier, first NBA minutes now: treat as undrafted-like
            rk.append(dict(lp=np.log(n) if n else np.log(75), und=0 if n else 1, yo=S.loc[p, "o"], yd=S.loc[p, "d"], w=S.loc[p, "poss"]))
    K = pd.DataFrame(rk)
    for side in ("o", "d"):
        F = np.column_stack([np.ones(len(K)), K.lp, K.und]); w = K.w.values
        out["rk_" + side] = np.linalg.solve((F * w[:, None]).T @ F + 1e-6 * np.eye(3), (F * w[:, None]).T @ K["y" + side].values)
    return out


def project(t, M, pids=None):
    """projected o, d, net for season t for the given players (all with a window, plus rookies by draft slot)."""
    X = pred_rows(t)
    P = pd.DataFrame(index=X.index)
    for side in ("o", "d"):
        c = M[side]; P[side] = c[0] + c[1] * X[side] + c[2] * X[side] * X.lposs + c[3] * X.a1 + c[4] * X.a2 + c[5] * X.lposs
    if pids is not None:
        dr = draft()
        new = [p for p in pids if p not in P.index]
        rows = []
        for p in new:
            y, n = dr.get(int(p), (None, None))
            lp, und = (np.log(n), 0) if n else (np.log(75), 1)
            rows.append(dict(pid=p, o=M["rk_o"][0] + M["rk_o"][1] * lp + M["rk_o"][2] * und, d=M["rk_d"][0] + M["rk_d"][1] * lp + M["rk_d"][2] * und, rookie=True))
        if rows: P = pd.concat([P.assign(rookie=False), pd.DataFrame(rows).set_index("pid")])
    P["net"] = P.o + P.d
    return P


# ------------------------------------------------------------------ minutes and team strength
def expected_mpg(t, pids, P):
    """minutes per game before the season: last season's mpg (10+ games), else two seasons ago, else by draft slot / 12."""
    g1 = games(t - 1); g2 = games(t - 2) if t - 2 >= 2010 else None
    out = {}
    dr = draft()
    for p in pids:
        if p in g1.index and g1.loc[p, "gp"] >= 10: out[p] = g1.loc[p, "mpg"]
        elif g2 is not None and p in g2.index and g2.loc[p, "gp"] >= 10: out[p] = g2.loc[p, "mpg"] * 0.9
        else:
            y, n = dr.get(int(p), (None, None))
            out[p] = 22 - 8 * math.log10(n) if n and y and y >= t - 1 else 10.0
    return pd.Series(out)


def team_strength(P, roster, mpg, avail=None):
    """roster: Series pid -> team. Minutes per game scaled to 240 per team. avail: share of games each player plays."""
    D = pd.DataFrame(dict(team=roster, mpg=mpg)).join(P[["o", "d", "net"]], how="left")
    D["o"] = D.o.fillna(-1.0); D["d"] = D.d.fillna(-1.0); D["net"] = D.o + D.d
    if avail is not None: D["mpg"] = D.mpg * avail.reindex(D.index).fillna(1.0)
    D["m"] = D.mpg.clip(upper=38) * 240 / D.groupby("team").mpg.transform(lambda s: s.clip(upper=38).sum())
    D["m"] = D.m.clip(upper=40)
    D["m"] = D.m * 240 / D.groupby("team").m.transform("sum")
    T = D.assign(wo=D.o * D.m / 48, wd=D.d * D.m / 48).groupby("team")[["wo", "wd"]].sum()
    T["net"] = T.wo + T.wd
    return T, D


def season_inputs(t, M):
    g = games(t)
    roster = g.team0                      # opening-night team
    P = project(t, M, list(roster.index))
    mpg = expected_mpg(t, list(roster.index), P)
    T, D = team_strength(P, roster, mpg)
    T["net_c"] = T.net - T.net.mean()
    return T, P, D


# ------------------------------------------------------------------ backtest
def backtest():
    res = dict(F1=[], F2=[], seasons=[])
    allg = []
    for t in range(FIRST_T, LAST_T + 1):
        M = fit_player_model(range(2012 + 1, t))        # windows ending t-1 or earlier: targets 2013..t-1
        T, P, D = season_inputs(t, M)
        mov = team_mov(t); prev = team_mov(t - 1)
        df = pd.DataFrame(dict(model=T.net_c, actual=mov, base=0.6 * prev.reindex(T.index))).dropna()
        df["season"] = t
        res["seasons"].append(df.reset_index().rename(columns={"index": "team"}))
        R = results(t).copy()
        R["s_model"] = R.home.map(T.net_c) - R.away.map(T.net_c); R["s_base"] = R.home.map(0.6 * prev) - R.away.map(0.6 * prev)
        R["season"] = t; allg.append(R)
        print(t, "teams", len(df), "corr", round(df.model.corr(df.actual), 3), "base corr", round(df.base.corr(df.actual), 3), flush=True)
    S = pd.concat(res["seasons"]); G = pd.concat(allg)
    return S, G


def fit_scale(S, G, train):
    """scale k (team strength -> MOV), home court and game spread, fit on training seasons only."""
    s = S[S.season.isin(train)]
    k = float((s.model * s.actual).sum() / (s.model ** 2).sum())
    kb = float((s.base * s.actual).sum() / (s.base ** 2).sum())
    g = G[G.season.isin(train)]
    m = g.hp - g.ap
    hca = float((m - k * g.s_model).mean()); sd = float((m - k * g.s_model - hca).std())
    hcb = float((m - kb * g.s_base).mean()); sdb = float((m - kb * g.s_base - hcb).std())
    return dict(k=k, kb=kb, hca=hca, sd=sd, hcb=hcb, sdb=sdb)


def run_backtest():
    S, G = backtest()
    S.to_csv(os.path.join(OUT, "fc_bt_teams.csv"), index=False); G.to_csv(os.path.join(OUT, "fc_bt_games.csv"), index=False)
    out = dict(F1=[], F2=[], F3=None)
    preds = []
    for t in range(FIRST_T, LAST_T + 1):
        c = fit_scale(S, G, [x for x in range(FIRST_T, t)] or [t])   # first season: fit on itself only for scale (flagged)
        s = S[S.season == t]; g = G[G.season == t].copy()
        mae_m = float((c["k"] * s.model - s.actual).abs().mean()); mae_b = float((c["kb"] * s.base - s.actual).abs().mean())
        out["F1"].append(dict(season=t, mae_model=mae_m, mae_base=mae_b))
        y = (g.hp > g.ap).astype(int)
        pm = norm.cdf((c["k"] * g.s_model + c["hca"]) / c["sd"]); pb = norm.cdf((c["kb"] * g.s_base + c["hcb"]) / c["sdb"])
        pm = np.clip(pm, 1e-4, 1 - 1e-4); pb = np.clip(pb, 1e-4, 1 - 1e-4)
        ll = lambda p: float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        out["F2"].append(dict(season=t, acc_model=float(((pm > .5) == y).mean()), acc_base=float(((pb > .5) == y).mean()), ll_model=ll(pm), ll_base=ll(pb)))
        preds.append(pd.DataFrame(dict(p=pm, y=y, season=t)))
    Pp = pd.concat(preds); Pp = Pp[Pp.season > FIRST_T]
    Pp["bin"] = (Pp.p * 10).astype(int).clip(0, 9)
    cal = Pp.groupby("bin").agg(n=("y", "size"), pred=("p", "mean"), act=("y", "mean")).reset_index()
    out["F3"] = cal.to_dict("records")
    f1 = pd.DataFrame(out["F1"]); f1 = f1[f1.season > FIRST_T]
    f2 = pd.DataFrame(out["F2"]); f2 = f2[f2.season > FIRST_T]
    out["summary"] = dict(
        F1=dict(mae_model=f1.mae_model.mean(), mae_base=f1.mae_base.mean(), gain=1 - f1.mae_model.mean() / f1.mae_base.mean(), pass_=bool(1 - f1.mae_model.mean() / f1.mae_base.mean() >= 0.05)),
        F2=dict(acc_model=f2.acc_model.mean(), acc_base=f2.acc_base.mean(), ll_model=f2.ll_model.mean(), ll_base=f2.ll_base.mean(),
                pass_=bool(f2.acc_model.mean() > f2.acc_base.mean() and f2.ll_model.mean() < f2.ll_base.mean())),
        F3=dict(max_gap=float(max(abs(r["pred"] - r["act"]) for r in out["F3"] if r["n"] >= 300)), pass_=bool(all(abs(r["pred"] - r["act"]) <= 0.05 for r in out["F3"] if r["n"] >= 300))),
        note="scored on 2015-16 to 2025-26; 2014-15 is used only to fit the first season's scale")
    json.dump(out, open(os.path.join(OUT, "fc_backtest.json"), "w"), indent=1, default=float)
    print(json.dumps(out["summary"], indent=1, default=float))



# ------------------------------------------------------------------ F4: blending preseason strength with results
def blend_eval(G, S, K, seasons, c):
    """log loss on games after each team's 20th game, strengths blended with K."""
    lls = {"blend": [], "pre": [], "res": []}
    for t in seasons:
        g = G[G.season == t].sort_values(["date", "gid"]).reset_index(drop=True)
        pre = (S[S.season == t].set_index("team").model * c["k"]).to_dict()
        n = {}; sm = {}                                          # games, sum of opponent-and-home-adjusted margins
        for r in g.itertuples():
            h, a = r.home, r.away
            ph, pa = pre.get(h, 0), pre.get(a, 0)
            def cur(x, p):
                N = n.get(x, 0); obs = sm.get(x, 0) / N if N else p
                return p * K / (K + N) + obs * N / (K + N), obs
            bh, oh = cur(h, ph); ba, oa = cur(a, pa)
            if n.get(h, 0) >= 20 and n.get(a, 0) >= 20:
                y = 1 if r.hp > r.ap else 0
                for key, d in (("blend", bh - ba), ("pre", ph - pa), ("res", oh - oa)):
                    p = min(max(norm.cdf((d + c["hca"]) / c["sd"]), 1e-4), 1 - 1e-4)
                    lls[key].append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
            m = r.hp - r.ap - c["hca"]
            sm[h] = sm.get(h, 0) + m + pa; sm[a] = sm.get(a, 0) - m + ph   # margin plus opponent's preseason strength
            n[h] = n.get(h, 0) + 1; n[a] = n.get(a, 0) + 1
    return {k: float(np.mean(v)) for k, v in lls.items()}


def run_f4():
    S = pd.read_csv(os.path.join(OUT, "fc_bt_teams.csv")); G = pd.read_csv(os.path.join(OUT, "fc_bt_games.csv"), parse_dates=["date"])
    c = fit_scale(S, G, list(range(FIRST_T, 2020)))
    train, test = list(range(FIRST_T, 2020)), list(range(2020, LAST_T + 1))
    grid = {K: blend_eval(G, S, K, train, c)["blend"] for K in (5, 10, 15, 20, 25, 30, 40, 50, 70)}
    K = min(grid, key=grid.get)
    ev = blend_eval(G, S, K, test, c)
    out = dict(K=K, grid=grid, test=ev, pass_=bool(ev["blend"] < ev["pre"] and ev["blend"] < ev["res"]))
    bt = json.load(open(os.path.join(OUT, "fc_backtest.json"))); bt["F4"] = out
    c_all = fit_scale(S, G, list(range(FIRST_T, LAST_T + 1)))
    s = S.copy(); res = (c_all["k"] * s.model - s.actual)
    bt["scale"] = dict(c_all, team_resid_sd=float(res.std()))
    json.dump(bt, open(os.path.join(OUT, "fc_backtest.json"), "w"), indent=1, default=float)
    print(json.dumps(dict(F4=out, scale=bt["scale"]), indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1:] == ["backtest"]: run_backtest(); run_f4()
    if sys.argv[1:] == ["f4"]: run_f4()
