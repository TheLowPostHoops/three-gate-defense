"""Award voter models (A1 in FORECAST_PREREG.txt) and award odds inside the season simulation.

    python3 fc_awards.py test      # fit on 2010-11..2019-20 voting, test on 2020-21..2025-26
    python3 fc_awards.py sim       # award odds for 2026-27 from the saved simulations (fc_sim.py)
"""
import os, sys, json, pickle, unicodedata, re
import numpy as np, pandas as pd
import fc_project as FP

HERE = FP.HERE; OUT = FP.OUT; INP = FP.INP
AWARDS = ["mvp", "dpoy", "roy", "smoy", "mip"]
FEATS = {
    "mvp": ["value", "ppg", "winpct", "gp82"],
    "dpoy": ["dvalue", "stocks", "dreb", "winpct", "gp82"],
    "roy": ["value", "ppg", "gp82"],
    "smoy": ["value", "ppg", "gp82"],
    "mip": ["d_ppg", "d_value", "ppg", "gp82"],
}
LABEL = dict(mvp="Most Valuable Player", dpoy="Defensive Player of the Year", roy="Rookie of the Year", smoy="Sixth Man of the Year", mip="Most Improved Player", coy="Coach of the Year")


def norm_name(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s); s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return " ".join(s.split())


ALIAS = {"enes freedom": "enes kanter", "nene": "nene hilario"}


def votes():
    V = []
    for line in open(os.path.join(INP, "award_votes.txt"), encoding="utf-8"):
        head, rest = line.rstrip("\n").split(":", 1)
        y, aw = 2000 + int(head[:2]) - 1, head[2:]          # season start year
        for x in rest.split(";"):
            n, t, s = x.split("|")
            V.append(dict(season=y, award=aw, name=n, team=t, share=float(s or 0)))
    return pd.DataFrame(V)


_names = {}


def all_names():
    """pid -> name from every season's play-by-play names, the 2026-27 rosters and the injury page."""
    if not _names:
        for k in range(2010, 2026):
            n = pd.read_csv(os.path.join(OUT, f"names_{k}.csv")).dropna()
            _names.update({int(p): v for p, v in zip(n.pid, n.name)})
        for r in json.load(open(os.path.join(INP, "rosters_2026.json"))): _names[int(r[1])] = r[2]
        try:
            for r in json.load(open(os.path.join(os.path.dirname(HERE), "data", "load.json")))["rows"]:
                if not str(r["name"]).isdigit(): _names.setdefault(int(r["pid"]), r["name"])
        except Exception: pass
    return _names


def starts(y):
    """games started per player, from the first lineup of each game."""
    s = pickle.load(open(os.path.join(OUT, f"stints_{y}.pkl"), "rb"))
    seen = set(); c = {}
    for x in s:
        if x[2] != 1 or x[1] in seen: continue
        seen.add(x[1])
        for p in x[3] + x[4]: c[p] = c.get(p, 0) + 1
    return pd.Series(c)


def season_table(y):
    """player-season features from what happened (used to fit and test the voter models)."""
    g = FP.games(y); T = FP.tally(y) if hasattr(FP, "tally") else pickle.load(open(os.path.join(OUT, f"tally_{y}.pkl"), "rb"))
    S = FP.single(y)
    R = FP.results(y)
    wins = pd.concat([R.assign(t=R.home, w=(R.hp > R.ap)), R.assign(t=R.away, w=(R.ap > R.hp))]).groupby("t").w.mean()
    st = starts(y)
    names = all_names()
    prev = FP.games(y - 1) if y > 2010 else None
    Tp = pickle.load(open(os.path.join(OUT, f"tally_{y - 1}.pkl"), "rb")) if y > 2010 else {}
    Sp = FP.single(y - 1) if y > 2012 else None
    first = {}
    for k in range(2010, y + 1):
        for p in FP.games(k).index: first.setdefault(p, k)
    rows = []
    for p in g.index:
        t = T.get(p, {}); gp = g.loc[p, "gp"]
        if gp < 10: continue
        r = dict(pid=int(p), season=y, name=names.get(p, ""), team=g.loc[p, "team"], gp=gp, mpg=g.loc[p, "mpg"],
                 ppg=t.get("pts", 0) / gp, rpg=(t.get("oreb", 0) + t.get("dreb", 0)) / gp, apg=t.get("ast", 0) / gp,
                 stocks=(t.get("stl", 0) + t.get("blk", 0)) / gp, dreb=t.get("dreb", 0) / gp,
                 winpct=wins.get(g.loc[p, "team"], 0.5), start_share=st.get(p, 0) / gp,
                 net=S.net.get(p, np.nan), o=S.o.get(p, np.nan), d=S.d.get(p, np.nan),
                 rookie=first.get(p) == y and y > 2010)
        if prev is not None and p in prev.index and prev.loc[p, "gp"] >= 10:
            tp = Tp.get(p, {}); r["ppg_prev"] = tp.get("pts", 0) / prev.loc[p, "gp"]
            r["net_prev"] = Sp.net.get(p, np.nan) if Sp is not None else np.nan
            r["mpg_prev"] = prev.loc[p, "mpg"]
        rows.append(r)
    D = pd.DataFrame(rows)
    return add_feats(D)


def add_feats(D):
    D = D.copy()
    D["gp82"] = D.gp / 82
    D["value"] = D.net.fillna(-2) * D.mpg / 36 * np.minimum(D.gp82, 1)
    D["dvalue"] = D.d.fillna(-1) * D.mpg / 36 * np.minimum(D.gp82, 1)
    D["d_ppg"] = D.ppg - D.get("ppg_prev", np.nan)
    D["d_value"] = D.value - D.get("net_prev", np.nan).fillna(-2) * D.get("mpg_prev", D.mpg).fillna(D.mpg) / 36
    return D


def pool(D, aw):
    if aw in ("mvp", "dpoy"): m = (D.gp >= 40) & (D.mpg >= 24)
    elif aw == "roy": m = D.rookie & (D.gp >= 30)
    elif aw == "smoy": m = (D.gp >= 40) & (D.start_share < 0.5) & (D.mpg >= 15)
    else: m = (D.gp >= 40) & D.d_ppg.notna() & (D.mpg >= 20)
    return D[m]


def zs(X):
    return (X - X.mean()) / X.std().replace(0, 1)


def fit(aw, tables, V):
    """conditional logit on vote shares (normalized within season), features standardized within season."""
    from scipy.optimize import minimize
    blocks = []
    for y, D in tables.items():
        P = pool(D, aw).copy()
        if not len(P): continue
        v = V[(V.season == y) & (V.award == aw)]
        P["key"] = P.name.map(norm_name).replace(ALIAS)
        sh = {ALIAS.get(norm_name(n), norm_name(n)): s for n, s in zip(v.name, v.share)}
        P["share"] = P.key.map(sh).fillna(0)
        if P.share.sum() == 0: continue
        X = zs(P[FEATS[aw]].fillna(0)).values; w = (P.share / P.share.sum()).values
        blocks.append((X, w))
    k = len(FEATS[aw])
    def nll(b):
        tot = 0; grad = np.zeros(k)
        for X, w in blocks:
            s = X @ b; s -= s.max(); p = np.exp(s); p /= p.sum()
            tot -= (w * np.log(p + 1e-12)).sum(); grad -= X.T @ (w - p)
        return tot + 0.01 * (b @ b), grad + 0.02 * b
    r = minimize(nll, np.ones(k), jac=True, method="L-BFGS-B")
    return r.x


def probs(aw, b, D):
    P = pool(D, aw)
    X = zs(P[FEATS[aw]].fillna(0)).values; s = X @ b; s -= s.max(); p = np.exp(s); p /= p.sum()
    return pd.Series(p, index=P.pid.values)


def test():
    V = votes()
    tables = {y: season_table(y) for y in range(2012, 2026)}        # single-season ratings start 2012
    out = {}
    for aw in AWARDS:
        b = fit(aw, {y: tables[y] for y in range(2012, 2020)}, V)
        res = []
        for y in range(2020, 2026):
            p = probs(aw, b, tables[y]); D = tables[y].set_index("pid")
            v = V[(V.season == y) & (V.award == aw)].sort_values("share", ascending=False)
            winner = v.name.iloc[0]
            top = D.name.get(p.idxmax(), "?")
            res.append(dict(season=f"{y}-{str(y + 1)[2:]}", actual=winner, model_top=top, hit=norm_name(top) == norm_name(winner),
                            p_actual=float(p[[q for q in p.index if norm_name(D.name.get(q, "")) == norm_name(winner)]].sum())))
        hits = sum(r["hit"] for r in res)
        out[aw] = dict(coef=dict(zip(FEATS[aw], [float(x) for x in b])), seasons=res, hits=hits, of=len(res), pass_=bool(hits >= 4))
        print(aw, hits, "/", len(res), [(r["season"], r["actual"], r["model_top"]) for r in res], flush=True)
    # final coefficients on all seasons, used for 2026-27
    out["final"] = {aw: dict(zip(FEATS[aw], [float(x) for x in fit(aw, tables, V)])) for aw in AWARDS}
    json.dump(out, open(os.path.join(OUT, "fc_awards_test.json"), "w"), indent=1)
    return out




# ------------------------------------------------------------------ coach of the year (team level)
BBREF = {"PHO": "PHX", "CHO": "CHA", "BRK": "BKN", "NOH": "NOP", "NJN": "BKN", "CHA": "CHA"}


def coy_tables():
    """team-seasons: win share, preseason expectation (backtest preseason strength), COY vote share."""
    import fc_sim
    S = pd.read_csv(os.path.join(OUT, "fc_bt_teams.csv")); bt = json.load(open(os.path.join(OUT, "fc_backtest.json")))
    A = {int(k): v for k, v in json.load(open(os.path.join(HERE, "coaches.json")))["teams"].items()}
    V = votes(); V = V[V.award == "coy"]
    rows = []
    for r in S.itertuples():
        t = A.get(int(r.team), "?")
        R = FP.results(r.season)
        g = R[(R.home == r.team) | (R.away == r.team)]
        w = ((g.home == r.team) & (g.hp > g.ap) | (g.away == r.team) & (g.ap > g.hp)).mean()
        exp = 0.5 + bt["scale"]["k"] * r.model * 0.0333                 # about 3.3 points of win share per point of margin
        v = V[(V.season == r.season) & (V.team.map(lambda x: BBREF.get(x, x)) == t)]
        rows.append(dict(season=r.season, team=t, winpct=w, over=w - exp, share=float(v.share.sum())))
    return pd.DataFrame(rows)


def coy_fit(D):
    from scipy.optimize import minimize
    blocks = []
    for y, P in D.groupby("season"):
        if P.share.sum() == 0: continue
        X = zs(P[["winpct", "over"]]).values; blocks.append((X, (P.share / P.share.sum()).values))
    def nll(b):
        tot = 0; g = np.zeros(2)
        for X, w in blocks:
            s = X @ b; s -= s.max(); p = np.exp(s); p /= p.sum(); tot -= (w * np.log(p + 1e-12)).sum(); g -= X.T @ (w - p)
        return tot + 0.01 * b @ b, g + 0.02 * b
    return minimize(nll, np.ones(2), jac=True, method="L-BFGS-B").x


def coy_test():
    D = coy_tables(); b = coy_fit(D[D.season < 2020]); res = []
    for y in range(2020, 2026):
        P = D[D.season == y]; X = zs(P[["winpct", "over"]]).values; s = X @ b; p = np.exp(s - s.max()); p /= p.sum()
        top = P.team.values[p.argmax()]; act = P.team.values[P.share.values.argmax()]
        res.append(dict(season=f"{y}-{str(y + 1)[2:]}", actual=act, model_top=top, hit=bool(top == act)))
    out = dict(coef=dict(winpct=float(b[0]), over=float(b[1])), seasons=res, hits=sum(r["hit"] for r in res), of=len(res))
    out["final"] = dict(zip(["winpct", "over"], [float(x) for x in coy_fit(D)]))
    print("coy", out["hits"], "/", out["of"], [(r["season"], r["actual"], r["model_top"]) for r in res])
    return out


# ------------------------------------------------------------------ 2026-27 award odds from the simulations
def build_award_inputs():
    """static player inputs for the award simulation (from completed seasons), written once per season."""
    import fc_sim
    D = pd.read_csv(os.path.join(OUT, f"fc_players_{FP_SEASON}.csv"), index_col=0)
    last = season_table(FP_SEASON - 1).set_index("pid")
    rk = pd.concat([season_table(y) for y in range(2018, FP_SEASON)]); rk = rk[rk.rookie]
    dr = FP.draft(); rk["lp"] = [np.log(dr.get(int(p), (0, 75))[1]) for p in rk.pid]
    cr = np.polyfit(rk.lp, rk.ppg, 1)
    I = pd.DataFrame(index=D.index)
    I["ppm"] = (last.ppg / last.mpg).reindex(D.index)
    I["ppg_rookie"] = [np.polyval(cr, np.log(dr.get(int(p), (0, 75))[1])) for p in D.index]
    I["ppg_prev"] = last.ppg.reindex(D.index); I["net_prev"] = last.net.reindex(D.index); I["mpg_prev"] = last.mpg.reindex(D.index)
    I["o_prev"] = last.o.reindex(D.index)
    I["stocks"] = last.stocks.reindex(D.index); I["dreb"] = last.dreb.reindex(D.index)
    played = set()
    for k in range(2010, FP_SEASON): played |= set(FP.games(k).index)
    I["first_year"] = [p not in played for p in D.index]
    pick = pd.Series({p: dr.get(int(p), (0, 99))[1] for p in D.index})
    I["start_share"] = last.start_share.reindex(D.index)
    I.loc[I.start_share.isna(), "start_share"] = np.where(pick[I.start_share.isna()] <= 10, 0.6, 0.25)
    I.to_csv(os.path.join(OUT, f"fc_award_inputs_{FP_SEASON}.csv"))
    sim = json.load(open(os.path.join(OUT, f"fc_sim_{FP_SEASON}.json")))
    json.dump({t["team"]: t["proj_w"] for t in sim["teams"]}, open(os.path.join(OUT, f"fc_preseason_{FP_SEASON}.json"), "w"))
    print("award inputs written", len(I))


def sim_awards(nmax=100_000, chunk=10_000, seed=7):
    import fc_sim
    T = json.load(open(mpath("fc_awards_test.json")))
    Z = np.load(os.path.join(OUT, "fc_sim_players.npz")); A = Z["avail"][:nmax]; W = Z["wins"][:nmax]; pids = Z["pids"]
    D = pd.read_csv(os.path.join(OUT, "fc_players_now.csv"), index_col=0).loc[pids]
    lp = os.path.join(OUT, "fc_live_players.csv")
    L = pd.read_csv(lp, index_col=0).reindex(D.index) if os.path.exists(lp) else pd.DataFrame(index=D.index, columns=["gp", "pts", "starts"])
    L = L[["gp", "pts", "starts"]].astype(float).fillna(0)
    team_rem = (Z["team_rem"] if "team_rem" in Z else np.full(30, 82)).astype(np.int64); team_played = Z["team_played"] if "team_played" in Z else np.zeros(30)
    I = pd.read_csv(mpath(f"fc_award_inputs_{FP_SEASON}.csv"), index_col=0)
    D = D.join(I.drop(columns=[c_ for c_ in ("name", "team") if c_ in I.columns]), how="left")
    D["first_year"] = D.first_year.fillna(False).astype(bool)
    D["ppg_proj"] = np.where(D.ppm.notna(), D.ppm * D.mpg, D.ppg_rookie).astype(float)
    D["ppg_proj"] = pd.Series(D.ppg_proj, index=D.index).fillna(4.0).clip(lower=0)
    D["stocks"] = D.stocks.fillna(1.0); D["dreb"] = D.dreb.fillna(2.5); D["start_share"] = D.start_share.fillna(0.25)
    cur = L.gp >= 5
    D.loc[cur, "start_share"] = (L.starts / L.gp.clip(lower=1))[cur]           # this season's starts once he has 5 games
    D["ppg_grow"] = (1.5 * (D.o - D.o_prev)).fillna(0).clip(-3, 3)
    D["ppg_sd"] = 1.2 + 0.35 * np.clip(27 - D.age.fillna(27), 0, None)
    team_i = D.team.map(fc_sim.TI).values
    sd_o, sd_d = 1.6, 1.2                                    # season-to-season spread of a player's rating around his projection
    rng = np.random.default_rng(seed)
    acc = {aw: np.zeros(len(D)) for aw in AWARDS}
    acc_coy = np.zeros(30); n = 0
    S = len(A)
    coef = {aw: np.array([T["final"][aw][f] for f in FEATS[aw]]) for aw in AWARDS}
    ccoy = np.array([T["coy"]["final"]["winpct"], T["coy"]["final"]["over"]])
    pre = json.load(open(mpath(f"fc_preseason_{FP_SEASON}.json")))
    exp_w = np.array([pre[tm] for tm in fc_sim.TEAMS]) / 82          # preseason expected wins (for "beat expectations")
    for s0 in range(0, S, chunk):
        a = A[s0:s0 + chunk]; w = W[s0:s0 + chunk]; B = len(a)
        rem_n = team_rem[team_i]; frac = np.sqrt(np.clip(rem_n / 82, 0, 1))          # what is left to play shrinks the uncertainty
        gp_rest = rng.binomial(rem_n, np.clip(a, 0, 1)); gp = L.gp.values + gp_rest
        zo = rng.normal(0, 1, (B, len(D))); zd = rng.normal(0, 1, (B, len(D)))
        o = D.o.values + sd_o * frac * zo; d = D.d.values + sd_d * frac * zd; net = o + d
        mpg = D.mpg.values; gp82 = gp / 82
        ppg_rest = np.clip(D.ppg_proj.values + D.ppg_grow.values + D.ppg_sd.values * (0.6 * zo + 0.8 * rng.normal(0, 1, (B, len(D)))), 0, None)
        ppg = (L.pts.values + ppg_rest * gp_rest) / np.maximum(gp, 1)
        winpct = w[:, team_i] / 82
        F = dict(value=net * mpg / 36 * np.minimum(gp82, 1), dvalue=d * mpg / 36 * np.minimum(gp82, 1), ppg=ppg, winpct=winpct, gp82=gp82,
                 stocks=np.broadcast_to(D.stocks.values, (B, len(D))), dreb=np.broadcast_to(D.dreb.values, (B, len(D))),
                 d_ppg=ppg - D.ppg_prev.values, d_value=net * mpg / 36 * np.minimum(gp82, 1) - np.nan_to_num(D.net_prev.values, nan=-2) * np.nan_to_num(D.mpg_prev.values, nan=10) / 36)
        elig = gp >= 65                                                      # the 65-game rule
        for aw in AWARDS:
            if aw in ("mvp", "dpoy"): m = elig & (mpg >= 24)
            elif aw == "roy": m = elig & D.first_year.values[None, :]
            elif aw == "smoy": m = elig & (D.start_share.values < 0.5)[None, :] & (mpg >= 15)
            else: m = elig & ~np.isnan(D.ppg_prev.values)[None, :] & (mpg >= 20)
            sc = np.zeros((B, len(D)))
            for j, f in enumerate(FEATS[aw]):
                x = np.where(m, F[f], np.nan)
                mu = np.nanmean(x, axis=1, keepdims=True); sd = np.nanstd(x, axis=1, keepdims=True); sd[sd == 0] = 1
                sc += coef[aw][j] * np.nan_to_num((x - mu) / sd)
            sc = np.where(m, sc, -np.inf); sc -= sc.max(axis=1, keepdims=True)
            p = np.exp(sc); p /= p.sum(axis=1, keepdims=True)
            acc[aw] += np.nan_to_num(p).sum(0)
        wp = w / 82; ov = wp - exp_w
        X = np.stack([(wp - wp.mean(1, keepdims=True)) / wp.std(1, keepdims=True), (ov - ov.mean(1, keepdims=True)) / ov.std(1, keepdims=True)], -1)
        s = X @ ccoy; p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); acc_coy += p.sum(0)
        n += B
    out = {}
    for aw in AWARDS:
        p = acc[aw] / n; o_ = np.argsort(-p)[:15]
        out[aw] = [dict(pid=int(D.index[i]), name=D.name.iloc[i], team=D.team.iloc[i], p=float(p[i])) for i in o_ if p[i] > 0.0005]
    pc = acc_coy / n
    out["coy"] = [dict(team=fc_sim.TEAMS[i], p=float(pc[i])) for i in np.argsort(-pc)[:10]]
    # chance each player clears 65 games
    g65 = ((L.gp.values + np.random.default_rng(1).binomial(team_rem[team_i], np.clip(A[:20000], 0, 1))) >= 65).mean(0)
    stars = D.assign(g65=g65).sort_values("net", ascending=False).head(40)
    out["games65"] = [dict(pid=int(p), name=r["name"], team=r["team"], p=float(r["g65"])) for p, r in stars.iterrows()]
    out["sims"] = n
    json.dump(out, open(os.path.join(OUT, f"fc_awards_{FP_SEASON}.json"), "w"), indent=1)
    for aw in AWARDS + ["coy"]: print(aw, [(x.get("name", x.get("team")), round(x["p"], 3)) for x in out[aw][:6]])
    return out


FP_SEASON = 2026
STATIC = os.path.join(HERE, "forecast_model")


def mpath(name):
    """static model files committed to the repo (pipeline/forecast_model/), falling back to out/."""
    p = os.path.join(STATIC, name)
    return p if os.path.exists(p) else os.path.join(OUT, name)


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        o = test(); o["coy"] = coy_test(); json.dump(o, open(os.path.join(OUT, "fc_awards_test.json"), "w"), indent=1)
    if sys.argv[1:] == ["inputs"]: build_award_inputs()
    if sys.argv[1:] == ["sim"]: sim_awards()
