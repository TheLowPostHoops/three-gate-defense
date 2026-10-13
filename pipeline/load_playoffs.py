"""Playoff load and games missed the next season (tests P1-P4 in LOAD_PREREG.txt).

    python pipeline/load_playoffs.py

Needs pipeline/out/load_features.csv.gz (load_model.py), v2_features.csv.gz (load_v2.py) and the player-game tables.
Writes pipeline/out/playoffs_tests.json.
"""
import os, json
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
import load_model as LM
from load_v2 import exact_age

OUT = LM.OUT


def bins(n):
    return np.select([n == 0, n <= 6, n <= 13], ["none", "1-6", "7-13"], "14+")


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def team_playoffs(TG):
    po = TG[TG.kind == "po"].groupby(["season", "team"]).gid.nunique().rename("team_po").reset_index()
    last = pd.concat([TG.groupby(["season", "team"]).date.max().rename("last_game").reset_index()])
    opener = TG[TG.kind == "rs"].groupby("season").date.min()
    last["next_opener"] = last.season.map(lambda s: opener.get(s + 1))
    last["offseason_days"] = (last.next_opener - last.last_game).dt.days
    return po, last


def p1(S, y, label):
    """raw and adjusted rates of 20+ missed games by team playoff games."""
    S = S.copy(); S["yb"] = (S[y] >= LM.BIG_MISS).astype(int); S["grp"] = pd.Categorical(bins(S.team_po), ["none", "1-6", "7-13", "14+"])
    raw = []
    for g, x in S.groupby("grp", observed=True):
        lo, hi = wilson(x.yb.sum(), len(x)); raw.append(dict(group=str(g), n=len(x), rate=float(x.yb.mean()), lo=lo, hi=hi, avg_missed=float(x[y].mean())))
    f = "yb ~ C(grp, Treatment('none')) + min_rs_k + mpg + age_next + I((age_next-27)**2) + C(pos) + hist0 + hist1 + C(season)"
    m = smf.logit(f, data=S).fit(disp=0, cov_type="cluster", cov_kwds={"groups": S.pid.values})
    ors = {}
    for g in ["1-6", "7-13", "14+"]:
        k = f"C(grp, Treatment('none'))[T.{g}]"
        if k in m.params: ors[g] = dict(odds_ratio=float(np.exp(m.params[k])), lo=float(np.exp(m.conf_int().loc[k, 0])), hi=float(np.exp(m.conf_int().loc[k, 1])), p=float(m.pvalues[k]))
    return dict(label=label, n=len(S), raw=raw, adjusted=ors, pass_=bool(ors.get("14+", {}).get("lo", 0) > 1))


def p2(S, y, extra):
    S = S.copy(); S["y_missed"] = S[y]
    LM.FEATS["CTL"] = ["min_rs_k", "mpg", "age_next", "age_sq", "pos_big", "pos_guard", "hist0", "hist1"]
    LM.FEATS["X"] = [extra]
    S["age_sq"] = (S.age_next - 27) ** 2; S["pos_big"] = (S.pos == "big").astype(float); S["pos_guard"] = (S.pos == "guard").astype(float)
    yb = (S[y] >= LM.BIG_MISS).astype(int).values
    _, p0 = LM.loso(S, ["CTL"]); _, p1_ = LM.loso(S, ["CTL", "X"])
    b0, b1 = LM.brier(p0, yb), LM.brier(p1_, yb)
    return dict(exposure=extra, brier_without=b0, brier_with=b1, gain=(b0 - b1) / b0, pass_=(b0 - b1) / b0 >= 0.005)


def p3(S, y):
    """each player compared with himself: demean everything within player."""
    S = S[S.groupby("pid").pid.transform("size") >= 2].copy()
    cols = [y, "team_po", "min_rs_k", "mpg", "age_next", "hist0"]
    W = S[cols].astype(float) - S.groupby("pid")[cols].transform("mean")
    X = sm.add_constant(W[["team_po", "min_rs_k", "mpg", "age_next", "hist0"]])
    m = sm.OLS(W[y], X).fit(cov_type="cluster", cov_kwds={"groups": S.pid.values})
    ci = m.conf_int().loc["team_po"]
    per10 = dict(coef=float(m.params["team_po"] * 10), lo=float(ci[0] * 10), hi=float(ci[1] * 10))
    return dict(n=len(S), players=int(S.pid.nunique()), games_missed_per_10_playoff_games=per10, pass_=bool(ci[0] > 0))


def main():
    G, TG, names = LM.load_games()
    po, last = team_playoffs(TG)
    F = exact_age(pd.read_csv(os.path.join(OUT, "load_features.csv.gz")))
    F = F.merge(po.rename(columns={"team": "last_team"}), on=["season", "last_team"], how="left").fillna({"team_po": 0})
    F = F.merge(last.rename(columns={"team": "last_team"})[["season", "last_team", "offseason_days"]], on=["season", "last_team"], how="left")
    tp = F.set_index(["season", "pid"]).team_po
    F["team_po_3y"] = [sum(tp.get((s - k, p), 0) for k in range(3)) for s, p in zip(F.season, F.pid)]
    F["min_rs_k"] = F.min_rs / 1000; F["hist0"] = F.missed.fillna(0); F["hist1"] = F.missed_l1.fillna(F.missed_l1.mean())
    F["po_mpg"] = F.min_po / F.team_po.replace(0, np.nan)

    seasons = F.groupby("pid").season.apply(set)
    gone = F.y_missed.isna() & np.array([(s + 2 in seasons[p]) or (s + 3 in seasons[p]) for s, p in zip(F.season, F.pid)])
    F["y82"] = F.y_missed; F.loc[gone, "y82"] = 82.0                 # P5: a season with no games, before he played again
    H = F[(F.min_rs >= 1500) & (F.mpg >= 25) & F.season.between(2010, 2024) & F.y_missed.notna() & F.age_next.notna()].reset_index(drop=True)
    R = {}
    R["sample_H"] = dict(n=len(H), seasons="2010-11 to 2024-25 (outcomes 2011-12 to 2025-26)", players=int(H.pid.nunique()))
    R["P1"] = p1(H, "y_missed", "long history, missed games (runs of 3+), regulars")
    R["P2"] = p2(H, "y_missed", "team_po"); R["P2_3y"] = p2(H, "y_missed", "team_po_3y"); R["P2_minutes"] = p2(H, "y_missed", "min_po")
    R["P3"] = p3(H, "y_missed")
    H5 = F[(F.min_rs >= 1500) & (F.mpg >= 25) & F.season.between(2010, 2024) & F.y82.notna() & F.age_next.notna()].reset_index(drop=True)
    w5 = H5.y_missed.isna()
    R["P5"] = dict(added=int(w5.sum()), p1=p1(H5, "y82", "including whole missed seasons as 82"), p3=p3(H5, "y82"),
                   whole=[dict(player=names.get(p, str(p)), season=f"{s}-{str(s + 1)[2:]}", team_po=int(t)) for s, p, t in zip(H5.season[w5], H5.pid[w5], H5.team_po[w5])])
    # three-year load, raw
    H3 = H.assign(g3=pd.cut(H.team_po_3y, [-1, 0, 15, 35, 200], labels=["none", "1-15", "16-35", "36+"]))
    R["three_year"] = [dict(group=str(g), n=len(x), rate20=float((x.y_missed >= 20).mean()), avg=float(x.y_missed.mean())) for g, x in H3.groupby("g3", observed=True)]
    # official labels
    V = pd.read_csv(os.path.join(OUT, "v2_features.csv.gz"))
    V = V.merge(F[["season", "pid", "team_po", "team_po_3y", "min_rs_k", "hist0", "hist1"]], on=["season", "pid"], how="left")
    RS = V[(V.min_rs >= LM.MIN_MIN) & V.season.between(2021, 2024) & V.y_inj.notna() & V.age_next.notna()].reset_index(drop=True)
    RS["hist0"] = RS.inj_l0.fillna(0); RS["hist1"] = RS.inj_l1.fillna(RS.inj_l1.mean())
    try: R["P4"] = dict(p1=p1(RS, "y_inj", "official injury games, 2021-22 on"), p2=p2(RS, "y_inj", "team_po"))
    except Exception as e: R["P4"] = dict(error=str(e))
    # descriptive: playoff minutes bump for each team's top five; offseason length
    rs = G[G.kind == "rs"].groupby(["season", "team", "pid"]).secs.agg(["sum", "size"]); rs["mpg"] = rs["sum"] / 60 / rs["size"]
    pp = G[G.kind == "po"].groupby(["season", "team", "pid"]).secs.agg(["sum", "size"]); pp["mpg"] = pp["sum"] / 60 / pp["size"]
    pp = pp.rename(columns={"mpg": "mpg_po", "size": "gp_po"})
    j = rs[["mpg", "sum"]].join(pp[["mpg_po", "gp_po"]], how="inner").reset_index()
    j = j[j.gp_po >= 4]
    top = j.sort_values("sum").groupby(["season", "team"]).tail(5)
    R["top5_bump"] = dict(avg=float((top.mpg_po - top.mpg).mean()), by_season=(top.assign(b=top.mpg_po - top.mpg).groupby("season").b.mean().round(2).to_dict()))
    R["offseason"] = H.assign(grp=bins(H.team_po)).groupby("grp").offseason_days.mean().round(0).to_dict()
    # Warriors core
    core = {"Stephen Curry": 201939, "Klay Thompson": 202691, "Draymond Green": 203110, "Andre Iguodala": 2738, "Kevin Durant": 201142}
    W = []
    for n, p in core.items():
        for s in range(2014, 2022):
            x = F[(F.pid == p) & (F.season == s)]
            if len(x): W.append(dict(player=n, season=f"{s}-{str(s + 1)[2:]}", team_po=int(x.team_po.iloc[0]), min_po=float(x.min_po.iloc[0]), three_year=int(x.team_po_3y.iloc[0]),
                                     missed_next=None if pd.isna(x.y82.iloc[0]) else float(x.y82.iloc[0]), whole_season=bool(pd.isna(x.y_missed.iloc[0]) and pd.notna(x.y82.iloc[0]))))
    R["warriors"] = W
    json.dump(R, open(os.path.join(OUT, "playoffs_tests.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in R.items() if k not in ("warriors",)}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
