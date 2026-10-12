"""Team-level tests for the Load & Durability model (TM1-TM3 in LOAD_PREREG.txt).

    python pipeline/load_team.py

Needs pipeline/out/load_features.csv.gz (from load_model.py) and the player-game tables.
Writes pipeline/out/load_team.json and pipeline/out/load_team_tests.json.
"""
import os, json
import numpy as np, pandas as pd
import load_model as LM

OUT = LM.OUT


def runs_by_player(G, TG):
    """for every player-season-team stretch: list of (start index, length) of missed runs (any length) in team-game order."""
    g = G[G.kind == "rs"]; tg = TG[TG.kind == "rs"].sort_values(["season", "team", "date", "gid"]).copy()
    tg["k"] = tg.groupby(["season", "team"]).cumcount()
    n = tg.groupby(["season", "team"]).k.max().add(1)
    kidx = dict(zip(zip(tg.season, tg.team, tg.gid), tg.k))
    g = g.assign(k=[kidx.get(x) for x in zip(g.season, g.team, g.gid)]).dropna(subset=["k"])
    out = []
    for (s, p, t), x in g.groupby(["season", "pid", "team"]):
        ks = set(x.k.astype(int)); lo, hi = min(ks), max(ks)
        if hi - lo < 10: continue
        r, start = 0, None
        for k in range(lo, hi + 1):
            if k not in ks:
                if r == 0: start = k
                r += 1
            else:
                if r: out.append((s, p, t, start, r))
                r = 0
    return pd.DataFrame(out, columns=["season", "pid", "team", "start", "len"])


def rehab(R):
    """re-absence: a new run of 3+ that starts within 15 games of coming back from a 10+ game run."""
    rows = []
    for (s, p, t), x in R.sort_values("start").groupby(["season", "pid", "team"]):
        st, ln = x.start.values, x.len.values
        for i in range(len(st)):
            if ln[i] >= 10:
                back = st[i] + ln[i]
                nxt = [(a, b) for a, b in zip(st[i + 1:], ln[i + 1:]) if b >= 3]
                re = bool(nxt and nxt[0][0] - back <= 15)
                rows.append((s, p, t, ln[i], re))
    return pd.DataFrame(rows, columns=["season", "pid", "team", "len", "reabs"])


def main():
    G, TG, names = LM.load_games()
    D = pd.read_csv(os.path.join(OUT, "load_features.csv.gz"))
    D["tload"] = D.next_team_load.fillna(D.team_load)
    S = D[(D.min_rs >= LM.MIN_MIN) & (D.season >= LM.FIRST + 2) & (D.season <= LM.LAST - 1)].dropna(subset=["y_missed"]).reset_index(drop=True)
    full = ["H", "A", "P", "M", "W", "T"]
    mu, pb = LM.loso(S, full)
    S["exp"] = mu; S["resid"] = S.y_missed - mu
    # TM1 team health effect: by the team he played for in the outcome season (his next team)
    te = S.groupby(["next_team", "season"]).agg(eff=("resid", "mean"), n=("pid", "size")).reset_index()
    te["season_out"] = te.season + 1
    a = te.rename(columns={"eff": "eff0"}); b = te[["next_team", "season", "eff"]].copy(); b["season"] -= 1
    pair = a.merge(b, on=["next_team", "season"])
    r_tm1 = float(np.corrcoef(pair.eff0, pair.eff)[0, 1])
    # as a model input: last season's effect for his next team (known before the season)
    prev = te[["next_team", "season", "eff"]].copy(); prev["season"] += 1
    S2 = S.merge(prev.rename(columns={"eff": "team_eff_prev"}), on=["next_team", "season"], how="left")
    yb = (S2.y_missed >= LM.BIG_MISS).astype(int).values
    # leave-one-season-out with the team effect recomputed without the held-out outcome season
    LM.FEATS["E"] = ["team_eff_prev"]
    mu0, pb0 = LM.loso(S2, full)
    pbE = np.zeros(len(S2))
    for s in sorted(S2.season.unique()):
        tr = S2[S2.season != s].copy(); tes = S2[S2.season == s].copy()
        # team effect from training seasons only: residuals computed above are out-of-sample per player, but the
        # outcome season s itself must not feed the team effect used for season s; prev effects come from season s-1 outcomes, which is fine
        _, p, *_ = LM.fit_predict(tr, tes, full + ["E"])
        pbE[S2.season.values == s] = p
    b0, bE = LM.brier(pb0, yb), LM.brier(pbE, yb)
    TM1 = dict(r_repeat=r_tm1, pairs=len(pair), pass_repeat=r_tm1 >= 0.20, brier_without=b0, brier_with=bE,
               pass_input=(b0 - bE) / b0 >= 0.005)
    print("TM1", TM1)

    # TM2 rehab
    R = runs_by_player(G, TG)
    RH = rehab(R)
    RH = RH[RH.season.between(LM.FIRST, LM.LAST)]
    tr = RH.groupby(["team", "season"]).agg(rate=("reabs", "mean"), n=("reabs", "size")).reset_index()
    tr = tr[tr.n >= 3]
    nx = tr[["team", "season", "rate"]].copy(); nx["season"] -= 1
    pr = tr.merge(nx, on=["team", "season"], suffixes=("", "_next"))
    r_tm2 = float(np.corrcoef(pr.rate, pr.rate_next)[0, 1])
    # player input: any re-absence in seasons t-1..t
    pre = RH.groupby(["pid", "season"]).reabs.max().reset_index()
    pre2 = pre.copy(); pre2["season"] += 1
    pp = pd.concat([pre, pre2]).groupby(["pid", "season"]).reabs.max().rename("reabs_2y").reset_index()
    S3 = S.merge(pp, on=["pid", "season"], how="left"); S3["reabs_2y"] = S3.reabs_2y.fillna(False).astype(float)
    LM.FEATS["R"] = ["reabs_2y"]
    m3, p3 = LM.loso(S3, full + ["R"])
    b3 = LM.brier(p3, (S3.y_missed >= LM.BIG_MISS).astype(int).values)
    TM2 = dict(events=int(len(RH)), reabsence_rate=float(RH.reabs.mean()), r_repeat=r_tm2, pairs=len(pr), pass_repeat=r_tm2 >= 0.20,
               brier_without=b0, brier_with=b3, pass_input=(b0 - b3) / b0 >= 0.005)
    print("TM2", TM2)

    # TM3 rest: 1-2 game runs per player-season of rotation players (20+ mpg), by team
    W = D[["season", "pid", "mpg"]]
    short = R[R.len <= 2].groupby(["season", "pid", "team"]).size().rename("short").reset_index()
    base = R.groupby(["season", "pid", "team"]).size().reset_index()[["season", "pid", "team"]]
    allp = G[G.kind == "rs"].groupby(["season", "pid", "team"]).size().rename("gp").reset_index()
    allp = allp.merge(short, on=["season", "pid", "team"], how="left").fillna({"short": 0}).merge(W, on=["season", "pid"], how="left")
    allp = allp[(allp.mpg >= 20) & (allp.gp >= 20)]
    rest = allp.groupby(["team", "season"]).apply(lambda x: (x.short * 82 / x.gp.clip(lower=1)).mean(), include_groups=False).rename("rest").reset_index()
    rnx = rest.copy(); rnx["season"] -= 1
    rp = rest.merge(rnx, on=["team", "season"], suffixes=("", "_next"))
    TM3 = dict(r_repeat=float(np.corrcoef(rp.rest, rp.rest_next)[0, 1]), league_by_season=rest.groupby("season").rest.mean().round(2).to_dict())
    print("TM3", TM3)

    json.dump(dict(TM1=TM1, TM2=TM2, TM3=TM3), open(os.path.join(OUT, "load_team_tests.json"), "w"), indent=1, default=float)
    te.to_csv(os.path.join(OUT, "team_effect.csv"), index=False)
    tr.to_csv(os.path.join(OUT, "team_rehab.csv"), index=False)
    rest.to_csv(os.path.join(OUT, "team_rest.csv"), index=False)
    RH.to_csv(os.path.join(OUT, "rehab_events.csv"), index=False)
    S[["season", "pid", "exp", "resid", "next_team"]].to_csv(os.path.join(OUT, "loso_pred.csv"), index=False)


if __name__ == "__main__":
    main()
