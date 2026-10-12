"""Write the Load & Durability page data: data/load.json (players), data/load_teams.json, data/load_meta.json.

    python pipeline/load_publish.py

Runs after load_model.py, load_team.py and load_extra.py. Only the blocks that passed a pre-set test (none of the
add-ons did) go into the risk number: absence history, age and position. Everything else is shown as a description.
"""
import os, json, datetime
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import load_model as LM

OUT = LM.OUT; PUB = LM.PUB
SHIP = ["H", "A", "P"]
CUR = LM.LAST            # 2025 = 2025-26, the last completed season; projection is for 2026-27


def r(x, n=1):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def main():
    G, TG, names = LM.load_games()
    D = pd.read_csv(os.path.join(OUT, "load_features_plus.csv.gz"))
    D["tload"] = D.next_team_load.fillna(D.team_load)
    cj = json.load(open(os.path.join(LM.HERE, "coaches.json")))
    abbr = {int(k): v for k, v in cj["teams"].items()}
    S = D[(D.min_rs >= LM.MIN_MIN) & (D.season >= LM.FIRST + 2) & (D.season <= LM.LAST - 1)].dropna(subset=["y_missed"]).reset_index(drop=True)
    yb = (S.y_missed >= LM.BIG_MISS).astype(int).values

    # ---- backtest of the shipped model (out of sample) + range and calibration checks
    mu, pb = LM.loso(S, SHIP)
    bmu, bpb = LM.baseline(S)
    lo, hi = LM.ranges(S, mu)
    cover = float(np.mean((S.y_missed.values >= lo) & (S.y_missed.values <= hi)))
    q = pd.qcut(pb, 5, labels=False)
    calib = [dict(pred=float(pb[q == i].mean()), actual=float(yb[q == i].mean()), n=int((q == i).sum())) for i in range(5)]
    BT = dict(n=len(S), seasons="2013-14 to 2025-26 outcomes", mae=float(np.mean(np.abs(mu - S.y_missed))), mae_base=float(np.mean(np.abs(bmu - S.y_missed))),
              brier=LM.brier(pb, yb), brier_base=LM.brier(bpb, yb), auc=float(roc_auc_score(yb, pb)), auc_base=float(roc_auc_score(yb, bpb)),
              cover80=cover, L4_pass=0.74 <= cover <= 0.86, calib=calib, L5_pass=all(abs(c["pred"] - c["actual"]) <= 0.05 for c in calib),
              rate20=float(yb.mean()), avg_missed=float(S.y_missed.mean()), median_missed=float(np.median(S.y_missed)))
    # also the full model's AUC for the record
    fmu, fpb = LM.loso(S, ["H", "A", "P", "M", "W", "T"])
    BT["auc_full"] = float(roc_auc_score(yb, fpb))
    print("backtest", {k: v for k, v in BT.items() if k != "calib"})

    # ---- final fit on every pair, projection from the 2025-26 season
    Pq = D[(D.season == CUR) & (D.min_rs >= LM.MIN_MIN)].copy().reset_index(drop=True)
    _, pp, pm, lm, st = LM.fit_predict(S, Pq, SHIP)
    Z, _ = LM.design(Pq, SHIP, st)
    Pq["exp"] = pm.predict(Z); Pq["p20"] = pp
    # 80% ranges from the backtest: bins of the out-of-sample forecast
    edges = np.quantile(mu, np.linspace(0, 1, 11)); edges[0] = -1; edges[-1] = 1e9
    b = np.digitize(mu, edges) - 1
    ql = [float(np.quantile(S.y_missed[b == i], .1)) for i in range(10)]; qh = [float(np.quantile(S.y_missed[b == i], .9)) for i in range(10)]
    bi = (np.digitize(Pq.exp, edges) - 1).clip(0, 9)
    Pq["lo"] = [ql[i] for i in bi]; Pq["hi"] = [qh[i] for i in bi]
    # contributions (log scale of the expected-games model): which inputs push the number up or down
    coef = dict(zip(Z.columns, pm.coef_))
    def contrib(i):
        z = Z.iloc[i]
        return dict(history=r(sum(coef[c] * z[c] for c in LM.FEATS["H"]), 3), age=r(sum(coef[c] * z[c] for c in LM.FEATS["A"]), 3),
                    position=r(sum(coef[c] * z[c] for c in LM.FEATS["P"]), 3))
    # descriptions
    M = pd.read_csv(os.path.join(LM.HERE, "inputs", "miles_seasons.csv")); M = M[M.speed <= 5.5]
    mrs = M[(M.kind == "rs") & (M.season == CUR)].groupby("pid").agg(mi=("miles", "sum"), mn=("min", "sum"), gp=("gp", "sum"), spd=("speed", "mean"))
    mrs["mi36"] = mrs.mi / mrs.mn.clip(lower=1) * 36; mrs["mipg"] = mrs.mi / mrs.gp.clip(lower=1)
    lg_mi36 = float(M[(M.kind == "rs") & (M.season == CUR) & (M["min"] >= 500)].pipe(lambda x: x.miles.sum() / x["min"].sum() * 36))
    rehab = pd.read_csv(os.path.join(OUT, "rehab_events.csv"))
    hist = D.set_index(["pid", "season"])
    firstseen = D.groupby("pid").season.min().to_dict()
    rows = []
    for i, x in Pq.sort_values("exp", ascending=False).reset_index().iterrows():
        p = int(x.pid)
        f0 = firstseen.get(p, 9999)
        h = [r(hist.missed.get((p, s)), 0) if (p, s) in hist.index else (82 if s > f0 else None) for s in (CUR - 4, CUR - 3, CUR - 2, CUR - 1, CUR)]
        mm = mrs.loc[p] if p in mrs.index else None
        rb = rehab[(rehab.pid == p) & (rehab.season >= CUR - 2)]
        rows.append([names.get(p, str(p)), abbr.get(int(x.last_team), "?"), r(x.age_next, 0), x.pos,
                     r(x.exp, 1), r(x["lo"], 0), r(x["hi"], 0), r(100 * x.p20, 0), h,
                     r(x.mpg, 1), int(round(x.gp)), r(100 * x.g38, 0), r(100 * x.b2b30, 0), r(x.spike, 2), r(x.ppm, 3),
                     r(x.career_min / 1000, 1), r(x.min_po, 0),
                     r(mm.mi36, 2) if mm is not None else None, r(mm.spd, 2) if mm is not None else None,
                     None if pd.isna(x.col_years) else int(x.col_years), r(x.col_min, 0) if not pd.isna(x.col_years) else None,
                     r(100 * x.col_missed, 0) if not pd.isna(x.col_years) else None, r(x.height, 0) if not pd.isna(x.height) else None,
                     int(x.no_college), int(len(rb)), int(rb.reabs.sum()), r(x.team_load, 1), contrib(int(x["index"])), p])
    cols = ["name", "team", "age", "pos", "exp", "lo", "hi", "p20", "missed_hist", "mpg", "gp", "pct38", "pct_b2b30", "spike", "pace_min",
            "career_k", "po_min", "mi36", "speed", "col_years", "col_min", "col_missed_pct", "height", "no_college", "long_abs_3y", "reabs_3y",
            "team_load", "contrib", "pid"]
    os.makedirs(PUB, exist_ok=True)
    json.dump(dict(cols=cols, rows=rows), open(os.path.join(PUB, "load.json"), "w"), separators=(",", ":"))

    # ---- teams and coaches
    TL = LM.team_load(G)
    rest = pd.read_csv(os.path.join(OUT, "team_rest.csv")); eff = pd.read_csv(os.path.join(OUT, "team_effect.csv")); reh = pd.read_csv(os.path.join(OUT, "team_rehab.csv"))
    eff = eff.rename(columns={"next_team": "team"}); eff["season"] = eff.season + 1          # effect measured in the outcome season
    T = TL.merge(rest, on=["team", "season"], how="left").merge(eff[["team", "season", "eff"]], on=["team", "season"], how="left").merge(reh[["team", "season", "rate", "n"]], on=["team", "season"], how="left")
    lg = T.groupby("season").team_load.mean()
    T["load_rel"] = T.team_load - T.season.map(lg)
    teams = []
    for t, x in T[T.season == CUR].iterrows():
        a = abbr.get(int(x.team), "?")
        teams.append(dict(team=a, coach=cj["current"].get(a), load=r(x.team_load, 1), load_rel=r(x.load_rel, 1), g36=r(x.g36_per_game, 2), rest=r(x.rest, 1),
                          eff=r(x.eff, 1), reabs=r(100 * x.rate, 0) if not pd.isna(x.rate) else None))
    coaches = []
    inv = {v: k for k, v in abbr.items()}
    for c, hist_c in cj["history"].items():
        vals = []
        for a, seasons in hist_c.items():
            for s in seasons:
                m = T[(T.team == inv[a]) & (T.season == s)]
                if len(m): vals.append((s, a, float(m.load_rel.iloc[0]), float(m.team_load.iloc[0]), float(m.rest.iloc[0]) if not pd.isna(m.rest.iloc[0]) else None))
        cur_team = next((k for k, v in cj["current"].items() if v == c), None)
        coaches.append(dict(coach=c, team=cur_team, seasons=len(vals), load_rel=r(np.mean([v[2] for v in vals]), 2) if vals else None,
                            load=r(np.mean([v[3] for v in vals]), 1) if vals else None,
                            rest=r(np.mean([v[4] for v in vals if v[4] is not None]), 1) if vals else None,
                            by_season=[[v[0], v[1], r(v[2], 1)] for v in vals]))
    json.dump(dict(teams=teams, coaches=coaches), open(os.path.join(PUB, "load_teams.json"), "w"), separators=(",", ":"))

    # ---- league trends and the test record
    trend = []
    A = D[D.min_rs >= LM.MIN_MIN]
    ext = json.load(open(os.path.join(OUT, "load_extra_tests.json")))
    tt = json.load(open(os.path.join(OUT, "load_team_tests.json")))
    first_run = json.load(open(os.path.join(OUT, "load_tests_partial.json")))
    rs = G[G.kind == "rs"]
    for s in range(LM.FIRST, LM.LAST + 1):
        g = rs[rs.season == s]; ng = TG[(TG.kind == "rs") & (TG.season == s)].gid.nunique()
        pace = float(g.poss.sum() / 10 / ng / 2) if ng else None
        trend.append(dict(season=f"{s}-{str(s + 1)[2:]}", pace=r(pace, 1), top5_mpg=r(lg.get(s), 1),
                          miles36=r(ext["miles36_by_season"].get(str(s)), 2), rest=r(tt["TM3"]["league_by_season"].get(str(s)), 1),
                          missed=r(A[A.season == s].missed.mean(), 1)))
    meta = dict(updated=datetime.date.today().isoformat(), projection_for=f"{CUR + 1}-{str(CUR + 2)[2:]}", built_from=f"{CUR}-{str(CUR + 1)[2:]}",
                backtest=BT, tests=dict(first_run=first_run, L1_L3=first_run, team=tt, extra=ext), trend=trend, league_mi36=lg_mi36,
                players=len(rows), min_minutes=LM.MIN_MIN, sample_min=LM.MIN_MIN)
    json.dump(meta, open(os.path.join(PUB, "load_meta.json"), "w"), indent=1, default=float)
    print("players", len(rows), "teams", len(teams), "coaches", len(coaches))


if __name__ == "__main__":
    main()
