"""Write the version 2 page data (data/load.json, data/load_meta.json, data/load_teams.json).

    python pipeline/load_v2_publish.py      # after load_publish.py (version 1) and load_v2.py

What goes in the season number follows the test record: only last season's injury games passed (the fuller season
model, V1, did not beat it). The game-level model (V2: recent minutes, back-to-backs) and the same-body-part
finding (V3) are shown as their own sections. Everything else stays a description.
"""
import os, re, json, datetime
import numpy as np, pandas as pd
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.metrics import roc_auc_score
import load_model as LM
from load_v2 import category, exact_age

OUT = LM.OUT; PUB = LM.PUB; INP = os.path.join(LM.HERE, "inputs")
REGIONS = [("head", r"head|concussion|nose|face|eye|jaw|orbital|dental|mouth"), ("neck", r"neck"), ("shoulder", r"shoulder|clavicle|rotator"),
           ("chest", r"chest|rib|pectoral|abdom|oblique|core|sternum"), ("back", r"back|lumbar|spine|sacr|tailbone|coccyx|sciatic|disc"),
           ("arm", r"elbow|arm|bicep|tricep|forearm"), ("hand", r"wrist|hand|finger|thumb"), ("hip", r"hip|groin|adductor|abductor|glute|pelvi"),
           ("thigh", r"hamstring|thigh|quad"), ("knee", r"knee|acl|mcl|meniscus|patell"), ("calf", r"calf|achilles|shin|tibia|fibula|leg"),
           ("ankle", r"ankle"), ("foot", r"foot|toe|heel|plantar")]
CUR = LM.LAST


def region(reason):
    r = str(reason).lower(); part = r.split(" - ", 1)[1] if " - " in r else r
    for g, pat in REGIONS:
        if re.search(pat, part): return g
    return "other"


def keep_v1(name):
    """version 1's output (written by load_publish.py) is kept in pipeline/out so this script can be rerun."""
    pub, keep = os.path.join(PUB, name), os.path.join(OUT, "v1_" + name)
    d = json.load(open(pub))
    if d.get("version") != 2 and not (name == "load.json" and "cols" not in d) and not (name == "load_teams.json" and os.path.exists(keep) and any("inj_games_2425" in t for t in d.get("teams", []))):
        json.dump(d, open(keep, "w"))
    return json.load(open(keep))


def r1(x, n=1):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def main():
    G, TG, names = LM.load_games()
    T2 = json.load(open(os.path.join(OUT, "v2_tests.json")))
    D = pd.read_csv(os.path.join(OUT, "v2_features.csv.gz"))
    PG = pd.read_csv(os.path.join(OUT, "v2_player_games.csv.gz"), parse_dates=["date"], low_memory=False)
    cj = json.load(open(os.path.join(LM.HERE, "coaches.json"))); abbr = {int(k): v for k, v in cj["teams"].items()}
    ros = json.load(open(os.path.join(INP, "rosters_2026.json")))
    team26 = {int(r[1]): abbr.get(int(r[0]), "?") for r in ros}; how26 = {int(r[1]): r[10] for r in ros}

    # ---- season number: last season's injury games only (what passed); backtest and ranges out of sample
    S = D[(D.min_rs >= LM.MIN_MIN) & D.season.between(2021, 2024) & D.y_inj.notna()].reset_index(drop=True)
    yb = (S.y_inj >= LM.BIG_MISS).astype(int).values
    x = S[["inj_l0"]].fillna(S.inj_l0.mean()).values
    mu, pb = np.zeros(len(S)), np.zeros(len(S))
    for s in sorted(S.season.unique()):
        tr = S.season.values != s
        pm = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(x[tr] / 82, S.y_inj.values[tr])
        lm = LogisticRegression().fit(x[tr] / 82, yb[tr])
        mu[~tr] = pm.predict(x[~tr] / 82); pb[~tr] = lm.predict_proba(x[~tr] / 82)[:, 1]
    edges = np.quantile(mu, np.linspace(0, 1, 6)); edges[0] = -1; edges[-1] = 1e9
    b = np.digitize(mu, edges) - 1
    ql = [float(np.quantile(S.y_inj[b == i], .1)) for i in range(5)]; qh = [float(np.quantile(S.y_inj[b == i], .9)) for i in range(5)]
    cover = float(np.mean([(ql[bi] <= y <= qh[bi]) for bi, y in zip(b, S.y_inj)]))
    q5 = pd.qcut(pb, 5, labels=False, duplicates="drop")
    BT = dict(n=len(S), seasons="outcomes 2022-23 to 2025-26 (2025-26 through Dec 21)", auc=float(roc_auc_score(yb, pb)), brier=LM.brier(pb, yb),
              rate20=float(yb.mean()), avg=float(S.y_inj.mean()), median=float(np.median(S.y_inj)), cover80=cover,
              calib=[dict(pred=float(pb[q5 == i].mean()), actual=float(yb[q5 == i].mean())) for i in sorted(set(q5))])
    pm = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(x / 82, S.y_inj.values); lm = LogisticRegression().fit(x / 82, yb)

    # ---- who to project: EVERY player on a 2026-27 roster
    #   basis "recent": injury games in his most recent season with 15+ covered games (games he played or was listed)
    #   basis "absent": he played no games last season; his injury games come straight from the reports he was listed on
    #   basis "rookie"/"no data": no NBA injury record to use; league average, flagged on the page
    cov = PG[PG.covered].groupby(["season", "pid"]).agg(inj=("inj_miss", "sum"), n=("covered", "size"))
    cov["inj82"] = cov.inj / cov.n * 82
    from load_v2 import load_reports
    RA = load_reports(names).dropna(subset=["pid", "date"]); RA["pid"] = RA.pid.astype(np.int64)
    RA["season"] = RA.date.dt.year - (RA.date.dt.month < 8)
    covered_dates = set(PG.loc[PG.covered, "date"].dt.normalize())
    tgr = TG[TG.kind == "rs"].copy(); tgr["abbr"] = tgr.team.map(abbr); tgr = tgr[tgr.date.dt.normalize().isin(covered_dates)]
    team_cov = tgr.groupby(["season", "abbr"]).gid.nunique()
    played = set(zip(PG.loc[PG.played == 1, "season"], PG.loc[PG.played == 1, "pid"]))
    def from_reports(p, s):
        x = RA[(RA.pid == p) & (RA.season == s)]
        if x.empty: return np.nan, 0
        t = x.abbr.mode().iloc[0]; n = int(team_cov.get((s, t), 0))
        k = x[x.out & x.cat.isin(["injury", "illness", "managed"])].date.nunique()
        return (min(k, n) / n * 82 if n >= 15 else np.nan), n
    def recent(p):
        for s in (CUR, CUR - 1, CUR - 2):
            if (s, p) in cov.index and cov.loc[(s, p), "n"] >= 15: return float(cov.loc[(s, p), "inj82"]), s, int(cov.loc[(s, p), "n"]), "recent"
            if (s, p) not in played:
                v, n = from_reports(p, s)
                if np.isfinite(v): return float(v), s, n, "absent"
        return np.nan, None, 0, None
    V1 = exact_age(pd.read_csv(os.path.join(OUT, "load_features_plus.csv.gz")))
    lastv = V1.sort_values("season").groupby("pid").tail(1).set_index("pid")
    BI = pd.read_csv(os.path.join(OUT, "bio_v2.csv")).set_index("pid")
    rp = {int(r[1]): r[4] for r in ros}
    def pos_from(p):
        if p in lastv.index and isinstance(lastv.loc[p, "pos"], str): return lastv.loc[p, "pos"]
        q = str(rp.get(p, "F"))
        return "big" if q.startswith("C") or q == "F-C" else "guard" if q.startswith("G") and q != "G-F" else "wing"
    rows0 = []
    for p in team26:
        v, s, n, basis = recent(p)
        bd = pd.to_datetime(BI.bday.get(p), errors="coerce") if p in BI.index else pd.NaT
        age = float(np.floor((pd.Timestamp(f"{CUR + 2}-02-01") - bd).days / 365.25)) if pd.notna(bd) else (float(lastv.loc[p, "age_next"]) + (CUR - lastv.loc[p, "season"]) if p in lastv.index else np.nan)
        if basis is None:
            basis = "rookie" if "Draft" in str(how26.get(p, "")) and p not in lastv.index else ("rookie" if p not in lastv.index else "no data")
        rows0.append(dict(pid=p, inj_recent=v, inj_season=s, inj_cov=n, basis=basis, age_next=age, pos=pos_from(p),
                          min_last=float(lastv.loc[p, "min_rs"]) if p in lastv.index else 0.0, last_season=int(lastv.loc[p, "season"]) if p in lastv.index else None))
    last = pd.DataFrame(rows0)
    have = last.inj_recent.notna()
    xp = last.loc[have, ["inj_recent"]].values / 82
    last.loc[have, "exp"] = pm.predict(xp); last.loc[have, "p20"] = lm.predict_proba(xp)[:, 1]
    last.loc[~have, "exp"] = float(S.y_inj.mean()); last.loc[~have, "p20"] = float(yb.mean())
    bi = (np.digitize(last.exp, edges) - 1).clip(0, 4)
    last["lo"] = [ql[i] for i in bi]; last["hi"] = [qh[i] for i in bi]
    last.loc[~have, "lo"] = float(np.quantile(S.y_inj, .1)); last.loc[~have, "hi"] = float(np.quantile(S.y_inj, .9))

    # ---- body map: injury games by region, last three seasons, and the latest injury
    inj = PG[PG.inj_miss & PG.cat.isin(["injury", "managed"]) & (PG.season >= CUR - 2)].copy()
    from load_v2 import load_reports
    RR = load_reports(names).dropna(subset=["pid"]); RR["pid"] = RR.pid.astype(np.int64); RR["region"] = RR.reason.map(region)
    RR = RR.drop_duplicates(["date", "pid"]).set_index(["date", "pid"])
    inj["region"] = RR.region.reindex(pd.MultiIndex.from_arrays([inj.date, inj.pid])).values
    inj["reason"] = RR.reason.reindex(pd.MultiIndex.from_arrays([inj.date, inj.pid])).values
    ab = RA[RA.out & RA.cat.isin(["injury", "managed"]) & (RA.season >= CUR - 2) & ~pd.Series(list(zip(RA.season, RA.pid)), index=RA.index).isin(played)].drop_duplicates(["date", "pid"]).copy()
    ab["region"] = ab.reason.map(region)
    inj = pd.concat([inj, ab[["pid", "date", "region", "reason"]]], ignore_index=True)
    bm = inj.groupby(["pid", "region"]).size().unstack(fill_value=0)
    latest = inj.sort_values("date").groupby("pid").tail(1).set_index("pid")

    # ---- descriptive lines from version 1 (load, miles, college, coach)
    v1 = keep_v1("load.json"); C1 = v1["cols"]; V1rows = {r[C1.index("pid")]: dict(zip(C1, r)) for r in v1["rows"]}
    hist = cov.inj82
    # ---- the season being played now (rebuilt daily): games played and team games since his first game for that team
    sofar, live_season = {}, None
    lp = os.path.join(OUT, f"games_{CUR + 1}.csv.gz")
    if os.path.exists(lp):
        L = pd.read_csv(lp, parse_dates=["date"]); L = L[L.kind == "rs"]
        if len(L):
            live_season = f"{CUR + 1}-{str(CUR + 2)[2:]}"
            ltg = pd.read_csv(os.path.join(OUT, f"teamgames_{CUR + 1}.csv.gz"), parse_dates=["date"]); ltg = ltg[ltg.kind == "rs"]
            for pid, x in L.groupby("pid"):
                t = x.sort_values("date").team.iloc[-1]; first = x[x.team == t].date.min()
                n = int(ltg[(ltg.team == t) & (ltg.date >= first)].gid.nunique())
                sofar[int(pid)] = dict(gp=int(len(x[x.team == t])), team_games=n)
            team_n = ltg.groupby("team").gid.nunique().to_dict()
            inv = {v: k for k, v in abbr.items()}
            for p_ in team26:
                if p_ not in sofar:
                    sofar[p_] = dict(gp=0, team_games=int(team_n.get(inv.get(team26[p_]), 0)))
    rows = []
    for _, x in last.sort_values("exp", ascending=False).iterrows():
        p = int(x.pid); d1 = V1rows.get(p, {})
        h = [r1(hist.get((s, p)), 0) if (s, p) in hist.index else (r1(from_reports(p, s)[0], 0) if (s, p) not in played and s >= 2021 else None) for s in (CUR - 4, CUR - 3, CUR - 2, CUR - 1, CUR)]
        body = {k: int(v) for k, v in bm.loc[p].items() if v > 0} if p in bm.index else {}
        lt = latest.loc[p] if p in latest.index else None
        rows.append(dict(name=names.get(p, str(p)), team=team26.get(p, "?"), moved=bool(re.search(r"(Traded|Signed|Claimed).*\d{2}/\d{2}/26$", how26.get(p, "") or "")),
                         how=how26.get(p, ""), age=r1(x.age_next, 0), pos=x.pos, exp=r1(x.exp), lo=r1(x["lo"], 0), hi=r1(x["hi"], 0), p20=r1(100 * x.p20, 0),
                         inj_recent=r1(x.inj_recent, 0) if pd.notna(x.inj_recent) else None, inj_season=f"{int(x.inj_season)}-{str(int(x.inj_season) + 1)[2:]}" if pd.notna(x.inj_season) else None, inj_cov=int(x.inj_cov),
                         basis=x.basis, min_last=r1(x.min_last, 0), last_season=None if pd.isna(x.last_season) else f"{int(x.last_season)}-{str(int(x.last_season) + 1)[2:]}",
                         hist=h, body=body, last_injury=None if lt is None else dict(date=str(lt.date.date()), region=lt.region if isinstance(lt.region, str) else "other", reason=str(lt.reason) if isinstance(lt.reason, str) else "Injury (reason not matched)"),
                         mpg=d1.get("mpg"), gp=d1.get("gp"), pct38=d1.get("pct38"), b2b=d1.get("pct_b2b30"), mi36=d1.get("mi36"), speed=d1.get("speed"),
                         career_k=d1.get("career_k"), col_years=d1.get("col_years"), col_min=d1.get("col_min"), col_missed_pct=d1.get("col_missed_pct"),
                         no_college=d1.get("no_college"), height=d1.get("height"), sofar=sofar.get(p), pid=p))
    json.dump(dict(rows=rows), open(os.path.join(PUB, "load.json"), "w"), separators=(",", ":"), allow_nan=False)

    # ---- teams: 2026-27 coach, coach load (v1), injury games lost per player last season (labeled)
    t1 = keep_v1("load_teams.json")
    lost = PG[PG.covered & (PG.season == CUR - 1)].groupby("abbr").agg(inj=("inj_miss", "sum"), n=("covered", "size"))
    for t in t1["teams"]:
        if t["team"] in lost.index: t["inj_games_2425"] = int(lost.loc[t["team"], "inj"])
    json.dump(t1, open(os.path.join(PUB, "load_teams.json"), "w"), separators=(",", ":"))

    # ---- meta
    m1 = keep_v1("load_meta.json")
    HZ = pd.read_csv(os.path.join(OUT, "v2_hazard.csv.gz"), usecols=["rest_next", "y", "y_strict"])
    g2 = HZ.groupby(HZ.rest_next == 0)[["y", "y_strict"]].mean()
    b2b = dict(second_night=float(g2.loc[True, "y"]), other=float(g2.loc[False, "y"]), second_night_strict=float(g2.loc[True, "y_strict"]), other_strict=float(g2.loc[False, "y_strict"]))
    allinj = PG[PG.inj_miss].copy()
    allinj["region"] = RR.region.reindex(pd.MultiIndex.from_arrays([allinj.date, allinj.pid])).values
    meta = dict(version=2, updated=datetime.date.today().isoformat(), projection_for=f"{CUR + 1}-{str(CUR + 2)[2:]}",
                backtest=BT, v2=T2, v1_tests=m1["tests"], v1_backtest=m1["backtest"], trend=m1["trend"], league_mi36=m1.get("league_mi36"),
                region_games=allinj.region.value_counts().to_dict(), players=len(rows), report_last=T2["log"]["last_date"],
                b2b=b2b, live_season=live_season, playoffs=json.load(open(os.path.join(OUT, "playoffs_tests.json"))) if os.path.exists(os.path.join(OUT, "playoffs_tests.json")) else None,
                basis_counts=pd.Series([r["basis"] for r in rows]).value_counts().to_dict())
    json.dump(meta, open(os.path.join(PUB, "load_meta.json"), "w"), indent=1, default=float)
    print("players", len(rows), "backtest", {k: v for k, v in BT.items() if k != "calib"})


if __name__ == "__main__":
    main()
