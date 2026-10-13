"""Load & Durability, version 2: official NBA injury reports as labels, game-level load tests, body parts.

    python pipeline/load_v2.py

Inputs: pipeline/inputs/injury_reports.csv (official reports, parsed), player-game tables from load_games.py,
pipeline/out/travel.csv, contact.csv, bio_v2.csv (load_v2_features.py), load_features_plus.csv.gz (version 1).
Tests: V0-V7 in LOAD_PREREG.txt. Writes pipeline/out/v2_*.csv / .json.
"""
import os, re, json, unicodedata
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
import load_model as LM

OUT = LM.OUT; INP = os.path.join(LM.HERE, "inputs")
TEAMNAME = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Cleveland Cavaliers": "CLE", "New Orleans Pelicans": "NOP", "Chicago Bulls": "CHI",
            "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "LA Clippers": "LAC",
            "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
            "Brooklyn Nets": "BKN", "New York Knicks": "NYK", "Orlando Magic": "ORL", "Indiana Pacers": "IND", "Philadelphia 76ers": "PHI",
            "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Oklahoma City Thunder": "OKC",
            "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Memphis Grizzlies": "MEM", "Washington Wizards": "WAS", "Detroit Pistons": "DET", "Charlotte Hornets": "CHA"}
GROUPS = [("head", r"head|concussion|nose|face|eye|jaw|orbital|neck"), ("knee", r"knee|acl|mcl|meniscus|patell"),
          ("ankle_foot", r"ankle|foot|toe|heel|achilles|plantar"), ("muscle", r"hamstring|calf|thigh|quad|groin|adductor|hip|glute|abductor"),
          ("back", r"back|lumbar|spine|core|abdom|oblique|rib"), ("upper", r"shoulder|arm|elbow|wrist|hand|finger|thumb|chest|pectoral|clavicle"),
          ("illness", r"illness|sick|flu|virus|covid|gastro|respiratory|migraine")]


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s).strip(); s = re.sub(r" (jr|sr|ii|iii|iv)\b", "", s)
    return s.replace(" ", "")


def flip(n):
    if "," not in n: return n
    last, first = n.split(",", 1); return first.strip() + " " + last.strip()


def category(reason, status):
    r = reason.lower()
    if r.startswith("injury/illness") or "concussion" in r or "reconditioning" in r:
        part = r.split(" - ", 1)[1] if " - " in r else r
        managed = "management" in part or "load" in part.split(";")[-1]
        if re.search(GROUPS[-1][1], part) and not re.search(r"ankle|knee|foot|hamstring|back|shoulder", part): return "illness", "illness"
        cat = "managed" if managed else "injury"
        for g, pat in GROUPS[:-1]:
            if re.search(pat, part): return cat, g
        return cat, "other"
    if "protocol" in r and "health" in r: return "protocols", ""
    if "g league" in r: return "gleague", ""
    if r.startswith("rest") or "rest" == r.strip(): return "rest", ""
    if "personal" in r or "not with team" in r: return "personal", ""
    if "suspension" in r: return "suspension", ""
    if "ineligible" in r: return "ineligible", ""
    return "other", ""


def load_reports(names):
    R = pd.read_csv(os.path.join(INP, "injury_reports.csv.gz"))
    R = R[R.player.notna() & (R.player.str.len() > 0)].copy()
    R["date"] = pd.to_datetime(R.date, format="%m/%d/%Y", errors="coerce")
    R["abbr"] = R.team.map(TEAMNAME)
    byname = {}
    for pid, n in names.items(): byname.setdefault(norm(n), []).append(pid)
    R["key"] = R.player.map(lambda x: norm(flip(x)))
    R["pid"] = R.key.map(lambda k: byname.get(k, [None])[-1]).astype(object)
    alias = {"poeltljakob": "jakobpoltl"}
    R.loc[R.pid.isna(), "pid"] = R.loc[R.pid.isna(), "key"].map(lambda k: byname.get(norm(alias.get(k, k)), [None])[-1])
    cats = [category(str(r), str(s)) for r, s in zip(R.reason, R.status)]
    R["cat"] = [c for c, _ in cats]; R["part"] = [p for _, p in cats]
    R["out"] = R.status.isin(["Out", "Doubtful"])
    return R


def exact_age(D):
    """age_next = age on Feb 1 of the next season, from NBA.com birth dates (bio_v2.csv); keeps the old estimate only
    where NBA.com has no birth date."""
    BI = pd.read_csv(os.path.join(OUT, "bio_v2.csv"), usecols=["pid", "bday"])
    bd = dict(zip(BI.pid, pd.to_datetime(BI.bday, errors="coerce")))
    feb = pd.to_datetime((D.season + 2).astype(str) + "-02-01")
    b = D.pid.map(bd)
    exact = np.floor((feb - b).dt.days / 365.25)
    D["age_source"] = np.where(exact.notna(), "nba.com", "estimate")
    D["age_next"] = exact.fillna(D.age_next)
    return D


def team_abbr(cj):
    return {int(k): v for k, v in cj["teams"].items()}


def main():
    G, TG, names = LM.load_games()
    cj = json.load(open(os.path.join(LM.HERE, "coaches.json"))); abbr = team_abbr(cj)
    R = load_reports(names)
    log = dict(report_rows=len(R), matched_share=float(R.pid.notna().mean()), report_dates=int(R.date.nunique()),
               last_date=str(R.date.max().date()), unmatched_examples=R[R.pid.isna()].player.value_counts().head(8).to_dict())
    print(log, flush=True)
    # fallback: an unmatched name whose last name is unique among the players who played for that team that season
    gg = G[G.kind == "rs"][["season", "team", "pid"]].drop_duplicates(); gg["abbr"] = gg.team.map(abbr)
    gg["last"] = gg.pid.map(lambda p: norm(str(names.get(p, "")).split(" ")[-1] if names.get(p) else ""))
    R["season"] = R.date.dt.year - (R.date.dt.month < 8)
    R["last"] = R.player.map(lambda x: norm(x.split(",")[0]))
    un = R.pid.isna()
    look = gg.groupby(["season", "abbr", "last"]).pid.agg(lambda x: x.iloc[0] if len(set(x)) == 1 else None).to_dict()
    R["pid"] = R.pid.astype(object)
    R.loc[un, "pid"] = pd.Series([look.get((s, a, l)) for s, a, l in zip(R.season[un], R.abbr[un], R["last"][un])], index=R.index[un], dtype=object)
    log["matched_share_after_fallback"] = float(R.pid.notna().mean())
    R = R.dropna(subset=["pid", "abbr", "date"]); R["pid"] = R.pid.astype(np.int64)
    rep = R.drop_duplicates(["date", "abbr", "pid"]).set_index(["date", "abbr", "pid"])
    covered = set(R.date.dt.normalize())

    # ---- every regular-season team game for every player on the team: played or missed, and why
    g = G[G.kind == "rs"].copy(); tg = TG[TG.kind == "rs"].sort_values(["season", "team", "date", "gid"]).copy()
    tg["k"] = tg.groupby(["season", "team"]).cumcount()
    g = g.merge(tg[["gid", "team", "k"]], on=["gid", "team"], how="left")
    g["mins"] = g.secs / 60
    rows = []
    for (s, p), x in g.sort_values("date").groupby(["season", "pid"], sort=False):
        teams = list(dict.fromkeys(x.team))
        for i, t in enumerate(teams):
            xt = x[x.team == t]; ks = dict(zip(xt.k, xt.mins))
            sched = tg[(tg.season == s) & (tg.team == t)]
            lo = 0 if i == 0 else min(ks); hi = max(ks) if i < len(teams) - 1 else sched.k.max()
            for r in sched[(sched.k >= lo) & (sched.k <= hi)].itertuples():
                rows.append((s, p, t, r.k, r.gid, r.date, int(r.k in ks), ks.get(r.k, 0.0)))
    PG = pd.DataFrame(rows, columns=["season", "pid", "team", "k", "gid", "date", "played", "mins"])
    PG["abbr"] = PG.team.map(abbr)
    PG["covered"] = PG.date.dt.normalize().isin(covered)
    idx = pd.MultiIndex.from_arrays([PG.date.dt.normalize(), PG.abbr, PG.pid])
    PG["status"] = rep.status.reindex(idx).values; PG["cat"] = rep.cat.reindex(idx).values
    PG["part"] = rep.part.reindex(idx).values; PG["listed_out"] = rep.out.reindex(idx).fillna(False).values.astype(bool)
    PG["inj_miss"] = (PG.played == 0) & PG.listed_out & PG.cat.isin(["injury", "illness", "managed"])
    PG.to_csv(os.path.join(OUT, "v2_player_games.csv.gz"), index=False)
    print("player-team-games", len(PG), "covered", int(PG.covered.sum()), flush=True)

    # ---- V0: how good was version 1's "runs of 3+" outcome?
    C = PG[PG.covered & (PG.season >= 2021)].sort_values(["pid", "season", "team", "k"]).copy()
    C["run"] = (C.played != C.groupby(["pid", "season", "team"]).played.shift()).cumsum()
    rl = C.groupby("run").played.transform("size")
    miss = C[C.played == 0]
    v1miss = miss[rl[miss.index] >= 3]
    V0 = dict(v1_missed_games=int(len(v1miss)), share_injury=float(v1miss.inj_miss.mean()),
              v1_mix=v1miss.cat.fillna("not listed").value_counts(normalize=True).round(3).to_dict(),
              injury_games=int(miss.inj_miss.sum()), injury_games_in_short_runs=float((miss.inj_miss & (rl[miss.index] < 3)).sum() / max(miss.inj_miss.sum(), 1)))
    print("V0", V0, flush=True)

    # ---- season outcome: injury games missed per 82 covered games
    S0 = PG[PG.covered].groupby(["season", "pid"]).agg(inj=("inj_miss", "sum"), cov=("covered", "size")).reset_index()
    S0 = S0[S0["cov"] >= 15]
    S0["inj82"] = S0.inj / S0["cov"] * 82
    parts = PG[PG.inj_miss].groupby(["season", "pid", "part"]).size().unstack(fill_value=0)
    parts.to_csv(os.path.join(OUT, "v2_parts.csv"))

    # ---- features: version-1 table (+ exact ages), contact, size
    D = pd.read_csv(os.path.join(OUT, "load_features_plus.csv.gz"))
    D["tload"] = D.next_team_load.fillna(D.team_load)
    CT = pd.read_csv(os.path.join(OUT, "contact.csv"))
    D = D.merge(CT, on=["season", "pid"], how="left")
    for c in ("rim_fga", "fta", "fouls_drawn", "charges_taken"): D[c + "36"] = D[c] / D.min_rs.clip(lower=1) * 36
    BI = pd.read_csv(os.path.join(OUT, "bio_v2.csv"))
    D = D.merge(BI[["pid", "height_in", "weight_lb", "bmi"]], on="pid", how="left")
    D = exact_age(D)
    inj = S0.set_index(["season", "pid"]).inj82
    for lag in (0, 1, 2):
        D[f"inj_l{lag}"] = [inj.get((s - lag, p), np.nan) for s, p in zip(D.season, D.pid)]
    # before 2021-22 there are no labels: use version 1's missed games for the history inputs
    for lag, v1 in ((0, "missed"), (1, "missed_l1"), (2, "missed_l2")):
        D[f"inj_l{lag}"] = D[f"inj_l{lag}"].fillna(D[v1])
    lowmusc = parts.get("muscle", pd.Series(dtype=float))
    D["muscle_2y"] = [lowmusc.get((s, p), 0) + lowmusc.get((s - 1, p), 0) for s, p in zip(D.season, D.pid)]
    D["y_inj"] = [inj.get((s + 1, p), np.nan) for s, p in zip(D.season, D.pid)]
    D.to_csv(os.path.join(OUT, "v2_features.csv.gz"), index=False)

    LM.FEATS["IH"] = ["inj_l0", "inj_l1", "inj_l2"]
    LM.FEATS["IB"] = ["inj_l0"]
    LM.FEATS["CX"] = ["rim_fga36", "fta36", "fouls_drawn36", "charges_taken36"]
    LM.FEATS["SZ"] = ["height_in", "weight_lb", "bmi"]
    LM.FEATS["MU"] = ["muscle_2y"]
    S = D[(D.min_rs >= LM.MIN_MIN) & D.season.between(2021, 2024) & D.y_inj.notna()].reset_index(drop=True)
    S["y_missed"] = S.y_inj                                      # the fit helpers read y_missed
    yb = (S.y_inj >= LM.BIG_MISS).astype(int).values
    def run(blocks):
        mu, pb = LM.loso(S, blocks); return mu, pb, LM.brier(pb, yb)
    _, pb0, b0 = run(["IB"])
    _, pb1, b1 = run(["IH", "A", "P"])
    V1 = dict(n=len(S), rate20=float(yb.mean()), brier_base=b0, brier_model=b1, gain=(b0 - b1) / b0, auc_base=float(roc_auc_score(yb, pb0)),
              auc_model=float(roc_auc_score(yb, pb1)), pass_=(b0 - b1) / b0 >= 0.02)
    _, _, bW = run(["IH", "A", "P", "M", "W"])
    _, _, bC = run(["IH", "A", "P", "CX"]); _, _, bZ = run(["IH", "A", "P", "SZ"]); _, _, bM = run(["IH", "A", "P", "MU"])
    V1["workload_gain"] = (b1 - bW) / b1
    V6a = (b1 - bC) / b1; V7a = (b1 - bZ) / b1; V3b = (b1 - bM) / b1
    # V4 model form
    Z, st = LM.design(S, ["IH", "A", "P"]); pbg = np.zeros(len(S))
    for s in sorted(S.season.unique()):
        tr = S.season.values != s
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40).fit(Z[tr], yb[tr])
        pbg[~tr] = m.predict_proba(Z[~tr])[:, 1]
    bg = LM.brier(pbg, yb)
    V4 = dict(brier_logistic=b1, brier_trees=bg, gain=(b1 - bg) / b1, pass_=(b1 - bg) / b1 >= 0.01)
    print("V1", V1, "V4", V4, "V6a", V6a, "V7a", V7a, "V3b", V3b, flush=True)

    # ---- V2 game level: does load in the last week predict an injury starting at the next game?
    TR = pd.read_csv(os.path.join(OUT, "travel.csv"), parse_dates=["date"])
    TR = TR[TR.po == 0]
    H = PG[(PG.season >= 2021)].sort_values(["pid", "season", "team", "k"]).copy()
    H["next_played"] = H.groupby(["pid", "season", "team"]).played.shift(-1)
    H["next_inj"] = H.groupby(["pid", "season", "team"]).inj_miss.shift(-1)
    H["next_cat"] = H.groupby(["pid", "season", "team"]).cat.shift(-1)
    H["next_cov"] = H.groupby(["pid", "season", "team"]).covered.shift(-1)
    H["next_date"] = H.groupby(["pid", "season", "team"]).date.shift(-1)
    H["prev_played"] = H.groupby(["pid", "season", "team"]).played.shift(1).fillna(1)
    # load windows by calendar date (minutes he played)
    feats = []
    for (p, s), x in H.groupby(["pid", "season"], sort=False):
        d = x.date.values.astype("datetime64[D]").astype(np.int64); m = (x.mins * x.played).values
        c7 = np.array([m[(d > di - 7) & (d < di)].sum() for di in d]); c28 = np.array([m[(d > di - 28) & (d < di)].sum() for di in d])
        n7 = np.array([((d > di - 7) & (d < di) & (x.played.values == 1)).sum() for di in d])
        cum = np.cumsum(m) - m
        prevm = np.r_[0, m[:-1]]
        feats.append(pd.DataFrame(dict(i=x.index, min7=c7, min28=c28, g7=n7, cum=cum, prev_mins=prevm)))
    F = pd.concat(feats).set_index("i")
    H = H.join(F)
    H["acwr"] = H.min7 / (H.min28 / 4).clip(lower=20)
    H["b2b"] = ((H.date - H.groupby(["pid", "season", "team"]).date.shift(1)).dt.days == 1).astype(int)
    H["rest_next"] = (H.next_date - H.date).dt.days - 1
    # last injury return
    H["games_since_inj"] = 0
    for (p, s, t), x in H.groupby(["pid", "season", "team"], sort=False):
        last = -99; out = []
        for k, im in zip(x.k, x.inj_miss):
            if im: last = k
            out.append(min(k - last, 82) if last > -99 else 82)
        H.loc[x.index, "games_since_inj"] = out
    TRk = TR.copy(); TRk["abbr"] = TRk.abbr.replace({"NOH": "NOP"})
    TRk = TRk.sort_values(["abbr", "date"])
    trav = []
    for a, x in TRk.groupby("abbr"):
        d = x.date.values.astype("datetime64[D]").astype(np.int64)
        fm = x.fly_miles.values; tz = x.tz.values
        trav.append(pd.DataFrame(dict(abbr=a, date=x.date.values, fly7=[fm[(d > di - 7) & (d <= di)].sum() for di in d],
                                      tz7=[tz[(d > di - 7) & (d <= di)].sum() for di in d], alt=x.altitude.values)))
    TV = pd.concat(trav)
    H = H.merge(TV, on=["abbr", "date"], how="left")
    # base rate: injury runs per game in the prior two seasons (version-1 run starts before labels)
    D2 = D.set_index(["season", "pid"])
    H["prior_rate"] = [np.nanmean([D2.inj_l1.get((s, p), np.nan), D2.inj_l2.get((s, p), np.nan)]) for s, p in zip(H.season, H.pid)]
    H = H.merge(D[["season", "pid", "age_next", "pos", "rim_fga36", "fta36", "fouls_drawn36", "charges_taken36", "height_in", "weight_lb", "bmi"]],
                on=["season", "pid"], how="left")
    H["age"] = H.age_next - 1
    U = H[(H.played == 1) & (H.prev_played == 1) & (H.next_cov == True) & H.next_played.notna()].copy()
    U["y"] = ((U.next_inj == True) & U.next_cat.isin(["injury", "managed"])).astype(int)          # as pre-registered (Injury/Illness, not illness)
    U["y_strict"] = ((U.next_inj == True) & (U.next_cat == "injury")).astype(int)                 # sensitivity: planned "injury management" left out
    U = U[U.mins > 0]
    print("hazard rows", len(U), "events", int(U.y.sum()), flush=True)
    U.to_csv(os.path.join(OUT, "v2_hazard.csv.gz"), index=False)
    base = ["prior_rate", "age", "big", "guard", "games_since_inj"]
    U["big"] = (U.pos == "big").astype(float); U["guard"] = (U.pos == "guard").astype(float)
    load = ["min7", "min28", "acwr", "prev_mins", "b2b", "g7", "cum"]
    travel = ["fly7", "tz7", "alt", "rest_next"]
    contact = ["rim_fga36", "fta36", "fouls_drawn36", "charges_taken36"]
    size = ["height_in", "weight_lb", "bmi"]
    def haz(cols):
        p = np.zeros(len(U))
        for s in sorted(U.season.unique()):
            tr = U.season.values != s
            X = U[cols].astype(float); mu = X[tr].mean(); sd = X[tr].std().replace(0, 1)
            Zx = ((X.fillna(mu) - mu) / sd).clip(-4, 4)
            m = LogisticRegression(C=1.0, max_iter=2000).fit(Zx[tr], U.y.values[tr]); p[~tr] = m.predict_proba(Zx[~tr])[:, 1]
        return p, log_loss(U.y, p)
    p0, l0 = haz(base); p1, l1 = haz(base + load); _, l5 = haz(base + travel); _, l6 = haz(base + contact); _, l7 = haz(base + size)
    hi = U.acwr >= U.acwr.quantile(0.9)
    rr = float(U.y[hi].mean() / U.y[~hi].mean())
    b2b_rr = float(U.y[U.rest_next == 0].mean() / U.y[U.rest_next > 0].mean())
    V2 = dict(rows=len(U), events=int(U.y.sum()), rate=float(U.y.mean()), logloss_base=l0, logloss_load=l1, gain=(l0 - l1) / l0, pass_=(l0 - l1) / l0 >= 0.005,
              auc_base=float(roc_auc_score(U.y, p0)), auc_load=float(roc_auc_score(U.y, p1)), rr_top_acwr=rr, rr_next_b2b=b2b_rr,
              by_acwr_tenth=U.groupby(pd.qcut(U.acwr, 10, labels=False, duplicates="drop")).y.mean().round(4).tolist())
    V5 = dict(gain=(l0 - l5) / l0, pass_=(l0 - l5) / l0 >= 0.005)
    ysave = U.y.copy(); U["y"] = U.y_strict
    q0, m0 = haz(base); q1, m1 = haz(base + load); _, m5 = haz(base + travel)
    hi2 = U.acwr >= U.acwr.quantile(0.9)
    V2["strict"] = dict(events=int(U.y.sum()), gain_load=(m0 - m1) / m0, gain_travel=(m0 - m5) / m0, rr_next_b2b=float(U.y[U.rest_next == 0].mean() / U.y[U.rest_next > 0].mean()),
                        rr_top_acwr=float(U.y[hi2].mean() / U.y[~hi2].mean()))
    U["y"] = ysave
    # which load inputs carry it (fit on all seasons, standardized coefficients), for the page
    X = U[base + load].astype(float); X = (X.fillna(X.mean()) - X.mean()) / X.std().replace(0, 1)
    cm = LogisticRegression(C=1.0, max_iter=2000).fit(X.clip(-4, 4), U.y)
    V2["coef"] = dict(zip(base + load, np.round(cm.coef_[0], 3).tolist()))
    V2["by_prev_mins"] = U.groupby(pd.cut(U.prev_mins, [-1, 10, 20, 30, 36, 60])).y.mean().round(4).tolist()
    V2["by_min7"] = U.groupby(pd.qcut(U.min7, 5, labels=False, duplicates="drop")).y.mean().round(4).tolist()
    V2["by_g7"] = U.groupby(U.g7.clip(upper=4)).y.mean().round(4).to_dict()
    V6 = dict(v1_gain=V6a, v2_gain=(l0 - l6) / l0, pass_=V6a >= 0.01 or (l0 - l6) / l0 >= 0.005)
    V7 = dict(v1_gain=V7a, v2_gain=(l0 - l7) / l0, pass_=V7a >= 0.01 or (l0 - l7) / l0 >= 0.005)
    print("V2", V2, "V5", V5, "V6", V6, "V7", V7, flush=True)

    # ---- V3 same body part again
    ev = PG[PG.inj_miss & (PG.cat == "injury")].sort_values(["pid", "season", "team", "k"]).copy()
    ev["new"] = (ev.k != ev.groupby(["pid", "season", "team"]).k.shift() + 1) | (ev.part != ev.groupby(["pid", "season", "team"]).part.shift())
    ev["spell"] = ev.new.cumsum()
    sp = ev.groupby("spell").agg(pid=("pid", "first"), season=("season", "first"), team=("team", "first"), k0=("k", "min"), k1=("k", "max"), part=("part", "first"), n=("k", "size")).reset_index()
    share = sp.part.value_counts(normalize=True)
    pairs = []
    for (p, s, t), x in sp.sort_values("k0").groupby(["pid", "season", "team"]):
        x = x.reset_index(drop=True)
        for i in range(len(x) - 1):
            if x.n[i] >= 3 and x.k0[i + 1] - x.k1[i] <= 31:
                pairs.append((x.part[i], x.part[i + 1]))
    P2 = pd.DataFrame(pairs, columns=["a", "b"])
    obs = float((P2.a == P2.b).mean()) if len(P2) else np.nan
    exp = float(P2.a.map(share).mean()) if len(P2) else np.nan
    rng = np.random.default_rng(0); boots = []
    for _ in range(2000):
        bb = P2.sample(len(P2), replace=True, random_state=int(rng.integers(1e9)))
        boots.append((bb.a == bb.b).mean() / bb.a.map(share).mean())
    lo, hi_ = np.percentile(boots, [2.5, 97.5])
    by = P2.assign(same=P2.a == P2.b).groupby("a").agg(n=("same", "size"), same=("same", "mean")).reset_index()
    by["expected"] = by.a.map(share)
    V3 = dict(pairs=len(P2), same_rate=obs, expected=exp, rr=obs / exp, ci=[float(lo), float(hi_)], pass_=obs / exp >= 2 and lo > 1,
              by_part=by.round(3).to_dict("records"), part_mix=share.round(3).to_dict(), v1_muscle_gain=V3b, v1_muscle_pass=V3b >= 0.01)
    print("V3", {k: v for k, v in V3.items() if k != "by_part"}, flush=True)
    json.dump(dict(log=log, V0=V0, V1=V1, V2=V2, V3=V3, V4=V4, V5=V5, V6=V6, V7=V7), open(os.path.join(OUT, "v2_tests.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
