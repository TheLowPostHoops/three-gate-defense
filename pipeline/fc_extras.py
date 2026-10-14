"""Tests S1-S4 (FORECAST_PREREG.txt, additions of 2026-10-14): schedule, projected health, surprise factor, late season.

    python3 fc_extras.py          # writes out/fc_extras.json
"""
import os, json, math
import numpy as np, pandas as pd
from scipy.stats import norm
import fc_project as FP

OUT = FP.OUT; HERE = FP.HERE
TRAIN, TEST = list(range(2014, 2020)), list(range(2020, 2026))
ARENA = {  # lat, lon, UTC offset (standard time), altitude flag
    "ATL": (33.757, -84.396, -5), "BOS": (42.366, -71.062, -5), "BKN": (40.683, -73.976, -5), "CHA": (35.225, -80.839, -5),
    "CHI": (41.881, -87.674, -6), "CLE": (41.496, -81.688, -5), "DAL": (32.790, -96.810, -6), "DEN": (39.749, -105.008, -7),
    "DET": (42.341, -83.055, -5), "GSW": (37.768, -122.388, -8), "HOU": (29.751, -95.362, -6), "IND": (39.764, -86.156, -5),
    "LAC": (33.945, -118.341, -8), "LAL": (34.043, -118.267, -8), "MEM": (35.138, -90.051, -6), "MIA": (25.781, -80.188, -5),
    "MIL": (43.045, -87.917, -6), "MIN": (44.980, -93.276, -6), "NOP": (29.949, -90.082, -6), "NYK": (40.751, -73.993, -5),
    "OKC": (35.463, -97.515, -6), "ORL": (28.539, -81.384, -5), "PHI": (39.901, -75.172, -5), "PHX": (33.446, -112.071, -7),
    "POR": (45.532, -122.667, -8), "SAC": (38.580, -121.500, -8), "SAS": (29.427, -98.438, -6), "TOR": (43.643, -79.379, -5),
    "UTA": (40.768, -111.901, -7), "WAS": (38.898, -77.021, -5)}
ALT = {"DEN", "UTA"}
LEAGUE_AVAIL = 0.86


def abbr():
    return {int(k): v for k, v in json.load(open(os.path.join(HERE, "coaches.json")))["teams"].items()}


def miles(a, b):
    (la1, lo1, _), (la2, lo2, _) = ARENA[a], ARENA[b]
    p1, p2 = math.radians(la1), math.radians(la2); dl = math.radians(lo2 - lo1); dp = p2 - p1
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(h))


def schedule_features(G):
    """G: games with date, home, away (tricodes). Adds per-game schedule features (home minus away where paired)."""
    G = G.sort_values(["date", "gid"]).reset_index(drop=True)
    last = {}; hist = {}
    rows = []
    for x in G.itertuples():
        venue = x.home
        f = {}
        for side, t in (("h", x.home), ("a", x.away)):
            prev = last.get(t)
            if prev is None: rest, mi, tz = 3, 0.0, 0
            else:
                rest = min(3, (x.date - prev[0]).days - 1)
                mi = miles(prev[1], venue); tz = abs(ARENA[venue][2] - ARENA[prev[1]][2])
            h = hist.setdefault(t, [])
            three4 = int(len([d for d in h if (x.date - d).days <= 3]) >= 2)     # this is at least the 3rd game in 4 nights
            f.update({f"{side}_b2b": int(rest == 0), f"{side}_r1": int(rest == 1), f"{side}_r3": int(rest >= 3), f"{side}_34": three4,
                      f"{side}_mi": mi / 1000, f"{side}_tz": tz})
            last[t] = (x.date, venue); h.append(x.date)
        f["alt"] = int(x.home in ALT)
        rows.append(f)
    F = pd.DataFrame(rows)
    out = pd.DataFrame(index=G.index)
    for k in ("b2b", "r1", "r3", "34", "mi", "tz"): out[k] = F["h_" + k] - F["a_" + k]
    out["alt"] = F.alt
    return pd.concat([G, out], axis=1)


SFEATS = ["b2b", "r1", "r3", "34", "mi", "tz", "alt"]


def s1(G, S):
    A = abbr()
    G = G.copy(); G["home"] = G.home.map(A); G["away"] = G.away.map(A)
    G = pd.concat([schedule_features(g) for _, g in G.groupby("season")], ignore_index=True)
    c = FP.fit_scale(S, G, TRAIN)                                                  # scale from training seasons only
    G["base"] = c["k"] * G.s_model + c["hca"]; G["m"] = G.hp - G.ap; G["res"] = G.m - G.base
    tr = G[G.season.isin(TRAIN)]; te = G[G.season.isin(TEST)]
    X = tr[SFEATS].values; b = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), tr.res.values, rcond=None)[0]
    def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    y = (te.m > 0).astype(int).values
    p0 = norm.cdf(te.base / c["sd"]); p1 = norm.cdf((te.base + b[0] + te[SFEATS].values @ b[1:]) / c["sd"])
    gain = (ll(p0, y) - ll(p1, y)) / ll(p0, y)
    # final coefficients on all seasons (used live)
    Xa = G[SFEATS].values; ba = np.linalg.lstsq(np.column_stack([np.ones(len(Xa)), Xa]), G.res.values, rcond=None)[0]
    return dict(coef_train=dict(zip(["const"] + SFEATS, map(float, b))), coef_all=dict(zip(["const"] + SFEATS, map(float, ba))),
                ll_without=ll(p0, y), ll_with=ll(p1, y), gain=gain, pass_=bool(gain >= 0.002), games=int(len(te)))


def availability_for(t, pids):
    rates = {}; seen = {}
    for k, w in ((t - 1, 3), (t - 2, 2), (t - 3, 1)):
        if k < 2010: continue
        g = FP.games(k)
        tg = pd.read_csv(os.path.join(OUT, f"teamgames_{k}.csv.gz")); tg = tg[tg.kind == "rs"].groupby("team").gid.nunique()
        for p in pids:
            if p in g.index:
                rates.setdefault(p, []).append((w, min(1.0, g.loc[p, "gp"] / tg.get(g.loc[p, "team"], 82)))); seen.setdefault(p, []).append(k)
    out = {}
    for p in pids:
        r = rates.get(p, []); s = seen.get(p, [])
        if s and (t - 1) not in s and (t - 2) in s: r = r + [(3, 0.0)]
        num = sum(w * x for w, x in r) + 1.5 * LEAGUE_AVAIL; den = sum(w for w, _ in r) + 1.5
        out[p] = float(np.clip(num / den, 0.3, 0.97))
    return pd.Series(out)


def team_frames():
    """per team-season: preseason strength with and without availability, continuity, youth, rookie minutes, actual."""
    rows = []
    for t in range(2014, 2026):
        M = FP.fit_player_model(range(2013, t))
        g = FP.games(t); roster = g.team0
        P = FP.project(t, M, list(roster.index)); mpg = FP.expected_mpg(t, list(roster.index), P)
        T0, D0 = FP.team_strength(P, roster, mpg)
        av = availability_for(t, list(roster.index))
        T1, _ = FP.team_strength(P, roster, mpg, av)
        prev = FP.games(t - 1)
        prev_min = (prev.secs / 60)
        tot_prev = prev.groupby("team").secs.sum() / 60
        ages = pd.Series({p: FP.age_at(p, t) for p in roster.index})
        D0["age"] = ages.reindex(D0.index).fillna(26)
        D0["new"] = ~D0.index.isin(FP.ratings(t - 1).index)
        mov = FP.team_mov(t)
        for tm in T0.index:
            ro = roster[roster == tm].index
            ret = sum(prev_min.get(p, 0) for p in ro if p in prev.index and prev.loc[p, "team"] == tm)
            d = D0[D0.team == tm]
            rows.append(dict(season=t, team=tm, s0=T0.loc[tm, "net"], s1=T1.loc[tm, "net"], actual=mov.get(tm, np.nan),
                             cont=ret / tot_prev.get(tm, np.nan), age=float(np.average(d.age, weights=d.m)), rook=float(d.m[d.new].sum() / 240)))
        print(t, flush=True)
    R = pd.DataFrame(rows)
    for k in ("s0", "s1"): R[k + "c"] = R[k] - R.groupby("season")[k].transform("mean")
    return R


def s2(R):
    out = {}
    for k in ("s0c", "s1c"):
        tr = R[R.season.isin(TRAIN)]; kk = float((tr[k] * tr.actual).sum() / (tr[k] ** 2).sum())
        te = R[R.season.isin(TEST)]; out[k] = float((kk * te[k] - te.actual).abs().mean()); out["k_" + k] = kk
    gain = 1 - out["s1c"] / out["s0c"]
    return dict(mae_without=out["s0c"], mae_with=out["s1c"], gain=gain, pass_=bool(gain >= 0.02))


def s3(R, base="s1c"):
    """heteroscedastic spread of (actual - k*strength) by continuity, age, rookie minutes."""
    from scipy.optimize import minimize
    tr = R[R.season.isin(TRAIN)].copy(); te = R[R.season.isin(TEST)].copy()
    k = float((tr[base] * tr.actual).sum() / (tr[base] ** 2).sum())
    for d in (tr, te, R): d["res"] = d.actual - k * d[base]
    mu = tr[["cont", "age", "rook"]].mean(); sd_ = tr[["cont", "age", "rook"]].std()
    Z = lambda d: ((d[["cont", "age", "rook"]] - mu) / sd_).values
    def nll(b, d, mean=False):
        z = Z(d); s = np.exp(b[0] + z @ b[1:4]); m = z @ b[4:7] if mean else 0
        return -norm.logpdf(d.res.values - m, 0, s).sum()
    c0 = float(np.log(tr.res.std()))
    fh = minimize(lambda b: nll(np.r_[b, 0, 0, 0], tr), np.r_[c0, 0, 0, 0]).x
    fm = minimize(lambda b: nll(b, tr, True), np.r_[fh, 0, 0, 0]).x
    const = -norm.logpdf(te.res.values, 0, tr.res.std()).sum()
    het = nll(np.r_[fh, 0, 0, 0], te); mean = nll(fm, te, True)
    # final fit on all seasons (used live)
    fh_all = minimize(lambda b: nll(np.r_[b, 0, 0, 0], R), np.r_[c0, 0, 0, 0]).x
    fm_all = minimize(lambda b: nll(b, R, True), np.r_[fh_all, 0, 0, 0]).x
    use_mean = bool(mean < het)
    corr = {f: float(np.corrcoef(tr[f], tr.res.abs())[0, 1]) for f in ("cont", "age", "rook")}
    return dict(k=k, logscore_const=float(const), logscore_spread=float(het), logscore_spread_mean=float(mean),
                pass_=bool(het < const), use_mean=use_mean and bool(mean < const),
                coef_spread=dict(zip(["log_sd", "cont", "age", "rook"], map(float, fh_all))),
                coef_mean=dict(zip(["cont", "age", "rook"], map(float, fm_all[4:7]))) if use_mean else None,
                z_mu=mu.to_dict(), z_sd=sd_.to_dict(), corr_abs_res_train=corr, sd_const=float(tr.res.std()))


def s4(G, S):
    """last 15 games: teams out of the race and teams with their seed locked, vs. their blended strength."""
    A = abbr(); EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}
    c = FP.fit_scale(S, G, list(range(2014, 2026))); K = 20
    rec = []
    for t, g in G.groupby("season"):
        g = g.sort_values(["date", "gid"])
        pre = (S[S.season == t].set_index("team").model * c["k"]).to_dict()
        n = {}; sm = {}; w = {}; l = {}
        tot = pd.concat([g.home, g.away]).value_counts().to_dict()
        for x in g.itertuples():
            h, a = x.home, x.away
            def cur(tm):
                N = n.get(tm, 0); p = pre.get(tm, 0); return p * K / (K + N) + (sm.get(tm, 0) / N if N else p) * N / (K + N)
            bh, ba = cur(h), cur(a); m = x.hp - x.ap; exp_m = bh - ba + c["hca"]
            # state at this point
            wp = {tm: w.get(tm, 0) / max(1, w.get(tm, 0) + l.get(tm, 0)) for tm in tot}
            league_bottom = set(sorted(wp, key=lambda z: wp[z])[:6])
            locked = set()
            for conf in (True, False):
                ts = sorted([z for z in tot if (A.get(z) in EAST) == conf], key=lambda z: -(w.get(z, 0) - l.get(z, 0)))
                if len(ts) >= 3:
                    g3 = (w.get(ts[2], 0) - l.get(ts[2], 0)) / 2
                    for z in ts[:2]:
                        if (w.get(z, 0) - l.get(z, 0)) / 2 - g3 >= 5: locked.add(z)
            for tm, sign in ((h, 1), (a, -1)):
                left = tot[tm] - n.get(tm, 0)
                if left <= 15:
                    rec.append(dict(season=t, team=tm, res=sign * (m - exp_m), tank=tm in league_bottom, lock=tm in locked))
            sm[h] = sm.get(h, 0) + m - c["hca"] + pre.get(a, 0); sm[a] = sm.get(a, 0) - m + c["hca"] + pre.get(h, 0)
            n[h] = n.get(h, 0) + 1; n[a] = n.get(a, 0) + 1
            if m > 0: w[h] = w.get(h, 0) + 1; l[a] = l.get(a, 0) + 1
            else: w[a] = w.get(a, 0) + 1; l[h] = l.get(h, 0) + 1
    D = pd.DataFrame(rec)
    out = {}
    rng = np.random.default_rng(0)
    for k in ("tank", "lock"):
        grp = D[D[k]]; oth = D[~D.tank & ~D.lock]
        shift = grp.res.mean() - oth.res.mean()
        keys = list(grp.groupby(["season", "team"]).groups.keys()); gg = {kk: v.res.values for kk, v in grp.groupby(["season", "team"])}
        boots = []
        for _ in range(2000):
            pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            boots.append(np.concatenate([gg[p] for p in pick]).mean() - oth.res.mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[k] = dict(games=int(len(grp)), team_seasons=len(keys), shift=float(shift), lo=float(lo), hi=float(hi),
                      pass_=bool(abs(shift) >= 1.5 and (hi < 0 or lo > 0)))
    return out


def main():
    S = pd.read_csv(os.path.join(OUT, "fc_bt_teams.csv")); G = pd.read_csv(os.path.join(OUT, "fc_bt_games.csv"), parse_dates=["date"])
    res = {}
    res["S1"] = s1(G, S); print("S1", {k: v for k, v in res["S1"].items() if not k.startswith("coef")}, flush=True)
    R = team_frames(); R.to_csv(os.path.join(OUT, "fc_team_frames.csv"), index=False)
    res["S2"] = s2(R); print("S2", res["S2"], flush=True)
    res["S3"] = s3(R, "s1c" if res["S2"]["pass_"] else "s0c"); print("S3", {k: v for k, v in res["S3"].items() if k in ("logscore_const", "logscore_spread", "logscore_spread_mean", "pass_", "use_mean", "corr_abs_res_train", "coef_spread", "coef_mean")}, flush=True)
    res["S4"] = s4(G, S); print("S4", res["S4"], flush=True)
    json.dump(res, open(os.path.join(OUT, "fc_extras.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
