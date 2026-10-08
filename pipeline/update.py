"""Refresh the Three Gates explorer data.

    python pipeline/update.py              # weekly refresh (downloads data, fits the model, writes data/*.json)
    python pipeline/update.py --boot 20    # quick test run with fewer resamples
    python pipeline/update.py --local DIR  # use nbastats_YYYY.csv / shotdetail_YYYY.csv already in DIR (testing)

Window: the two most recent completed seasons plus the current one. The oldest completed season fades out and the
current season fades in as games are played (weight = games played / 41, capped at 1), so a new season cannot swing
the numbers in its first few weeks.
"""
import os, sys, io, re, json, time, tarfile, shutil, pickle, argparse, datetime, subprocess, urllib.request
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "out"); PUB = os.path.join(REPO, "data")
LIST_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"
RAW_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets/{}.tar.xz"
XR, RA, SC, LAM = 1.331, 26.1, 1000.0, 2500.0       # rim attempts per 100 poss., league rim-shot value constants, ridge strength
MIN_MINUTES = 1350                                    # defensive minutes needed to be listed
RAMP_GAMES = 41                                       # games before the current season counts in full
LEAKY_SHARE = None                                    # filled in below
SPLINE = (0.027, 0.415, 0.208, 2.0)                   # projection: next-season rim score = a + b*x + c*max(x-knot, 0), from the backtest
PROJ_SD = 0.95                                        # residual spread of that projection in the backtest


def log(*a): print(*a, flush=True)


# ---------------------------------------------------------------- data
def get_list():
    txt = urllib.request.urlopen(LIST_URL, timeout=60).read().decode()
    return dict(l.split("=", 1) for l in txt.splitlines() if "=" in l)


def fetch(key):
    os.makedirs(os.path.join(DATA, "raw"), exist_ok=True)
    last = None
    for attempt in range(4):
        try:
            blob = urllib.request.urlopen(RAW_URL.format(key), timeout=300).read()
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:xz") as t:
                try: t.extractall(os.path.join(DATA, "raw"), filter="data")
                except TypeError: t.extractall(os.path.join(DATA, "raw"))
            return os.path.join(DATA, "raw", key + ".csv")
        except Exception as e:
            last = e; time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not download {key}: {last}")


def prepare_season(y, avail, local=None):
    """write data/nbastats_{y}.csv and data/shotdetail_{y}.csv. Returns False when the season has no data yet."""
    os.makedirs(DATA, exist_ok=True)
    if local:
        for k in ("nbastats", "shotdetail"):
            src = os.path.join(local, f"{k}_{y}.csv")
            if not os.path.exists(src): return False
            if os.path.abspath(src) != os.path.abspath(os.path.join(DATA, f"{k}_{y}.csv")): shutil.copy(src, os.path.join(DATA, f"{k}_{y}.csv"))
        return True
    po = y >= 2025                                    # 2024-25 is regular season only (as first published); later seasons add playoffs when they exist
    pbp, shots = [], []
    for suffix in ([""] + (["po_"] if po else [])):
        if f"nbastats_{suffix}{y}" in avail:
            pbp.append(pd.read_csv(fetch(f"nbastats_{suffix}{y}"), low_memory=False))
        elif f"cdnnba_{suffix}{y}" in avail:
            src = fetch(f"cdnnba_{suffix}{y}"); dst = os.path.join(DATA, f"conv_{suffix}{y}.csv")
            subprocess.run([sys.executable, os.path.join(HERE, "convert_cdn.py"), src, dst], check=True, stdout=subprocess.DEVNULL)
            pbp.append(pd.read_csv(dst, low_memory=False))
        if f"shotdetail_{suffix}{y}" in avail:
            shots.append(pd.read_csv(fetch(f"shotdetail_{suffix}{y}"), low_memory=False))
    if not pbp or not shots: return False
    pd.concat(pbp, ignore_index=True).to_csv(os.path.join(DATA, f"nbastats_{y}.csv"), index=False)
    pd.concat(shots, ignore_index=True).to_csv(os.path.join(DATA, f"shotdetail_{y}.csv"), index=False)
    return True


def build_stints(y):
    subprocess.run([sys.executable, os.path.join(HERE, "build_stints.py"), str(y)], check=True, cwd=HERE, stdout=subprocess.DEVNULL)
    stints = pickle.load(open(os.path.join(OUT, f"stints_{y}.pkl"), "rb")); tally = pickle.load(open(os.path.join(OUT, f"tally_{y}.pkl"), "rb"))
    return stints, tally


# ---------------------------------------------------------------- shots -> rim stints
def shot_values(y, tables):
    cols = ["GAME_ID", "TEAM_ID", "TEAM_NAME", "PLAYER_ID", "PLAYER_NAME", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING", "ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC", "SHOT_ZONE_RANGE", "SHOT_MADE_FLAG"]
    A = pd.read_csv(os.path.join(DATA, f"shotdetail_{y}.csv"), usecols=lambda c: c in cols, low_memory=False)
    A["pts"] = np.where(A.SHOT_TYPE.str.startswith("3"), 3, 2)
    A["act"] = np.where(A.ACTION_TYPE.isin(tables["top"]), A.ACTION_TYPE, "Other")
    g1 = {(a, b, c): (s, n) for a, b, c, s, n in tables["g1"]}; g2 = {(a, b): (s, n) for a, b, s, n in tables["g2"]}
    xfg = []
    for zb, zr, act in zip(A.SHOT_ZONE_BASIC, A.SHOT_ZONE_RANGE, A.act):
        s2, n2 = g2.get((zb, zr), (tables["league_fg"] * 40, 40))
        s1, n1 = g1.get((zb, zr, act), (0, 0))
        xfg.append((s1 + 40 * s2 / n2) / (n1 + 40))
    A["xfg"] = xfg; A["xpts"] = A.xfg * A.pts; A["rim"] = (A.SHOT_ZONE_BASIC == "Restricted Area").astype(int)
    return A


def build_rim(y, stints, tables):
    A = shot_values(y, tables)
    raw = pd.DataFrame(stints, columns=["season", "gid", "per", "H", "V", "secs", "pts_h", "pts_v", "poss_h", "poss_v", "home", "vis"])
    raw["ord"] = np.arange(len(raw)); raw["end"] = raw.groupby(["gid", "per"]).secs.cumsum()
    sh = A; el = np.where(sh.PERIOD <= 4, 720, 300) - (sh.MINUTES_REMAINING * 60 + sh.SECONDS_REMAINING)
    grp = {k: (v.end.values, v.ord.values) for k, v in raw.groupby(["gid", "per"])}; teams = raw[["gid", "home"]].drop_duplicates().set_index("gid").home.to_dict()
    n = len(raw); cols = {k: np.zeros(n) for k in ("xq_h", "xq_v", "rn_h", "rn_v", "rm_h", "rm_v", "rx_h", "rx_v")}
    for gid_, per, e, tid, xp, mk, isr, xf in zip(sh.GAME_ID, sh.PERIOD, el, sh.TEAM_ID, sh.xpts, sh.SHOT_MADE_FLAG, sh.rim, sh.xfg):
        gg = int(gid_); k = (gg, int(per)) if (gg, int(per)) in grp else (str(gg).zfill(10), int(per))
        if k not in grp: continue
        ends, ords = grp[k]; o = ords[min(np.searchsorted(ends, e, side="left"), len(ends) - 1)]; s = "h" if teams.get(gg) == tid else "v"
        if isr: cols["rn_" + s][o] += 1; cols["rm_" + s][o] += mk; cols["rx_" + s][o] += xf
        else: cols["xq_" + s][o] += xp
    for k, v in cols.items(): raw[k] = v
    raw["poss"] = (raw.poss_h + raw.poss_v) / 2
    names = A.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID").PLAYER_NAME.to_dict()
    tnames = A.drop_duplicates("TEAM_ID").set_index("TEAM_ID").TEAM_NAME.to_dict() if "TEAM_NAME" in A else {}
    return raw[raw.poss >= 1.0].reset_index(drop=True), names, tnames


def rows(r):
    a = pd.DataFrame(dict(att=r.H, deff=r.V, poss=r.poss_h, xq=r.xq_h, rn=r.rn_h, rm=r.rm_h, rx=r.rx_h, gid=r.gid, secs=r.secs, tm=r.vis))
    b = pd.DataFrame(dict(att=r.V, deff=r.H, poss=r.poss_v, xq=r.xq_v, rn=r.rn_v, rm=r.rm_v, rx=r.rx_v, gid=r.gid, secs=r.secs, tm=r.home))
    d = pd.concat([a, b], ignore_index=True); d = d[d.poss >= 1.0].reset_index(drop=True)
    d["nonrim"] = 100 * d.xq / d.poss; d["deter"] = 100 * d.rn / d.poss
    d["contest"] = np.where(d.rn > 0, (d.rm - d.rx) / d.rn.clip(lower=1) * 100, 0.0)
    return d


# ---------------------------------------------------------------- model
def fit(d, target, w):
    pl = sorted(set(p for t in d.att for p in t) | set(p for t in d.deff for p in t)); idx = {p: i for i, p in enumerate(pl)}; n = len(pl); r_, c_ = [], []
    for i, (a, df) in enumerate(zip(d.att.values, d.deff.values)):
        for p in a: r_.append(i); c_.append(idx[p])
        for p in df: r_.append(i); c_.append(n + idx[p])
    X = sp.csr_matrix((np.ones(len(r_)), (r_, c_)), shape=(len(d), 2 * n))
    m = Ridge(alpha=LAM, solver="sparse_cg", max_iter=3000, tol=1e-6).fit(X, d[target].values, sample_weight=w)
    return idx, m.coef_[n:2 * n]


def gates(d):
    wm = d.wm.values; f = {k: fit(d, k, d[wc].values * wm) for k, wc in (("nonrim", "poss"), ("deter", "poss"), ("contest", "rn"))}
    idx = f["nonrim"][0]
    return idx, -f["nonrim"][1], -f["deter"][1] * XR, -f["contest"][1] / 100 * 2 * RA


def exposure(d, key=None):
    s = {}
    for df, sec, w in zip(d.deff.values, d.secs.values, d.wm.values):
        for p in df: s[p] = s.get(p, 0) + sec * w
    return s


def backstop(d, z):
    num, den = {}, {}
    for df, sec, w in zip(d.deff.values, d.secs.values, d.wm.values):
        zs = [z.get(p, 0.0) for p in df]
        for i, p in enumerate(df):
            o = max(zs[:i] + zs[i + 1:]); num[p] = num.get(p, 0) + o * sec * w; den[p] = den.get(p, 0) + sec * w
    return {p: num[p] / den[p] for p in num}


def latest_team(frames):
    """team of the most recent season the player appeared in; 'Multiple teams' if he had more than one in that season."""
    out = {}
    for d in frames:                                      # oldest first, so later seasons overwrite
        c = {}
        for df, tm, sec in zip(d.deff.values, d.tm.values, d.secs.values):
            for p in df: c.setdefault(p, {}); c[p][tm] = c[p].get(tm, 0) + sec
        for p, t in c.items():
            tot = sum(t.values()); main = max(t, key=t.get)
            out[p] = ("Multiple teams" if len(t) > 1 and min(t.values()) / tot > 0.05 else main)
    return out


def ordinal_arch(P, leaky_share):
    gs = ["redirect", "deter", "contest"]; strong = {g: P[g] >= P[g].quantile(.8) for g in gs}
    P["sdv"] = P[gs].sum(axis=1); leak_cut = P.sdv.quantile(leaky_share)
    lab = {"redirect": "Redirector", "deter": "Deterrer", "contest": "Contester"}; order = {"contest": 0, "deter": 1, "redirect": 2}

    def arch(r):
        s = [g for g in gs if strong[g][r.name]]
        if len(s) == 1: return lab[s[0]]
        if len(s) == 2: return "Two-gate " + "+".join(lab[x] for x in sorted(s, key=lambda x: order[x]))
        if len(s) == 3: return "Three-gate"
        return "Leaky" if r.sdv < leak_cut else "Balanced"
    return P.apply(arch, axis=1)


def ordinal(n): return "%d%s" % (n, "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=200); ap.add_argument("--local"); ap.add_argument("--today"); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--season", type=int, help="override the current season (start year)"); ap.add_argument("--force-projection", action="store_true")
    ap.add_argument("--out", default=PUB)
    a = ap.parse_args()
    today = datetime.date.fromisoformat(a.today) if a.today else datetime.date.today()
    cur = a.season or (today.year if today.month >= 9 else today.year - 1)
    tables = json.load(open(os.path.join(HERE, "shot_tables.json")))
    avail = {} if a.local else get_list()
    os.makedirs(OUT, exist_ok=True)
    seasons = [y for y in (cur - 2, cur - 1, cur) if prepare_season(y, avail, a.local)]
    log("seasons with data:", seasons)
    D, T, NAMES, TNAMES, G = {}, {}, {}, {}, {}
    for y in seasons:
        st, tally = build_stints(y); r, nm, tn = build_rim(y, st, tables); d = rows(r)
        D[y], T[y] = d, tally; NAMES.update(nm); TNAMES.update(tn); G[y] = d.gid.nunique()
        log(y, "games", d.gid.nunique(), "rows", len(d))
    cur_has = cur in D
    # season weights
    games_cur = 0
    if cur_has:
        gm = {}
        for d in [D[cur]]:
            for gid, t in zip(d.gid.values, d.tm.values): gm.setdefault(t, set()).add(gid)
        games_cur = max(len(v) for v in gm.values())
    ramp = min(1.0, games_cur / RAMP_GAMES) if cur_has else 0.0
    W = {}
    for y in D:
        W[y] = ramp if y == cur else (1.0 if y == cur - 1 else 1.0 - ramp)
    used = [y for y in sorted(D) if W[y] > 0.0]
    if len(used) < 2: sys.exit("not enough seasons to fit: %s" % used)
    log("season weights:", {y: round(W[y], 3) for y in used}, "| current-season games per team:", games_cur)
    for y in used: D[y]["wm"] = W[y]
    # block-rate z per season (backstop context only)
    BR = {}
    for y in used:
        t = pd.DataFrame.from_dict(T[y], orient="index"); thr = 60000 * (min(1.0, games_cur / 82) if y == cur else 1.0)
        br = t.blk / (t.sec / 60.0); ok = t.sec >= max(thr, 6000)
        mu, sd = br[ok].mean(), br[ok].std(); BR[y] = ((br - mu) / sd).where(ok, 0.0).to_dict()
    d = pd.concat([D[y] for y in used], ignore_index=True)
    t0 = time.time(); idx, red, det, con = gates(d); sec = exposure(d); log("fit", round(time.time() - t0, 1), "s")
    keep = {p for p in idx if sec.get(p, 0) / 60 >= MIN_MINUTES}
    P = pd.DataFrame([dict(pid=p, mins=sec[p] / 60, redirect=red[i], deter=det[i], contest=con[i]) for p, i in idx.items() if p in keep]).set_index("pid")
    ma = {}
    for y in used:
        m = backstop(D[y], BR[y]); s = exposure(D[y])
        for p, v in m.items(): u = ma.get(p, (0, 0)); ma[p] = (u[0] + v * s[p], u[1] + s[p])
    P["ma"] = [ma[p][0] / ma[p][1] if p in ma and ma[p][1] else 0.0 for p in P.index]
    lt = latest_team([D[y] for y in used]); P["team"] = [TNAMES.get(lt[p], lt[p]) if lt.get(p) != "Multiple teams" else "Multiple teams" for p in P.index]
    # bootstrap by game
    gids = d.gid.values; ug = np.unique(gids); pos = {g: np.where(gids == g)[0] for g in ug}; rng = np.random.default_rng(a.seed); res = []
    for b in range(a.boot):
        pick = rng.choice(ug, len(ug)); ix = np.concatenate([pos[g] for g in pick]); s = d.iloc[ix].reset_index(drop=True)
        i2, r2, d2, c2 = gates(s)
        for p, i in i2.items():
            if p in keep: res.append((b, p, r2[i], d2[i], c2[i]))
        if b % 20 == 0: log("resample", b, round(time.time() - t0))
    R = pd.DataFrame(res, columns=["b", "pid", "redirect", "deter", "contest"]); R["rim"] = R.deter + R.contest
    se = R.groupby("pid")[["redirect", "deter", "contest", "rim"]].std().add_suffix("_se"); P = P.join(se)
    R["rk"] = R.groupby("b").rim.rank(ascending=False); q = R.groupby("pid").rk.quantile([.05, .95]).unstack()
    P["rk_lo"] = q[.05].round(); P["rk_hi"] = q[.95].round(); P["raw_rim"] = P.deter + P.contest
    P = P.dropna(subset=["rim_se"]); P["name"] = [NAMES.get(p, str(p)) for p in P.index]
    leaky_share = float(json.load(open(os.path.join(HERE, "settings.json")))["leaky_share"])
    P["arch"] = ordinal_arch(P, leaky_share); P = P.sort_values("raw_rim", ascending=False)
    # ---- write explorer data
    teams = sorted(set(P.team)); arcs = sorted(set(P.arch)); rows_out = []
    for p, r in P.iterrows():
        rows_out.append([r["name"], teams.index(r.team), int(round(r.mins)), round(r.redirect, 2), round(r.deter, 2), round(r.contest, 2), round(r.redirect_se, 2), round(r.deter_se, 2),
                         round(r.contest_se, 2), round(r.rim_se, 2), int(r.rk_lo), int(r.rk_hi), round(r.ma, 2), arcs.index(r.arch)])
    last_game = None
    try:
        sd = pd.read_csv(os.path.join(DATA, f"shotdetail_{used[-1]}.csv"), usecols=["GAME_ID", "GAME_DATE"], low_memory=False); have = set(int(g) for g in D[used[-1]].gid); raw = sd[sd.GAME_ID.astype(int).isin(have)].GAME_DATE.astype(str)
        last_game = datetime.datetime.strptime(raw.max(), "%Y%m%d").date().isoformat()
    except Exception: pass
    names = {y: f"{y}-{str(y + 1)[2:]}" for y in used}
    meta = dict(updated=today.isoformat(), data_through=last_game, seasons=[names[y] for y in used], weights={names[y]: round(W[y], 2) for y in used},
                current_season=f"{cur}-{str(cur + 1)[2:]}", current_season_games=int(games_cur), players=len(rows_out), resamples=a.boot, min_minutes=MIN_MINUTES)
    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, "players.json"); new = dict(meta=meta, teams=teams, arch=arcs, rows=rows_out)
    try:
        old = json.load(open(fp, encoding="utf-8")); same = old["rows"] == rows_out and old["teams"] == teams and {k: v for k, v in old["meta"].items() if k != "updated"} == {k: v for k, v in meta.items() if k != "updated"}
    except Exception: same = False
    if same: log("no change in the data, leaving players.json as it is")
    else:
        json.dump(new, open(fp, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":")); log("wrote players.json", len(rows_out), "players")
    # ---- preseason projection: only written before the new season has games (or when forced)
    if (not cur_has) or a.force_projection:
        f = lambda x: SPLINE[0] + SPLINE[1] * x + SPLINE[2] * np.maximum(x - SPLINE[3], 0)
        P["proj"] = f(P.raw_rim); rg = np.random.default_rng(5); S = P.proj.values[None, :] + rg.normal(0, PROJ_SD, (20000, len(P)))
        rk = (-S).argsort(1).argsort(1) + 1; P["p10"] = (rk <= 10).mean(0); P["p1"] = (rk == 1).mean(0)
        P["pr_lo"] = np.percentile(rk, 10, axis=0).round(); P["pr_hi"] = np.percentile(rk, 90, axis=0).round()
        P = P.sort_values("proj", ascending=False)
        proj = []
        for i, (p, r) in enumerate(P.head(60).iterrows(), 1):
            proj.append([i, r["name"], r.team, int(round(r.mins)), round(r.raw_rim, 2), round(r.proj, 2), round(r.proj - 1.28 * PROJ_SD, 2), round(r.proj + 1.28 * PROJ_SD, 2),
                         round(float(r.p10), 3), round(float(r.p1), 3), int(r.pr_lo), int(r.pr_hi), r.arch])
        pm = dict(season=f"{cur}-{str(cur + 1)[2:]}", built_from=[names[y] for y in used], built_on=today.isoformat(), slope=SPLINE[1], resid_sd=PROJ_SD)
        pp = os.path.join(a.out, "projection.json")
        try: old = json.load(open(pp, encoding="utf-8")); same = old["rows"] == proj and old["meta"]["built_from"] == pm["built_from"]
        except Exception: same = False
        if not same:
            json.dump(dict(meta=pm, rows=proj), open(pp, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":")); log("wrote projection.json")
    log("done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
