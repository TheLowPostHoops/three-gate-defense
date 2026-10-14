"""Simulate the 2026-27 season: every remaining game, the play-in, the playoffs and the draft lottery.

    python3 fc_sim.py --sims 1000000

Reads out/fc_players_2026.csv (fc_season.py), out/fc_backtest.json (scale, home court, blend K) and the schedule:
the live NBA schedule (with final scores) when it can be downloaded, otherwise pipeline/inputs/schedule_2026.txt.
Writes out/fc_sim_2026.json (team odds) and out/fc_sim_players.npz (per-sim player inputs for the awards step).
"""
import os, sys, json, math, argparse, datetime, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); INP = os.path.join(HERE, "inputs")
SEASON = 2026
EAST = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"]
WEST = ["DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"]
TEAMS = EAST + WEST
TI = {t: i for i, t in enumerate(TEAMS)}
LOTTERY = np.array([140, 140, 140, 125, 105, 90, 75, 60, 45, 30, 20, 15, 10, 5]) / 1000.0   # chance of the #1 pick, worst record first
SCHED_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
STATIC = os.path.join(HERE, "forecast_model")
KAPPA = 8.0             # spread of each player's share of games played (beta concentration)


def mpath(name):
    p = os.path.join(STATIC, name)
    return p if os.path.exists(p) else os.path.join(OUT, name)


# ------------------------------------------------------------------ schedule
def load_schedule():
    """games: date, home, away, status (3 = final), home pts, away pts. Live when possible."""
    try:
        j = json.loads(urllib.request.urlopen(urllib.request.Request(SCHED_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read())
        rows = []
        for d in j["leagueSchedule"]["gameDates"]:
            for g in d["games"]:
                if not str(g["gameId"]).startswith("002"): continue
                h, a = g["homeTeam"]["teamTricode"], g["awayTeam"]["teamTricode"]
                if not h or not a: continue
                dt = pd.Timestamp(g.get("gameDateTimeUTC") or g.get("gameDateUTC")).tz_localize(None) - pd.Timedelta(hours=5)
                rows.append(dict(gid=g["gameId"], date=dt.normalize(), home=h, away=a, status=int(g["gameStatus"]),
                                 hp=g["homeTeam"].get("score") or 0, ap=g["awayTeam"].get("score") or 0))
        S = pd.DataFrame(rows); S["source"] = "live"
        if len(S) > 1000: return S
    except Exception as e:
        print("live schedule unavailable:", str(e)[:120], flush=True)
    rows = []
    for line in open(os.path.join(INP, "schedule_2026.txt")):
        d, gs = line.strip().split(":")
        for x in gs.split(","):
            if x.startswith("TBD"): continue
            rows.append(dict(gid="", date=pd.Timestamp("20" + d), home=x[:3], away=x[3:6], status=1, hp=0, ap=0))
    S = pd.DataFrame(rows); S["source"] = "file"
    return S


# ------------------------------------------------------------------ series odds (2-2-1-1-1)
def series_win(ph, pa):
    """chance the team with home court wins a best-of-7. ph: its chance at home, pa: its chance away (arrays)."""
    order = [ph, ph, pa, pa, ph, pa, ph]
    P = {(0, 0): np.ones_like(ph)}
    win = np.zeros_like(ph)
    for g in order:
        Q = {}
        for (a, b), pr in P.items():
            for (na, nb, q) in ((a + 1, b, g), (a, b + 1, 1 - g)):
                v = pr * q
                if na == 4: win += v
                elif nb == 4: pass
                else: Q[(na, nb)] = Q.get((na, nb), 0) + v
        P = Q
    return win


def main(nsims=1_000_000, chunk=20_000, seed=None):
    t0 = datetime.datetime.now()
    D = pd.read_csv(mpath(f"fc_players_{SEASON}.csv"), index_col=0)
    bt = json.load(open(mpath("fc_backtest.json"))); sc = bt["scale"]; K = bt["F4"]["K"]
    k, hca = sc["k"], sc["hca"]
    S = load_schedule()
    D, live_info = apply_live(D, S)
    D = D[D.team.isin(TEAMS)].copy()
    sd_true = math.sqrt(max(sc["team_resid_sd"] ** 2 - (sc["sd"] ** 2) / 82, 1.0))
    sd_game = math.sqrt(max(sc["sd"] ** 2 - sd_true ** 2, 100.0))
    N = len(D); r = D.net.values.astype(np.float32); w = D.mpg.clip(upper=38).values.astype(np.float32)
    av = D.avail.clip(0.02, 0.97).values
    # S2 failed: projected health does not move a team's expected strength, so each player's draw is taken relative to
    # his own projection (it adds injury luck, not a penalty). This season's games played and current injuries do move it.
    av_pre = D.get("avail_pre", D.avail).clip(0.02, 0.97).values
    X = load_extras(); s1, s3, s4 = X.get("S1", {}), X.get("S3", {}), X.get("S4", {})
    hca_g = s1["hca_sched"] if s1.get("pass_") else hca
    onehot = np.zeros((N, 30), np.float32); onehot[np.arange(N), D.team.map(TI).values] = 1
    # expected (average-health) strength, used to center and for opponent adjustment
    def strength(A):
        A = A / av_pre.astype(np.float32)
        num = (A * (r * w)) @ onehot; den = (A * w) @ onehot
        return 5 * num / den
    exp_s = strength(av[None, :].astype(np.float32))[0]
    center = exp_s.mean()
    pre = k * (exp_s - center)

    done = S[S.status == 3].copy(); left = S[S.status != 3].copy()
    # results so far: wins, and blended strength observations (margin minus home court plus opponent's preseason strength)
    n_g = np.zeros(30); obs = np.zeros(30); w0 = np.zeros(30); l0 = np.zeros(30); pd0 = np.zeros(30)
    for x in done.itertuples():
        h, a = TI[x.home], TI[x.away]; m = x.hp - x.ap
        obs[h] += m - hca + pre[a]; obs[a] += -m + hca + pre[h]; n_g[h] += 1; n_g[a] += 1
        pd0[h] += m; pd0[a] -= m
        if m > 0: w0[h] += 1; l0[a] += 1
        else: w0[a] += 1; l0[h] += 1
    obs_avg = np.where(n_g > 0, obs / np.maximum(n_g, 1), 0)
    bw = K / (K + n_g)                                     # weight on preseason (+ its uncertainty) for each team
    # games still to play, plus the two NBA Cup knockout / makeup games each team has without a fixed opponent yet
    gh = left.home.map(TI).values; ga = left.away.map(TI).values; G = len(left)
    played = n_g + np.bincount(gh, minlength=30) + np.bincount(ga, minlength=30)
    extra = np.clip(82 - played, 0, 4).astype(int)
    sched_adj, sched_team = schedule_adjust(S, left, s1)                    # S1: rest, back-to-backs, travel, altitude
    # S4: each team's last 15 games are played after its standing is known (out of the race / seed locked)
    late = late_mask(left, n_g)
    # S3: how far each team can land from its projection, and the shift for continuity, age and rookie minutes
    mshift, sd_team = surprise(s3, sd_game)
    Hm = np.zeros((G, 30), np.float32); Hm[np.arange(G), gh] = 1
    Am = np.zeros((G, 30), np.float32); Am[np.arange(G), ga] = 1
    east = np.array([t in EAST for t in TEAMS])
    alt_b = s1["coef_all"]["alt"] if s1.get("pass_") else 0.0
    rng = np.random.default_rng(seed)

    # accumulators
    acc = {k_: np.zeros(30) for k_ in ("wins", "wins2", "top6", "playin", "playoffs", "seed1", "r2", "cf", "finals", "title", "lottery", "pick1", "best", "worst", "w60", "w50", "l50")}
    win_hist = np.zeros((30, 83))
    seed_hist = np.zeros((30, 15))
    title_pairs = {}
    player_out = dict(avail=[], wins=[], sims=0)
    keep_player = min(nsims, 100_000)                      # per-sim player inputs kept for the awards step
    a_ = av * KAPPA; b_ = (1 - av) * KAPPA
    rem = nsims
    while rem > 0:
        B = min(chunk, rem); rem -= B
        A = rng.beta(a_, b_, size=(B, N)).astype(np.float32)
        s_inj = k * (strength(A) - center)                 # B x 30 (injury luck)
        var_inj = float(np.var(s_inj - s_inj.mean(0), axis=0).mean())
        sd_s = np.sqrt(np.maximum(sd_team ** 2 - var_inj, 0.25)) if sd_team is not None else math.sqrt(max(sd_true ** 2 - var_inj, 0.25))
        shock = (mshift + rng.normal(0, 1, size=(B, 30)) * sd_s).astype(np.float32)
        s_pre = s_inj + shock
        s = bw * s_pre + (1 - bw) * obs_avg                # current strength (blended with results so far)
        mh = s[:, gh] - s[:, ga] + hca_g + sched_adj + rng.normal(0, sd_game, size=(B, G)).astype(np.float32)
        early = ~late.any(1)
        hw = (mh > 0).astype(np.float32)
        wins_e = w0 + hw[:, early] @ Hm[early] + (1 - hw[:, early]) @ Am[early]
        if late.any() and s4:
            # standing when each team's last 15 begin: bottom 6 in the league, or a top-2 seed with a 5-game cushion
            gp_e = n_g + Hm[early].sum(0) + Am[early].sum(0)
            wpct = wins_e / np.maximum(gp_e, 1)
            tank = np.zeros((B, 30), bool); lock = np.zeros((B, 30), bool)
            tank[np.arange(B)[:, None], np.argsort(wpct, axis=1)[:, :6]] = True
            for mask in (east, ~east):
                idx = np.where(mask)[0]; nl = 2 * wins_e[:, idx] - gp_e[idx]
                o = np.argsort(-nl, axis=1); srt = np.take_along_axis(nl, o, 1)
                ok = (srt[:, 1] - srt[:, 2]) / 2 >= 5
                for j in (0, 1): lock[np.arange(B)[ok], idx[o[ok, j]]] = True
            sh = np.zeros((B, 30), np.float32)
            if s4.get("tank", {}).get("pass_"): sh += tank * s4["tank"]["shift"]
            if s4.get("lock", {}).get("pass_"): sh += lock * s4["lock"]["shift"]
            lh = late[:, 0]; la = late[:, 1]
            mh = mh + np.where(lh, sh[:, gh], 0) - np.where(la, sh[:, ga], 0)
            hw = (mh > 0).astype(np.float32)
        wins = w0 + hw @ Hm + (1 - hw) @ Am
        marg = pd0 + mh @ Hm - mh @ Am
        # extra games without fixed opponent: against a league-average team on a neutral floor
        for t in range(30):
            if extra[t]:
                p = 1 / (1 + np.exp(-(s[:, t]) / (sd_game / 1.7)))
                wins[:, t] += rng.binomial(extra[t], p)
        key = wins + 1e-4 * marg + 1e-7 * rng.random((B, 30))
        acc["wins"] += wins.sum(0); acc["wins2"] += (wins ** 2).sum(0)
        acc["w60"] += (wins >= 60).sum(0); acc["w50"] += (wins >= 50).sum(0); acc["l50"] += (wins <= 32).sum(0)
        wi = np.clip(np.rint(wins).astype(int), 0, 82)
        for t in range(30): win_hist[t] += np.bincount(wi[:, t], minlength=83)[:83]
        best = key.argmax(1); worst = key.argmin(1)
        np.add.at(acc["best"], best, 1); np.add.at(acc["worst"], worst, 1)
        member = np.zeros((B, 30), bool)
        conf_champ = {}
        for conf, mask in (("E", east), ("W", ~east)):
            idx = np.where(mask)[0]
            order = idx[np.argsort(-key[:, idx], axis=1)]           # B x 15, seed order
            for sd_ in range(15): np.add.at(seed_hist, (order[:, sd_], sd_), 1)
            np.add.at(acc["seed1"], order[:, 0], 1)
            for sd_ in range(6): np.add.at(acc["top6"], order[:, sd_], 1)
            for sd_ in range(6, 10): np.add.at(acc["playin"], order[:, sd_], 1)
            rows = np.arange(B)
            def game(hi, lo):                                         # single game, hi at home
                p = s[rows, hi] - s[rows, lo] + hca_g + alt_b * ALT_IDX[hi] + rng.normal(0, sd_game, B)
                return np.where(p > 0, hi, lo), np.where(p > 0, lo, hi)
            s7, l78 = game(order[:, 6], order[:, 7])
            w910, _ = game(order[:, 8], order[:, 9])
            s8, _ = game(l78, w910)
            seeds = np.column_stack([order[:, :6], s7, s8])           # 1..8
            for sd_ in range(8): np.add.at(acc["playoffs"], seeds[:, sd_], 1); member[rows, seeds[:, sd_]] = True
            def series(x, y):                                        # x, y team indices; home court to better record
                xh = key[rows, x] >= key[rows, y]
                hi = np.where(xh, x, y); lo = np.where(xh, y, x)
                d = s[rows, hi] - s[rows, lo]
                ph = 1 - _ncdf(-(d + hca_g + alt_b * ALT_IDX[hi]) / sd_game); pa = 1 - _ncdf(-(d - hca_g - alt_b * ALT_IDX[lo]) / sd_game)
                win_hi = rng.random(B) < series_win(ph, pa)
                return np.where(win_hi, hi, lo)
            r1 = [series(seeds[:, 0], seeds[:, 7]), series(seeds[:, 3], seeds[:, 4]), series(seeds[:, 2], seeds[:, 5]), series(seeds[:, 1], seeds[:, 6])]
            for x in r1: np.add.at(acc["r2"], x, 1)
            r2 = [series(r1[0], r1[1]), series(r1[2], r1[3])]
            for x in r2: np.add.at(acc["cf"], x, 1)
            champ = series(r2[0], r2[1]); np.add.at(acc["finals"], champ, 1)
            conf_champ[conf] = champ
        ch = series_final(conf_champ["E"], conf_champ["W"], key, s, hca_g, sd_game, rng)
        np.add.at(acc["title"], ch, 1)
        for e_, w_ in zip(conf_champ["E"][:2000], conf_champ["W"][:2000]):
            kk = (TEAMS[e_], TEAMS[w_]); title_pairs[kk] = title_pairs.get(kk, 0) + 1
        lottery_odds(key, member, acc)
        player_out["sims"] += B
        if sum(len(x) for x in player_out["avail"]) < keep_player:
            player_out["avail"].append(A[: max(0, keep_player - sum(len(x) for x in player_out["avail"]))])
            player_out["wins"].append(wins[: len(player_out["avail"][-1])].astype(np.float32))
    n = nsims
    res = []
    for t in TEAMS:
        i = TI[t]; m = acc["wins"][i] / n; sdw = math.sqrt(max(acc["wins2"][i] / n - m * m, 0))
        cdf = np.cumsum(win_hist[i]) / n
        q = lambda p_: int(np.searchsorted(cdf, p_))
        res.append(dict(team=t, conf="East" if t in EAST else "West", strength=round(float(pre[i]), 2), now=round(float(bw[i] * pre[i] + (1 - bw[i]) * obs_avg[i]), 2),
                        w=int(w0[i]), l=int(l0[i]), proj_w=round(m, 1), proj_l=round(82 - m, 1), w10=q(.1), w90=q(.9),
                        **{k_: float(acc[k_][i] / n) for k_ in ("top6", "playin", "playoffs", "seed1", "r2", "cf", "finals", "title", "lottery", "pick1", "best", "worst", "w60", "w50", "l50")},
                        seeds=[float(x) for x in (seed_hist[i] / n)], wins_dist=[float(x) for x in (win_hist[i] / n)],
                        boom=float(win_hist[i][int(math.ceil(m + 8)):].sum() / n), bust=float(win_hist[i][:int(math.floor(m - 8)) + 1].sum() / n),
                        sched_wins=round(float(sched_team[i]), 2), shift=round(float(mshift[i]), 2), spread=round(float(sd_team[i]) if sd_team is not None else sd_true, 2)))
    # upcoming games (next 10 days): win probability with today's strengths
    today = pd.Timestamp(datetime.date.today())
    nowS = bw * pre + (1 - bw) * obs_avg
    up = left[(left.date >= today) & (left.date < today + pd.Timedelta(days=10))]
    nowS = nowS + np.where(bw > 0, bw * mshift, 0)
    adj_by = dict(zip(left.index, sched_adj))
    games = [dict(date=str(x.date.date()), home=x.home, away=x.away, p_home=round(float(_ncdf((nowS[TI[x.home]] - nowS[TI[x.away]] + hca_g + adj_by[i]) / sd_game)), 3),
                  spread=round(float(nowS[TI[x.home]] - nowS[TI[x.away]] + hca_g + adj_by[i]), 1), sched=round(float(adj_by[i] + hca_g - hca), 1),
                  why=dict(s_home=round(float(nowS[TI[x.home]]), 2), s_away=round(float(nowS[TI[x.away]]), 2), hca=round(float(hca_g), 2), sd=round(float(sd_game), 1),
                           pre_home=round(float(pre[TI[x.home]] + mshift[TI[x.home]]), 2), pre_away=round(float(pre[TI[x.away]] + mshift[TI[x.away]]), 2),
                           n_home=int(n_g[TI[x.home]]), n_away=int(n_g[TI[x.away]]), **SCHED_PARTS.get(i, {}))) for i, x in zip(up.index, up.itertuples())]
    recent = done.sort_values("date").tail(30)
    meta = dict(sims=n, run=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), schedule=S.source.iloc[0], games_played=int(len(done)), games_left=int(G),
                k=k, hca=hca, sd_game=sd_game, sd_true=sd_true, K=K, kappa=KAPPA, seconds=(datetime.datetime.now() - t0).seconds,
                finals_matchups=sorted([dict(east=a, west=b, p=v / sum(title_pairs.values())) for (a, b), v in title_pairs.items()], key=lambda x: -x["p"])[:12])
    out = dict(meta=meta, teams=res, games=games, recent=[dict(date=str(x.date.date()), home=x.home, away=x.away, hp=int(x.hp), ap=int(x.ap)) for x in recent.itertuples()])
    json.dump(out, open(os.path.join(OUT, f"fc_sim_{SEASON}.json"), "w"), default=float)
    np.savez_compressed(os.path.join(OUT, "fc_sim_players.npz"), avail=np.concatenate(player_out["avail"]), wins=np.concatenate(player_out["wins"]), pids=D.index.values,
                        team_rem=82 - n_g, team_played=n_g, team_of=D.team.map(TI).values)
    D.to_csv(os.path.join(OUT, "fc_players_now.csv"))
    out["meta"]["live"] = live_info
    out["out_now"] = {t: [n for n, a_, tt in zip(D.name, D.avail, D.team) if tt == t and a_ < 0.2] for t in TEAMS}
    print("done", n, "sims in", meta["seconds"], "s", flush=True)
    return out


def apply_live(D, S):
    """season so far: current team from box scores, ratings blended with box-score ratings, availability for the rest
    of the season blended with the share of games played so far; players listed out with an injury for their team's
    last game miss about 8 more games."""
    info = dict(box_games=0)
    if S.source.iloc[0] != "live": return D, info
    import fc_live
    try: B = fc_live.update(S)
    except Exception as e:
        print("live box scores failed:", e); return D, info
    if not len(B): return D, info
    spm = json.load(open(mpath("fc_spm_2025.json")))
    P = fc_live.so_far(B, spm)
    tg = B.groupby("team").gid.nunique()
    out_now = set(B[B.gid.isin(B.sort_values("date").groupby("team").gid.last().values) & (B.played == 0) & B.reason.str.contains("INJURY", na=False)].pid)
    D = D.copy(); D["avail_pre"] = D.avail
    new = [p for p in P.index if p not in D.index]
    if new:
        add = P.loc[new, ["name", "team"]].assign(o=-1.0, d=-1.0, net=-2.0, mpg=P.loc[new, "mpg"], avail=0.85, avail_pre=0.85, rookie=False)
        D = pd.concat([D, add])
    P0 = 3000.0
    for p in P.index:
        if p not in D.index: continue
        x = P.loc[p]; w = x.poss / (P0 + x.poss)
        D.loc[p, "team"] = x.team
        D.loc[p, "o"] = (1 - w) * D.loc[p, "o"] + w * x.o_box; D.loc[p, "d"] = (1 - w) * D.loc[p, "d"] + w * x.d_box
        ng = tg.get(x.team, 0)
        if ng: D.loc[p, "mpg"] = (D.loc[p, "mpg"] * 10 + x.mpg * x.gp) / (10 + x.gp)
    D["net"] = D.o + D.d
    for p in D.index:
        t = D.loc[p, "team"]; ng = tg.get(t, 0)
        if not ng: continue
        gp = P.gp.get(p, 0)
        a = (gp + 20 * D.loc[p, "avail"]) / (ng + 20)
        if p in out_now: a *= max(0.0, (82 - ng - 8) / max(82 - ng, 1))
        D.loc[p, "avail"] = float(np.clip(a, 0.02, 0.97))
    P.to_csv(os.path.join(OUT, "fc_live_players.csv"))
    info = dict(box_games=int(B.gid.nunique()), out_now=len(out_now))
    return D, info


SCHED_PARTS = {}
ALT_IDX = np.array([1.0 if t in ("DEN", "UTA") else 0.0 for t in TEAMS])


def load_extras():
    p = mpath("fc_extras.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def schedule_adjust(S, left, s1):
    """S1: points added to the home team's margin for rest, back-to-backs, 3 games in 4 nights, travel and altitude
    (home court itself is in hca_sched). Also each team's schedule effect in wins over the games left."""
    if not s1.get("pass_") or not len(left): return np.zeros(len(left), np.float32), np.zeros(30)
    import fc_extras
    G = S.copy(); G["gid"] = np.arange(len(G)); G["_i"] = G.index
    F = fc_extras.schedule_features(G).set_index("_i")
    b = s1["coef_all"]
    adj = sum(b[f] * F.loc[left.index, f].values for f in fc_extras.SFEATS)
    global SCHED_PARTS
    SCHED_PARTS = {i: dict(parts={f: round(float(b[f] * F.loc[i, f]), 2) for f in fc_extras.SFEATS if abs(b[f] * F.loc[i, f]) >= 0.05},
                           h=dict(b2b=int(F.loc[i, "raw_h_b2b"]), three4=int(F.loc[i, "raw_h_34"]), rest=int(3 if F.loc[i, "raw_h_r3"] else (1 if F.loc[i, "raw_h_r1"] else (0 if F.loc[i, "raw_h_b2b"] else 2))), mi=round(float(F.loc[i, "raw_h_mi"]) * 1000), tz=int(F.loc[i, "raw_h_tz"])),
                           a=dict(b2b=int(F.loc[i, "raw_a_b2b"]), three4=int(F.loc[i, "raw_a_34"]), rest=int(3 if F.loc[i, "raw_a_r3"] else (1 if F.loc[i, "raw_a_r1"] else (0 if F.loc[i, "raw_a_b2b"] else 2))), mi=round(float(F.loc[i, "raw_a_mi"]) * 1000), tz=int(F.loc[i, "raw_a_tz"])))
                   for i in left.index}
    # schedule effect in wins: change in each game's win chance for an even matchup, summed by team
    from scipy.special import ndtr
    d = ndtr((s1["hca_sched"] + adj) / 13.36) - ndtr(s1["hca_sched"] / 13.36)
    tw = np.zeros(30)
    for (h, a), x in zip(zip(left.home.map(TI), left.away.map(TI)), d): tw[h] += x; tw[a] -= x
    return adj.astype(np.float32), tw


def late_mask(left, n_g):
    """for each remaining game, whether it is among the home / away team's last 15 games."""
    L = np.zeros((len(left), 2), bool)
    if not len(left): return L
    cnt = {}
    order = list(range(len(left)))[::-1]
    hs = left.home.map(TI).values; as_ = left.away.map(TI).values
    for j in order:
        for c, t in ((0, hs[j]), (1, as_[j])):
            cnt[t] = cnt.get(t, 0) + 1
            L[j, c] = cnt[t] <= 15
    return L


def surprise(s3, sd_game):
    """S3: per-team spread (and shift) of season strength around the projection from continuity, age, rookie minutes."""
    if not s3.get("pass_"): return np.zeros(30, np.float32), None
    T = pd.read_csv(mpath(f"fc_teams_{SEASON}.csv"), index_col=0).reindex(TEAMS)
    z = np.column_stack([(T[f].values - s3["z_mu"][f]) / s3["z_sd"][f] for f in ("cont", "age", "rook")])
    c = s3["coef_spread"]; sd_tot = np.exp(c["log_sd"] + z @ np.array([c["cont"], c["age"], c["rook"]]))
    sd_true = np.sqrt(np.maximum(sd_tot ** 2 - sd_game ** 2 / 82, 1.0))
    m = z @ np.array([s3["coef_mean"][f] for f in ("cont", "age", "rook")]) if s3.get("use_mean") and s3.get("coef_mean") else np.zeros(30)
    return m.astype(np.float32), sd_true


def _ncdf(x):
    from scipy.special import ndtr
    return ndtr(x)


def series_final(e, w, key, s, hca, sd_game, rng):
    B = len(e); rows = np.arange(B)
    eh = key[rows, e] >= key[rows, w]
    hi = np.where(eh, e, w); lo = np.where(eh, w, e)
    d = s[rows, hi] - s[rows, lo]
    ph = 1 - _ncdf(-(d + hca) / sd_game); pa = 1 - _ncdf(-(d - hca) / sd_game)
    return np.where(rng.random(B) < series_win(ph, pa), hi, lo)


def lottery_odds(key, member, acc):
    """the 14 teams that miss the playoffs, worst record first; chance of the #1 pick from the league's odds table."""
    k2 = np.where(member, np.inf, key)
    order = np.argsort(k2, axis=1)[:, :14]
    for pos in range(14):
        np.add.at(acc["lottery"], order[:, pos], 1)
        np.add.at(acc["pick1"], order[:, pos], LOTTERY[pos])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sims", type=int, default=1_000_000); ap.add_argument("--chunk", type=int, default=20_000); ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    main(a.sims, a.chunk, a.seed)
