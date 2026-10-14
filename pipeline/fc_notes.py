"""A short write-up for every team: last season, who came and went, what the model sees.

    python3 fc_notes.py        # writes out/fc_team_notes.json (used by fc_publish.py and the write-up PDF)
"""
import os, json, re
import numpy as np, pandas as pd
import fc_project as FP

HERE = FP.HERE; OUT = FP.OUT; INP = FP.INP
SEASON = 2026
STATIC = os.path.join(HERE, "forecast_model")
CITY = {"ATL": "Atlanta", "BOS": "Boston", "BKN": "Brooklyn", "CHA": "Charlotte", "CHI": "Chicago", "CLE": "Cleveland", "DAL": "Dallas", "DEN": "Denver",
        "DET": "Detroit", "GSW": "Golden State", "HOU": "Houston", "IND": "Indiana", "LAC": "the Clippers", "LAL": "the Lakers", "MEM": "Memphis", "MIA": "Miami",
        "MIL": "Milwaukee", "MIN": "Minnesota", "NOP": "New Orleans", "NYK": "New York", "OKC": "Oklahoma City", "ORL": "Orlando", "PHI": "Philadelphia",
        "PHX": "Phoenix", "POR": "Portland", "SAC": "Sacramento", "SAS": "San Antonio", "TOR": "Toronto", "UTA": "Utah", "WAS": "Washington"}


def mpath(n):
    p = os.path.join(STATIC, n)
    return p if os.path.exists(p) else os.path.join(OUT, n)


def sg(x, d=1):
    return ("+" if x > 0 else "") + f"{x:.{d}f}"


def main():
    A = {int(k): v for k, v in json.load(open(os.path.join(HERE, "coaches.json")))["teams"].items()}
    D = pd.read_csv(mpath(f"fc_players_{SEASON}.csv"), index_col=0)
    sim = json.load(open(os.path.join(OUT, f"fc_sim_{SEASON}.json"))); T = {t["team"]: t for t in sim["teams"]}
    R = FP.ratings(SEASON - 1)
    g = FP.games(SEASON - 1); g["tm"] = g.team.map(A)
    res = FP.results(SEASON - 1)
    rec = {}
    for tid, tm in A.items():
        x = res[(res.home == tid) | (res.away == tid)]
        w = int(((x.home == tid) & (x.hp > x.ap) | (x.away == tid) & (x.ap > x.hp)).sum()); rec[tm] = (w, len(x) - w)
    mov = {A[k]: v for k, v in FP.team_mov(SEASON - 1).items()}
    rank = {t: i + 1 for i, t in enumerate(sorted(mov, key=lambda z: -mov[z]))}
    names = {}
    for k in range(2010, SEASON):
        n = pd.read_csv(os.path.join(OUT, f"names_{k}.csv")).dropna(); names.update(dict(zip(n.pid, n.name)))
    names.update(dict(zip(D.index, D.name)))
    TI = pd.read_csv(mpath(f"fc_teams_{SEASON}.csv"), index_col=0); X3 = json.load(open(mpath("fc_extras.json")))["S3"]
    pc = lambda p: "<0.1%" if p < 0.0005 else (f"{100 * p:.1f}%" if p < 0.1 else f"{round(100 * p)}%")
    played = set()
    for k in range(2010, SEASON): played |= set(FP.games(k).index)
    notes = {}
    for tm in sorted(T):
        t = T[tm]; d = D[D.team == tm].copy(); d["share"] = d.min_share / 240
        last = g[g.tm == tm]
        tot = last.secs.sum()
        # arrivals: on the roster now, played for someone else (or nobody) last season, real minutes projected
        arr = d[~d.index.isin(last.index) & (d.share > 0.035)].sort_values("min_share", ascending=False)
        # departures: 800+ minutes for this team last season, not on the roster now
        dep = last[(last.secs / 60 >= 800) & ~last.index.isin(d.index)].sort_values("secs", ascending=False)
        def val(r, m): return r * m / 48
        arr_val = sum(val(D.loc[p, "net"], D.loc[p, "min_share"]) for p in arr.index)
        dep_val = sum(val(R.net.get(p, -1.0), 240 * last.loc[p, "secs"] / tot) for p in dep.index)
        stars = d.sort_values("min_share", ascending=False).head(8).sort_values("net", ascending=False).head(3)
        w, l = rec[tm]; pw = t["proj_w"]
        lines = []
        lines.append(f"Last season: {w}-{l}, {sg(mov[tm])} points per game ({rank[tm]}{'st' if rank[tm] % 10 == 1 and rank[tm] != 11 else 'nd' if rank[tm] % 10 == 2 and rank[tm] != 12 else 'rd' if rank[tm] % 10 == 3 and rank[tm] != 13 else 'th'} in the league).")
        if len(arr):
            def who(p):
                h = D.loc[p, "how"] if isinstance(D.loc[p, "how"], str) else ""
                m = re.search(r"from (\w{3}) on \d\d/\d\d/(\d\d)", h)
                src = f" from {CITY.get(m.group(1), m.group(1))}" if m and m.group(1) in CITY and int(m.group(2)) >= SEASON % 100 - 1 else ""
                g2 = FP.games(SEASON - 2)
                if p in g2.index and A.get(g2.loc[p, "team"]) == tm: return f"{D.loc[p, 'name']} (back after missing last season, {sg(D.loc[p, 'net'])})"
                if p not in played and p in FP.draft() and FP.draft()[p][0] >= SEASON - 1: return f"{D.loc[p, 'name']} (rookie, {sg(D.loc[p, 'net'])})"
                return f"{D.loc[p, 'name']}{src} ({sg(D.loc[p, 'net'])})"
            lines.append("New faces: " + ", ".join(who(p) for p in arr.index[:4]) + ".")
        if len(dep):
            out = []
            for p in dep.index[:4]:
                dest = D.team.get(p); nm = names.get(p, str(p))
                out.append(f"{nm} ({'to ' + CITY.get(dest, dest) if isinstance(dest, str) else 'unsigned or retired'})")
            lines.append("Gone: " + ", ".join(out) + ".")
        net_move = arr_val - dep_val
        core = ", ".join(f"{names.get(p, p)} ({sg(D.loc[p, 'net'])})" for p in stars.index)
        # the read
        ch = pw - w
        if abs(net_move) >= 1.0:
            mv = f"The moves are worth about {sg(net_move)} points a game on paper"
        else:
            mv = "The roster changes roughly wash out"
        why = []
        if ch >= 6: why.append(f"the model expects a big jump from {w} wins")
        elif ch <= -6: why.append(f"the model expects a step back from {w} wins")
        else: why.append(f"the model sees about the same team as last year's {w} wins")
        if abs(t["shift"]) >= 0.5:
            ti = TI.loc[tm]; c = X3["coef_mean"]
            parts = {f: (ti[f] - X3["z_mu"][f]) / X3["z_sd"][f] * c[f] for f in ("cont", "age", "rook")}
            f = max(parts, key=lambda k: abs(parts[k])); up = t["shift"] > 0
            phrase = {("cont", True): f"{round(100 * ti['cont'])}% of last season's minutes are back, and teams that keep their rotation together tend to beat their projections",
                      ("cont", False): f"only {round(100 * ti['cont'])}% of last season's minutes are back, and rebuilt rotations tend to fall short of their projections",
                      ("rook", False): f"rookies and players with no NBA track record get about {round(100 * ti['rook'])}% of the projected minutes, and teams like that usually fall short of their projections",
                      ("rook", True): "very few minutes go to rookies, which tends to help",
                      ("age", True): f"it is a veteran roster (average age {ti['age']:.1f}), which tends to beat projections slightly",
                      ("age", False): f"it is a young roster (average age {ti['age']:.1f}), which tends to fall short slightly"}
            why.append(phrase.get((f, up), ""))
        if abs(t["sched_wins"]) >= 1: why.append(f"the schedule is worth {sg(t['sched_wins'])} wins, mostly altitude at home")
        sent = f"{mv}, and {why[0]}" + ("; " + "; ".join(why[1:]) if len(why) > 1 else "") + "."
        lines.append(f"Best players by our rating: {core}.")
        lines.append(sent)
        rng = f"{t['w10']}-{t['w90']}"
        lines.append(f"Projection: {pw:.0f} wins (80% range {rng}), {pc(t['playoffs'])} to make the playoffs, {pc(t['title'])} to win the title. Chance of beating the projection by 8+ wins: {round(100 * t['boom'])}%.")
        notes[tm] = dict(text=" ".join(lines[:-1]), projection=lines[-1], last=dict(w=w, l=l, mov=round(mov[tm], 1)), arrivals=[names.get(p, p) for p in arr.index[:4]], departures=[names.get(p, p) for p in dep.index[:4]], moves=round(net_move, 1))
    json.dump(notes, open(os.path.join(OUT, "fc_team_notes.json"), "w"), indent=1)
    return notes


if __name__ == "__main__":
    n = main()
    for t in ("OKC", "SAC"): print(t, n[t]["text"], n[t]["projection"], "\n")
