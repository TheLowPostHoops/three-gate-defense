"""Write the forecast page data: data/forecast.json and data/forecast_history.json.

    python3 fc_publish.py
"""
import os, json, datetime
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); REPO = os.path.dirname(HERE); PUB = os.path.join(REPO, "data")
SEASON = 2026
STATIC = os.path.join(HERE, "forecast_model")


def mpath(name):
    p = os.path.join(STATIC, name)
    return p if os.path.exists(p) else os.path.join(OUT, name)


def surname(n):
    parts = [w for w in str(n).split() if w.rstrip(".") not in ("Jr", "Sr", "II", "III", "IV")]
    return parts[-1] if parts else n


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def main():
    sim = json.load(open(os.path.join(OUT, f"fc_sim_{SEASON}.json")))
    aw = json.load(open(os.path.join(OUT, f"fc_awards_{SEASON}.json")))
    bt = json.load(open(mpath("fc_backtest.json"))); at = json.load(open(mpath("fc_awards_test.json")))
    P = pd.read_csv(os.path.join(OUT, "fc_players_now.csv"), index_col=0)
    names = json.load(open(os.path.join(HERE, "coaches.json")))
    # rosters: rotation by minutes share
    rost = {}
    for t, g in P.groupby("team"):
        g = g.sort_values("mpg", ascending=False).head(12)
        rost[t] = [dict(name=x["name"], pid=int(p), mpg=r(x.mpg, 1), o=r(x.o, 1), d=r(x.d, 1), net=r(x.net, 1), avail=r(x.avail, 2), rookie=bool(x.get("rookie", False) is True)) for p, x in g.iterrows()]
    top = P[P.mpg >= 15].sort_values("net", ascending=False).head(50)
    players = [dict(name=x["name"], pid=int(p), team=x.team, o=r(x.o, 1), d=r(x.d, 1), net=r(x.net, 1), mpg=r(x.mpg, 1), avail=r(x.avail, 2)) for p, x in top.iterrows()]
    s = bt["summary"]
    tests = [
        dict(id="F1", q="Team strength: preseason point differential", bar="beat last season x 0.6 by 5%+", result=f"error {s['F1']['mae_model']:.2f} vs {s['F1']['mae_base']:.2f} points per game ({100 * s['F1']['gain']:.0f}% better)", pass_=s["F1"]["pass_"]),
        dict(id="F2", q="Picking every game's winner before the season", bar="more right and better log loss than the baseline", result=f"{100 * s['F2']['acc_model']:.1f}% right vs {100 * s['F2']['acc_base']:.1f}%", pass_=s["F2"]["pass_"]),
        dict(id="F3", q="Win probabilities mean what they say", bar="every tenth within 5 points", result=f"largest miss {100 * s['F3']['max_gap']:.1f} points", pass_=s["F3"]["pass_"]),
        dict(id="F4", q="Updating with results during the season", bar="beat preseason-only and results-only", result=f"blend after K={bt['F4']['K']} games: log loss {bt['F4']['test']['blend']:.3f} vs {bt['F4']['test']['pre']:.3f} and {bt['F4']['test']['res']:.3f}", pass_=bt["F4"]["pass_"]),
    ]
    for a, lab in (("mvp", "MVP"), ("dpoy", "Defensive Player of the Year"), ("roy", "Rookie of the Year"), ("smoy", "Sixth Man"), ("mip", "Most Improved")):
        x = at[a]
        tests.append(dict(id="A1" if a == "mvp" else "A1·" + a, q=f"{lab} voter model picks the winner", bar="4 of 6 seasons (2020-21 to 2025-26)" if a == "mvp" else "reported, no bar",
                          result=f"{x['hits']} of {x['of']}: " + ", ".join(f"{z['season'][2:]} {surname(z['actual'])}{'' if z['hit'] else ' (picked ' + surname(z['model_top']) + ')'}" for z in x["seasons"]),
                          pass_=x["pass_"] if a == "mvp" else None))
    X = json.load(open(mpath("fc_extras.json")))
    S1, S2, S3, S4 = X["S1"], X["S2"], X["S3"], X["S4"]
    tests += [
        dict(id="S1", q="Schedule: rest, back-to-backs, 3 games in 4 nights, travel, altitude", bar="game picks 0.2% better (log loss)", result=f"{100 * S1['gain']:.2f}% better. Second night of a back-to-back costs {abs(S1['coef_all']['b2b']):.1f} points; Denver and Utah get {S1['coef_all']['alt']:.1f} extra at home", pass_=S1["pass_"]),
        dict(id="S2", q="Projected health moves a team's expected strength", bar="team error 2% lower", result=f"{100 * S2['gain']:+.1f}%: injury luck is simulated, but not as a preseason penalty", pass_=S2["pass_"]),
        dict(id="S3", q="Surprise factor: some teams' outcomes are harder to call", bar="better score than one spread for every team", result="spread and shift set by returning minutes, age and rookie minutes", pass_=S3["pass_"]),
        dict(id="S4a", q="Teams out of the race play worse late", bar="1.5+ points, range excludes 0", result=f"{S4['tank']['shift']:+.1f} points per game in the last 15 ({S4['tank']['lo']:+.1f} to {S4['tank']['hi']:+.1f})", pass_=S4["tank"]["pass_"]),
        dict(id="S4b", q="Teams with their seed locked play worse late", bar="1.5+ points, range excludes 0", result=f"{S4['lock']['shift']:+.1f} points per game in the last 15 ({S4['lock']['lo']:+.1f} to {S4['lock']['hi']:+.1f})", pass_=S4["lock"]["pass_"]),
    ]
    TI = pd.read_csv(mpath(f"fc_teams_{SEASON}.csv"), index_col=0)
    team_inputs = {t: dict(cont=r(TI.loc[t, "cont"], 3), age=r(TI.loc[t, "age"], 1), rook=r(TI.loc[t, "rook"], 3)) for t in TI.index}
    x = at["coy"]
    tests.append(dict(id="A1·coy", q="Coach of the Year (by team)", bar="reported, no bar", result=f"{x['hits']} of {x['of']}", pass_=None))
    out = dict(meta=dict(sim["meta"], season="2026-27", updated=datetime.date.today().isoformat()), teams=sim["teams"], games=sim["games"], recent=sim["recent"],
               awards={k: aw[k] for k in ("mvp", "dpoy", "roy", "smoy", "mip", "coy")}, games65=aw["games65"], award_sims=aw["sims"],
               rosters=rost, players=players, tests=tests, team_inputs=team_inputs, backtest=dict(seasons="2015-16 to 2025-26", calib=bt["F3"]))
    os.makedirs(PUB, exist_ok=True)
    json.dump(out, open(os.path.join(PUB, "forecast.json"), "w"), separators=(",", ":"), allow_nan=False)
    # history: one snapshot per day
    hp = os.path.join(PUB, "forecast_history.json")
    H = json.load(open(hp)) if os.path.exists(hp) else []
    today = datetime.date.today().isoformat()
    H = [h for h in H if h["date"] != today]
    H.append(dict(date=today, gp=sim["meta"]["games_played"], t={t["team"]: [r(t["proj_w"], 1), r(t["playoffs"], 3), r(t["title"], 4)] for t in sim["teams"]},
                  mvp=[[a["name"], r(a["p"], 3)] for a in aw["mvp"][:8]]))
    json.dump(H, open(hp, "w"), separators=(",", ":"))
    print("published forecast", len(sim["teams"]), "teams,", len(H), "days of history")


if __name__ == "__main__":
    main()
