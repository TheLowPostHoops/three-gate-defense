"""Player ratings for the season forecast (FORECAST_PREREG.txt).

    python3 fc_ratings.py            # ratings for every window, writes out/fc_ratings_<y>.csv

Rating = adjusted plus-minus from lineup stints (offense and defense separately, points per 100 possessions above
average), shrunk toward a box-score prior instead of toward zero. Window: seasons y-2..y with weights 0.25/0.5/1.
"""
import os, pickle, json
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); INP = os.path.join(HERE, "inputs")
FIRST, LAST = 2010, 2025
LAM = 3000.0                      # ridge strength (possession-weighted, points per 100)
LAM_SPM = 1500.0                  # ridge strength when fitting the box prior's target (pure plus-minus)
WIN_W = (0.25, 0.5, 1.0)          # weights for seasons y-2, y-1, y
BOX = ["pts", "fga", "fta", "f3a", "f3m", "ast", "tov", "oreb", "dreb", "stl", "blk"]
_cache = {}


def stints(y):
    if y not in _cache:
        s = pickle.load(open(os.path.join(OUT, f"stints_{y}.pkl"), "rb"))
        s = [x for x in s if str(x[1]).startswith("2") and x[8] + x[9] > 0]     # regular season, real possessions
        _cache[y] = s
    return _cache[y]


def tally(y):
    return pickle.load(open(os.path.join(OUT, f"tally_{y}.pkl"), "rb"))


def season_frame(y):
    """per player: possessions on offense/defense, seconds, box totals."""
    poss = {}
    for x in stints(y):
        H, V, ph, pv = x[3], x[4], x[8], x[9]
        for p in H: a = poss.setdefault(p, [0.0, 0.0]); a[0] += ph; a[1] += pv
        for p in V: a = poss.setdefault(p, [0.0, 0.0]); a[0] += pv; a[1] += ph
    T = tally(y)
    rows = []
    for p, (op, dp) in poss.items():
        t = T.get(p, {})
        rows.append(dict(pid=int(p), season=y, oposs=op, dposs=dp, secs=t.get("sec", 0), **{k: t.get(k, 0) for k in BOX}))
    return pd.DataFrame(rows)


def window_box(y, ww=WIN_W):
    fr = []
    for k, w in zip(range(y - 2, y + 1), ww):
        if w == 0: continue
        if k < FIRST: continue
        f = season_frame(k); f[["oposs", "dposs", "secs"] + BOX] *= w; fr.append(f)
    W = pd.concat(fr).groupby("pid").sum(numeric_only=True).drop(columns="season")
    for k in BOX: W[k + "_100"] = 100 * W[k] / W.oposs.clip(lower=1)
    return W


def design(y, pids, ww=WIN_W):
    idx = {p: i for i, p in enumerate(pids)}; P = len(pids)
    rows, cols, vals, Y, Wt = [], [], [], [], []
    r = 0
    for k, w in zip(range(y - 2, y + 1), ww):
        if k < FIRST or w == 0: continue
        for x in stints(k):
            H, V, pth, ptv, ph, pv = x[3], x[4], x[6], x[7], x[8], x[9]
            for off, de, pts, ps, home in ((H, V, pth, ph, 1.0), (V, H, ptv, pv, 0.0)):
                if ps <= 0: continue
                for p in off:
                    if p in idx: rows.append(r); cols.append(idx[p]); vals.append(1.0)
                for p in de:
                    if p in idx: rows.append(r); cols.append(P + idx[p]); vals.append(1.0)
                rows.append(r); cols.append(2 * P); vals.append(home)
                Y.append(100.0 * pts / ps); Wt.append(ps * w); r += 1
    X = sp.csr_matrix((vals, (rows, cols)), shape=(r, 2 * P + 1))
    return X, np.array(Y), np.array(Wt)


def solve(X, Y, Wt, prior, lam):
    """ridge toward prior: minimize sum w (y - mu - Xb)^2 + lam |b - prior|^2 (home and mean not penalized much)."""
    mu = np.average(Y, weights=Wt)
    yr = Y - mu - X @ prior
    XtW = X.T.multiply(Wt).tocsr()
    A = (XtW @ X).toarray()
    pen = np.full(X.shape[1], lam); pen[-1] = 1.0
    A[np.diag_indices_from(A)] += pen
    b = np.linalg.solve(A, XtW @ yr)
    return prior + b, mu


def ratings(y, spm=None, ww=WIN_W):
    """ratings for the window ending season y. spm = box-prior coefficients (dict o/d -> (intercept, coefs)); None -> zero prior."""
    W = window_box(y, ww)
    W = W[W.oposs > 0]
    pids = list(W.index)
    X, Y, Wt = design(y, pids, ww)
    P = len(pids)
    if spm is None:
        prior = np.zeros(2 * P + 1); lam = LAM_SPM
    else:
        F = W[[k + "_100" for k in BOX]].values
        po = spm["o"][0] + F @ np.array(spm["o"][1]); pd_ = spm["d"][0] + F @ np.array(spm["d"][1])
        prior = np.concatenate([po, pd_, [0.0]]); lam = LAM
    b, mu = solve(X, Y, Wt, prior, lam)
    R = pd.DataFrame(dict(pid=pids, o=b[:P], d=-b[P:2 * P]), index=pids)    # d: positive = good defense
    R["net"] = R.o + R.d; R["poss"] = W.oposs.values; R["secs"] = W.secs.values
    R["home"] = b[-1]; R["mu"] = mu
    for k in BOX: R[k + "_100"] = W[k + "_100"].values
    return R


def fit_spm(windows):
    """box prior: regress pure plus-minus (o, -d) on per-100 box rates, possession weighted, over several windows."""
    fr = []
    for y in windows:
        fp = os.path.join(OUT, f"fc_pure_{y}.pkl")
        if os.path.exists(fp): R = pd.read_pickle(fp)
        else: R = ratings(y, None); R.to_pickle(fp)
        fr.append(R)
    R = pd.concat(fr); R = R[R.poss > 500]
    F = R[[k + "_100" for k in BOX]].values; w = R.poss.values
    out = {}
    for side, yv in (("o", R.o.values), ("d", -R.d.values)):          # d stored as points allowed above average for the fit
        Fm = np.column_stack([np.ones(len(F)), F])
        A = (Fm * w[:, None]).T @ Fm; A[np.arange(1, A.shape[0]), np.arange(1, A.shape[0])] += 1e3
        c = np.linalg.solve(A, (Fm * w[:, None]).T @ yv)
        out[side] = (float(c[0]), [float(v) for v in c[1:]])
    return out


def main():
    import sys
    ys = range(FIRST + 2, LAST + 1)
    for y in ys:
        spm = fit_spm([k for k in ys if k <= y])        # only windows ending by y (all data before the season predicted, y+1)
        R = ratings(y, spm); R.to_csv(os.path.join(OUT, f"fc_ratings_{y}.csv"), index=False)
        json.dump(spm, open(os.path.join(OUT, f"fc_spm_{y}.json"), "w"))
        top = R[R.poss > 3000].sort_values("net", ascending=False).head(5)
        S = ratings(y, spm, (0, 0, 1.0)); S.to_csv(os.path.join(OUT, f"fc_single_{y}.csv"), index=False)
        print(y, len(R), "home", round(R.home.iloc[0], 2), [(int(p), round(n, 1)) for p, n in zip(top.pid, top.net)], flush=True)


if __name__ == "__main__":
    main()
