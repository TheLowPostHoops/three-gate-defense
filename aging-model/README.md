# Aging curve model

Predicts a veteran's LEBRON rating next season from this season's numbers and age, and describes how fast players decline by position. Live page: https://thelowposthoops.github.io/three-gate-defense/aging/

- `PREREG.txt` - the seven pass/fail bars, written before the first run.
- `code/model.py` - builds the player-season pairs (age 26 and up), runs the walk-forward test (2016-17 to 2025-26) and scores the bars.
- `code/project.py` - fits the final model, writes the 2026-27 projections and the average age curve.
- `results/` - bar results, the age curve and the 2026-27 projections.

Result: 6 of 7 bars passed. A7 failed: giving guards, forwards and centers separate age curves beat a plain age curve in 5 of 10 test seasons (6 needed). The first run is the only run.

To rerun, put `lebron_master.csv` (BBall Index LEBRON, 2009-10 to 2025-26) in a `data/` folder next to `code/`, then run `python3 code/model.py` and `python3 code/project.py`. Regular season only. Players with fewer than 300 minutes the next season are not scored, so players who retire are missing and the real average decline is larger than shown.
