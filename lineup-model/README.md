# Lineup chemistry model

Tests whether five-man NBA lineups play better or worse than the sum of their players' ratings. Live page: https://thelowposthoops.github.io/three-gate-defense/lineups/

- `PREREG.txt` - the seven pass/fail bars, written before the model was run.
- `data/lineups_2015_2026.csv` - five-man lineups with 100+ possessions, 2014-15 to 2025-26, from the NBA's league lineup table (advanced measures).
- `code/model.py` - builds lineup features from prior-season LEBRON ratings, runs the walk-forward test (2017-18 to 2025-26) and scores the bars.
- `code/project.py` - writes the lineup gap tables and model weights used on the page.
- `results/` - bar results and model weights.

Result: 4 of 7 bars passed. The chemistry model did not beat the plain sum-of-parts line (L1, L2, L4 failed). Lineups with returning players beat their parts in 7 of 9 seasons (L7 passed). The first run is the only run.

To rerun, put `lebron_master.csv` (BBall Index LEBRON) and `index_master.csv` (player, year, team_id) in `data/`, then run `python3 code/model.py` and `python3 code/project.py`. Lineups with under 100 possessions are not in the data, and lineups that reach that mark skew good because coaches keep playing what works.
