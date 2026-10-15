# Breakout model

Predicts a young NBA player's LEBRON rating next season from this season's numbers. Live page: https://thelowposthoops.github.io/three-gate-defense/breakout/

- `PREREG.txt` - the seven pass/fail bars, written before the first run.
- `code/model.py` - builds the player-season pairs, runs the walk-forward test (2016-17 to 2025-26) and scores the bars.
- `code/project.py` - fits the final model and writes the 2026-27 projections.
- `results/` - bar results, model weights and the 2026-27 projections.

Result: 5 of 7 bars passed. B4 (jump finder, +0.193 vs +0.20 needed) and B5 (80% ranges covered 85.3% vs an 85% cap) failed. The first run is the only run.

To rerun, put `lebron_master.csv` (BBall Index LEBRON, 2009-10 to 2025-26) and `drafts_2008_2026.csv` (year, pick, player) in a `data/` folder next to `code/`, then run `python3 code/model.py` and `python3 code/project.py`. Regular season only. Players with fewer than 300 minutes the next season are not scored. Draft slot is missing for players drafted before 2008, who are treated as undrafted for the slot feature and given years in the league from their age.
