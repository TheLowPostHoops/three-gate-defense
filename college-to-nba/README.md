# College to NBA

Scores every 2026 draft pick and the 2027 draft board from college numbers and draft slot, with an 80% range for each player.
Part of Ghost Screen Analytics. Live page: https://thelowposthoops.github.io/three-gate-defense/college/

The rules and pass/fail bars are in `PREREG.txt`. They were written before any model was run.

## Result
Five of seven pre-set bars passed (accuracy, consistency across classes, ranking, not-luck shuffle test, college stats alone). Two failed
(late-pick finds, range honesty). Details in `results/backtest_summary.json`.

A first run missed 8 players whose college names carry Jr. or III (a name-matching gap found when a 2026 rookie showed no college data).
That run passed 3 of 7 (`results/backtest_summary_first_run.json`). The matching was fixed and the run repeated with the bars unchanged;
the second run is the official one. Two bars passed by thin margins (accuracy 5.3% against 5%, college-only 10.1% against 10%).

## Files
- `PREREG.txt`: outcome definition, features, test design, bars.
- `data/barttorvik_prospects.csv`, `data/barttorvik_prospects_suffix.csv`: college seasons from Barttorvik for drafted players and the two draft boards.
- `data/drafts_2008_2026.csv`, `data/draft2026.csv`, `data/board2027.csv`: draft results and the 2027 big board (Tankathon).
- `data/nba_outcomes.csv`: NBA rating (LEBRON, seasons 2 to 4) per drafted player, built by `code/build_sample.py` from a LEBRON export.
- `code/`: `build_sample.py`, `model.py` (backtest and final model), `model_core.py`, `project_2027.py`, `make_site_data.py`.
- `results/`: backtest summary, held-out predictions, coefficients, 2026 rookie and 2027 board projections.

## Limits
Seven test classes and a few hundred players. International and non-college picks are left out of training and get a draft-slot-only projection.
The 2027 board uses last season's college numbers because the 2026-27 season has not been played. Playoffs are not used.
