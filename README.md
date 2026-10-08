# Three Gates: rim protection explorer

Live page: https://thelowposthoops.github.io/three-gate-defense/

The page reads `data/players.json` (every defender) and `data/projection.json` (preseason projections).
A GitHub Action (`.github/workflows/update.yml`) refits the model every Monday and commits fresh data when the
source dataset has new games. `python pipeline/update.py --boot 20` runs a quick local test.

Data: play-by-play and shot data from the open `shufinskiy/nba_data` repository.
