# Ghost Screen Analytics

Live site: https://thelowposthoops.github.io/three-gate-defense/

- `index.html` home page, `models/` list of models, `assets/site.css` shared masthead and navigation
- `defense/` Three Gates rim protection explorer (moved here from the root)
- `durability/` Load & Durability model

## Load & Durability

Rules and every test result: `pipeline/LOAD_PREREG.txt`. Rebuild once a year, after the playoffs:

    python pipeline/load_games.py 2010 2025   # player-game tables from play-by-play
    python pipeline/load_model.py             # features and tests L1-L3
    python pipeline/load_team.py              # team tests TM1-TM3
    python pipeline/load_extra.py             # college (C1) and miles (L6) tests
    python pipeline/load_publish.py           # version 1 page data (kept in pipeline/out for version 2)
    python pipeline/load_v2_features.py       # travel, contact exposure, body size
    python pipeline/load_v2.py                # official injury reports as labels, tests V0-V7
    python pipeline/load_v2_publish.py        # data/load.json, data/load_teams.json, data/load_meta.json

`pipeline/inputs/` (not in the repo; pull again before a rebuild) holds files pulled by hand through a browser:
`miles_seasons.csv` from NBA.com speed/distance (leaguedashptstats, PtMeasureType=SpeedDistance, every season from
2013-14) and `college_seasons.csv` from Bart Torvik's player files (getadvstats.php, 2008 on), because those sites
block scripts. Version 2 adds `injury_reports.csv` (the NBA's official injury report PDFs from
ak-static.cms.nba.com, parsed in the browser with pdf.js; 2021-22 through Dec 21, 2025, after which NBA.com's firewall
refuses the files), `rosters_2026.json` (team pages on nba.com) and `bio_players.json` (player pages on nba.com). `pipeline/coaches.json` is the head-coach table.

## Three Gates

The page reads `data/players.json` (every defender) and `data/projection.json` (preseason projections).
A GitHub Action (`.github/workflows/update.yml`) refits the model every Monday and commits fresh data when the
source dataset has new games. `python pipeline/update.py --boot 20` runs a quick local test.

Data: play-by-play and shot data from the open `shufinskiy/nba_data` repository.
