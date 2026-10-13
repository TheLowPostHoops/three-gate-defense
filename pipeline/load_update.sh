#!/usr/bin/env bash
# Rebuild the Load & Durability model and its page data from scratch.
#   bash pipeline/load_update.sh            # used by the weekly GitHub Action and by hand
# Play-by-play is downloaded fresh; inputs pulled by hand (NBA.com tracking and injury reports, rosters, bios,
# college) are read from pipeline/inputs/. Every step writes into pipeline/out/, the last ones into data/.
set -euo pipefail
cd "$(dirname "$0")"
LIVE=$(python3 -c "import datetime as d; t=d.date.today(); print(t.year if t.month>=10 else t.year-1)")
python3 load_games.py 2010 "$LIVE" --force-current     # completed seasons are cached between runs; the live one is always rebuilt
python3 load_model.py
python3 load_team.py
python3 load_extra.py
python3 load_publish.py
python3 load_v2_features.py
python3 load_v2.py
python3 load_playoffs.py
python3 load_v2_publish.py
echo "load model rebuilt"
