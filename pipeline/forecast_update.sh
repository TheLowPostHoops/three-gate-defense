#!/usr/bin/env bash
# Daily season forecast: 1,000,000 simulated seasons, award odds, page data.
#   bash pipeline/forecast_update.sh
# Static model files (player projections, test results, voter models) are in pipeline/forecast_model/;
# the schedule, final scores and box scores are downloaded fresh from the NBA's public data feed.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out live
python3 fc_sim.py --sims 1000000
python3 fc_awards.py sim
python3 fc_publish.py
echo "forecast published"
