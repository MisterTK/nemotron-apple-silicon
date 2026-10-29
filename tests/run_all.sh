#!/bin/bash
# Runs every test with default (short) settings. About 3-4 hours. Each script also runs on its own.
cd "$(dirname "$0")" || exit 1
for t in 00_env 10_single_user 20_concurrency 30_scenarios 40_mlx 50_quality 60_serving; do
  echo; echo "######## $t $(date +%T)"; bash $t.sh || echo "!! $t failed"
done
