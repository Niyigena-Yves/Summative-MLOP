#!/usr/bin/env bash
# Run the same Locust flood against 1, 2 and 3 API containers behind nginx.
# Usage: bash locust/run_experiments.sh [users] [duration]
set -euo pipefail
cd "$(dirname "$0")/.."
USERS=${1:-50}
DURATION=${2:-60s}
mkdir -p locust/results

for N in 1 2 3; do
  echo "=== $N container(s), $USERS users, $DURATION ==="
  docker compose up -d --scale api=$N
  docker compose restart nginx >/dev/null
  # wait until every replica answers
  for _ in $(seq 1 60); do
    curl -sf localhost:8080/health >/dev/null && break; sleep 2
  done
  sleep 5
  locust -f locust/locustfile.py --host http://localhost:8080 --headless \
    --users "$USERS" --spawn-rate 10 --run-time "$DURATION" \
    --csv "locust/results/containers_${N}" --html "locust/results/containers_${N}.html" --only-summary
done

python locust/summarize.py
