#!/bin/bash
# queue.sh ID [ID ...]: run lab experiments one after another, then publish the plots.
# Uses this checkout. H3B_LAB_PYTHON selects the interpreter (default: python3).
cd "$(dirname "$0")/.."
PY="${H3B_LAB_PYTHON:-python3}"
mkdir -p lab/runs
for id in "$@"; do
  "$PY" lab/run.py "$id" > "lab/runs/$id.log" 2>&1
  echo "DONE $id $(date +%H:%M)"
  "$PY" lab/publish.py > /dev/null 2>&1 || true
done
