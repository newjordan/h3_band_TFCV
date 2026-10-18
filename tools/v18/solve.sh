#!/bin/bash
# Solve the Knight's Suite (Clair de Lune -> Rachmaninoff Op. 3/2 -> Chopin Nocturne in C# minor Op. posth., score
# 39.9-390.5 s) with the code defaults (V3 since 2026-10-17 evening, lab/NOTEBOOK.md) on the regular MPFB2 hand at full size. 24 overlapping parts in parallel, then stitched.
# Usage: tools/v18/solve.sh OUT_DIR        (~50 min on 20 cores)
set -e
O=$(realpath -m "${1:?usage: solve.sh OUT_DIR}"); mkdir -p "$O"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
P=${PY:-/home/frosty40/h3_band_TFCV/.venv_blockout/bin/python}
M=${MIDI_DIR:-/home/frosty40/h3b_suite/examples/piano}
cd "$REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HAND_SCALE=1
export PIANO_OVR=${PIANO_OVR-}     # V3 settings are the code defaults; set this only for experiments
END=390.5; N=24
for k in $(seq 0 $((N-1))); do
  a=$(python3 -c "print(round(39.9+($END-39.9)*$k/$N,3))"); b=$(python3 -c "print(round(39.9+($END-39.9)*($k+1)/$N,3))")
  s=$(python3 -c "print(max(39.9,$a-1.0))"); d=$(python3 -c "print(round($b-max(39.9,$a-1.0),3))")
  $P -m blockout.piano $M/knights_suite.mid $O/part_$k.json --feeling $M/knights_suite.feeling.json --start $s --dur $d > $O/part_$k.log 2>&1 &
done
wait; echo parts done
$P -m blockout.piano $M/knights_suite.mid $O/full_notes_dummy.json --feeling $M/knights_suite.feeling.json --start 250 --dur 0.5 --notes-out $O/notes.json > /dev/null 2>&1
$P tools/v18/stitch.py "$O" && echo stitched
