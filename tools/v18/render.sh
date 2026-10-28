#!/bin/bash
# Render a solved suite (OUT_DIR/suite.json) with the still keyboard edit, piano audio from the Salamander sampler.
# Usage: tools/v18/render.sh OUT_DIR
set -e
O=$(realpath "${1:?usage: render.sh OUT_DIR}"); REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
P=${PY:-/home/frosty40/h3_band_TFCV/.venv_blockout/bin/python}
SFZ=${SFZ:-/home/frosty40/h3_band_TFCV/models/piano_dl/SalamanderGrandPiano-SFZ+FLAC-V3+20200602/SalamanderGrandPiano-V3+20200602.sfz}
$P -m blockout.cameras $O/suite.json $O/edit.json > $O/cams.log 2>&1
$P -m blockout.sampler $O/notes.json $O/ks.wav --sfz "$SFZ" > $O/sampler.log 2>&1
N=$($P -c "import json;print(len(json.load(open('$O/edit.json'))['frames']))"); q=$(( (N+9)/10 ))
mkdir -p $O/frames
for k in $(seq 0 9); do a=$((k*q)); b=$((a+q)); [ $b -gt $N ] && b=$N; HAND_SCALE=1 blender -b --factory-startup -P blockout/blender_piano.py -- $O/edit.json $O/frames --view edit --frames $a:$b > $O/r$k.log 2>&1 & done; wait
ffmpeg -y -loglevel error -framerate 24 -i $O/frames/f_%05d.png -ss 39.9 -i $O/ks.wav -c:v libx264 -pix_fmt yuv420p -crf 20 -c:a aac -b:a 192k -shortest $O/suite_grey.mp4
echo RENDERED
