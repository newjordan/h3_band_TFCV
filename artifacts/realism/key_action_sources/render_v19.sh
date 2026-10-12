#!/bin/bash
O=/tmp/claude-1000/suite_v19; cd /home/frosty40/h3b_wrist; P=/home/frosty40/h3_band_TFCV/.venv_blockout/bin/python
$P -m blockout.cameras $O/ks_v19.json $O/edit.json > $O/cams.log 2>&1
$P -m blockout.sampler $O/notes.json $O/ks.wav --sfz "/home/frosty40/h3_band_TFCV/models/piano_dl/SalamanderGrandPiano-SFZ+FLAC-V3+20200602/SalamanderGrandPiano-V3+20200602.sfz" > $O/sampler.log 2>&1
N=$($P -c "import json;print(len(json.load(open('$O/edit.json'))['frames']))"); q=$(( (N+9)/10 ))
mkdir -p $O/frames
for k in $(seq 0 9); do a=$((k*q)); b=$((a+q)); [ $b -gt $N ] && b=$N; blender -b --factory-startup -P blockout/blender_piano.py -- $O/edit.json $O/frames --view edit --frames $a:$b > $O/r$k.log 2>&1 & done; wait
ffmpeg -y -loglevel error -framerate 24 -i $O/frames/f_%05d.png -ss 39.9 -i $O/ks.wav -c:v libx264 -pix_fmt yuv420p -crf 20 -c:a aac -b:a 192k -shortest $O/suite_grey_v19.mp4
echo RENDERED
