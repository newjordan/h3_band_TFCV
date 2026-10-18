#!/bin/bash
# Render a solved suite (OUT_DIR/suite.json) as the performance cut: moving cameras that follow the fingers
# (blockout/cameras.py --style fingers), the high-contrast EEVEE look (blender_piano.py --look show: orange hands,
# white / black keys, navy case, grey room; frames stay clean for H3), Salamander piano audio. The MP4 for viewing
# gets the piece titles; the frames in OUT_DIR/frames_show do not.
# Usage: tools/v18/render_show.sh OUT_DIR [JOBS]        (JOBS parallel Blender processes, default 6)
set -e
O=$(realpath "${1:?usage: render_show.sh OUT_DIR [JOBS]}"); J=${2:-6}; REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
P=${PY:-/home/frosty40/h3_band_TFCV/.venv_blockout/bin/python}
SFZ=${SFZ:-/home/frosty40/h3_band_TFCV/models/piano_dl/SalamanderGrandPiano-SFZ+FLAC-V3+20200602/SalamanderGrandPiano-V3+20200602.sfz}
$P -m blockout.cameras $O/suite.json $O/edit_show.json --style fingers > $O/cams_show.log 2>&1
[ -s $O/ks.wav ] || $P -m blockout.sampler $O/notes.json $O/ks.wav --sfz "$SFZ" > $O/sampler.log 2>&1
N=$($P -c "import json;print(len(json.load(open('$O/edit_show.json'))['frames']))"); q=$(( (N+J-1)/J ))
mkdir -p $O/frames_show
for k in $(seq 0 $((J-1))); do a=$((k*q)); b=$((a+q)); [ $b -gt $N ] && b=$N
  HAND_SCALE=1 blender -b --factory-startup -P blockout/blender_piano.py -- $O/edit_show.json $O/frames_show \
    --view edit --look show --res 1920x1080 --frames $a:$b > $O/rs$k.log 2>&1 &
done; wait
[ $(ls $O/frames_show | wc -l) -eq $N ] || { echo "missing frames: $(ls $O/frames_show | wc -l) of $N"; exit 1; }
# titles (video time = suite time - 39.9 s): each piece named for 5 s as it starts
T=$O/titles; mkdir -p $T
printf "Debussy - Clair de Lune" > $T/a.txt
printf "Rachmaninoff - Prelude in C-sharp minor Op. 3 No. 2" > $T/b.txt
printf "Chopin - Nocturne in C-sharp minor Op. posth." > $T/c.txt
F=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
dt() { echo "drawtext=fontfile=$F:textfile=$T/$1.txt:fontsize=40:fontcolor=white:shadowcolor=black@0.7:shadowx=2:shadowy=2:x=64:y=h-110:alpha='if(lt(t,$2),0,if(lt(t,$2+0.8),(t-$2)/0.8,if(lt(t,$2+5.2),1,if(lt(t,$2+6),($2+6-t)/0.8,0))))'"; }
echo "$(dt a 1.0),$(dt b 37.6),$(dt c 120.6)" > $T/filter.txt
ffmpeg -y -loglevel error -framerate 24 -i $O/frames_show/f_%05d.png -ss 39.9 -i $O/ks.wav -filter_script:v $T/filter.txt \
  -c:v libx264 -preset slow -pix_fmt yuv420p -crf 18 -c:a aac -b:a 192k -shortest $O/suite_show.mp4
echo RENDERED
