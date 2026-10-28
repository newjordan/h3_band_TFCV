#!/bin/bash
# queue.sh ID [ID ...]: run lab experiments one after another, then publish the plots
cd /home/frosty40/h3b_wrist
for id in "$@"; do /home/frosty40/h3_band_TFCV/.venv_blockout/bin/python lab/run.py $id > lab/runs/$id.log 2>&1; echo "DONE $id $(date +%H:%M)"; python3 lab/publish.py > /dev/null 2>&1; done
